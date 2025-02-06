# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType

import torch
from torch import nn


def load_state_dict_biencoder(self, state_dict, strict=True, assign=False):
    """
    When loading a uni-encoder checkpoint into a bi-encoder architecture,
    we need a custom load function that copies the encoder state to the question & context encoders
    """
    if strict:
        unexpected_keys = list(
            filter(
                lambda k: k.find("question_encoder") == -1
                and k.find("ctx_encoder") == -1,
                state_dict.keys(),
            ),
        )
        if len(unexpected_keys) > 0:
            raise RuntimeError(
                f'Unexpected key(s) in state_dict: {unexpected_keys}.'
            )
    # If state dict is missing context encoder,
    # we copy question encoder's dict to context
    if all(k.find("ctx_encoder") == -1 for k in state_dict.keys()):
        ctx_state_dict = {
            k.replace("question_encoder.", "ctx_encoder."): v
            for k, v in state_dict.items()
            if k.find("question_encoder") >= 0
        }
        for key, value in ctx_state_dict.items():
            state_dict[key] = value
    return torch.nn.Module.load_state_dict(
        self, state_dict, strict=strict, assign=assign
    )


def mean_pooling(token_embeddings, attention_mask):
    """
    Jina uses mean pooling over all token embeddings to form sentence embeddings.
    https://huggingface.co/jinaai/jina-embeddings-v2-base-en#why-mean-pooling
    https://huggingface.co/jinaai/jina-bert-implementation/blob/f3ec4cf7de7e561007f27c9efc7148b0bd713f81/modeling_bert.py#L1166

    Args:
        token_embeddings: (batch_size, MSL, hidden_size)
                        all output token embeddings.

        attention_mask: (batch_size, MSL),
                        attention mask to decide effective token embeddings
                        to be averaged, drop the ones that are not attended

    Returns:
        averaged_embedding: (batch_size, hidden_size)
                        final sentence embedding

    For all input and output, for context, batch_size will be multiplied by num_context
    """
    original_dtype = token_embeddings.dtype
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return (
        torch.sum(token_embeddings * input_mask_expanded, 1)
        / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    ).to(original_dtype)


class DPRModel(nn.Module):
    def __init__(
        self,
        question_encoder,
        ctx_encoder,
        pooler_type,
        mutual_information,
    ):
        super().__init__()
        self.question_encoder = question_encoder
        self.ctx_encoder = ctx_encoder
        self.pooler_type = pooler_type
        self.mutual_information = mutual_information

        # When loading a uni-encoder checkpoint into a bi-encoder architecture,
        # we need a custom load function that duplicates the encoder state
        if self.ctx_encoder:
            custom_load_state_dict_fn = load_state_dict_biencoder
            self.load_state_dict = MethodType(custom_load_state_dict_fn, self)

    def flatten_ctx_input(self, context_tensor):
        """
        Args:
            context_tensor: (batch_size, num_context, MSL),
                            output from DPRHDF5DataProcessor

        Returns:
            flatten_tensor: (batch_size x num_context, MSL)
        """
        flatten_tensor = context_tensor.view(-1, context_tensor.shape[-1])
        return flatten_tensor

    def forward(
        self,
        questions_input_ids,
        questions_attention_mask,
        questions_token_type_ids,
        ctx_input_ids,
        ctx_attention_mask,
        ctx_token_type_ids,
    ):
        """
        Vanilla question-document contrastive training has single question-document
        pairs that denote the correct document that should be retrieved for each
        question. The other question's documents are used as in-batch negatives,
        to help the model learn what should *not* be close to a question.

        Papers (DPR, Jina, etc.) have shown that it is beneficial to also choose
        *hard-negatives* for each question. These are chosen to be similar to the
        goal document, but still *not* the correct document for a question. This
        is to help learn to separate the best document from similar but distractor
        documents. To include 'num_context' hard-negatives for each question, we need
        the tensors storing documents/contexts to have 'num_context' times as many examples
        as the tensors storing questions. For each element in a batch, we will
        have one question and (num_context + 1) documents
        (1 positive and num_context hard-negatives).

        For questions, we use tensors of shape:
            [batch_size, max_sequence_length]

        For contexts/documents.
            [batch_size, num_context, max_sequence_length]

        The context tensor at element i contains one positive and num_context
        hard-negative documents for the question at element i. The batch dimension
        gives us perfect correspondence between questions and their documents
        -- however this introduces interleaved positive and negative documents if flattened.

        The usual contrastive loss formulation has a [batch_size x batch_size]
        tensor where the diagonal entries correspond are the positive matches.
        If you had a document tensor with shape [batch_size, max_sequence_length]
        of only positive documents, this would be the case. Even with hard-negatives,
        people will often keep the positive entries in the first `batch_size`
        columns of the [batch_size x num_context*batch_size] loss tensor.

        For simplicity of data-processing logic and avoidance of expensive redist
        operations, we use the interleaved documents which requires the labels to be the
        even entries [0, num_context, 2*num_context, ..., batch_size*num_context]
        rather than the first [0, ..., batch_size] entries.
        """
        batch_size = questions_input_ids.shape[0]
        num_context = ctx_input_ids.shape[1]
        ctx_input_ids = self.flatten_ctx_input(ctx_input_ids)
        ctx_attention_mask = self.flatten_ctx_input(ctx_attention_mask)
        ctx_token_type_ids = self.flatten_ctx_input(ctx_token_type_ids)

        q_embds, q_embds_pooled = self.question_encoder(
            input_ids=questions_input_ids,
            attention_mask=questions_attention_mask,
            segment_ids=questions_token_type_ids,
        )

        # Uniencoder (e.g., jina) and Biencoder (e.g., dpr) choices
        if self.ctx_encoder:
            ctx_embds, ctx_embds_pooled = self.ctx_encoder(
                input_ids=ctx_input_ids,
                attention_mask=ctx_attention_mask,
                segment_ids=ctx_token_type_ids,
            )
        else:
            # When loading initial checkpoing weights for uniencoder,
            # we only load it once for question_encoder and do not load it again
            # for ctx_encoder
            ctx_embds, ctx_embds_pooled = self.question_encoder(
                input_ids=ctx_input_ids,
                attention_mask=ctx_attention_mask,
                segment_ids=ctx_token_type_ids,
            )

        # Pooling layer choices:
        # (1) mean pooling across hidden representations for all tokens.
        # (2) directly use CLS hidden representation
        # (3) use FFN pooler on-top of CLS token
        assert (
            self.pooler_type == "mean"
            or self.pooler_type == "cls"
            or self.pooler_type == "ffn_pooler"
        ), f"Only mean or cls or ffn_pooler  is supported  for now, but got {self.pooler_type}"

        if self.pooler_type == "mean":
            q_embds_pooled = mean_pooling(q_embds, questions_attention_mask)
            ctx_embds_pooled = mean_pooling(ctx_embds, ctx_attention_mask)
        elif self.pooler_type == "cls":
            if not self.question_encoder.pooler:
                q_embds_pooled = q_embds[:, 0]
            # (1) Uniencoder with cls (2) biencoder with cls
            if not self.ctx_encoder or not self.ctx_encoder.pooler:
                ctx_embds_pooled = ctx_embds[:, 0]
        else:
            pass

        if self.mutual_information:
            ctx_embeds_for_c2q = ctx_embds_pooled.view(
                batch_size, num_context, -1
            )[:, 0, :]
            c2q_scores = ctx_embeds_for_c2q @ q_embds_pooled.T
        else:
            c2q_scores = None

        q2c_scores = q_embds_pooled @ ctx_embds_pooled.T
        return (q2c_scores, c2q_scores, q_embds_pooled, ctx_embds_pooled)
