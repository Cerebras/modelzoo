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

import math

from torch import nn


class DPRModel(nn.Module):
    def __init__(self, q_encoder, ctx_encoder, hidden_size, scale_similarity):
        super(DPRModel, self).__init__()
        self.question_encoder = q_encoder
        self.ctx_encoder = ctx_encoder
        self.hidden_size = hidden_size
        self.scale_similarity = scale_similarity

    def get_similarity_scores(self, q_embds, ctx_embds):
        """
        Args:
            q_embds: (batch_size, hidden_size)
            ctx_embds: (batch_size x num_contexts, hidden_size)

        Returns:
            scores: (batch_size, batch_size x num_contexts)
        """
        # we use dot-product as other distances were not found to be better
        if self.scale_similarity:
            scores = q_embds @ ctx_embds.T / math.sqrt(self.hidden_size)
        else:
            scores = q_embds @ ctx_embds.T
        return scores

    def flatten_ctx_input(self, context_tensor):
        """
        Args:
            context_tensor: (batch_size, num_contexts, MSL),
                            output from DPRHDF5DataProcessor

        Returns:
            flatten_tensor: (batch_size x num_contexts, MSL)
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
        q_embds, q_embds_pooled = self.question_encoder(
            input_ids=questions_input_ids,
            attention_mask=questions_attention_mask,
            segment_ids=questions_token_type_ids,
        )

        """
        Vanilla question-document contrastive training has single question-document 
        pairs that denote the correct document that should be retrieved for each 
        question. The other question's documents are used as in-batch negatives, 
        to help the model learn what should *not* be close to a question.

        Papers (DPR, Jina, etc.) have shown that it is beneficial to also choose 
        *hard-negatives* for each question. These are chosen to be similar to the
        goal document, but still *not* the correct document for a question. This is 
        to help learn to separate the best document from similar but distractor
        documents. To include hard-negatives for each question, we need the tensors 
        storing documents/contexts to have twice as many examples as the tensors 
        storing questions. For each element in a batch, we will have one question, 
        and two documents (positive and hard-negative). 
        
        For questions, we use tensors of shape: 
            [batch_size, max_sequence_length] 
        
        For contexts/documents.
            [batch_size, num_context, max_sequence_length] 
        
        
        The context tensor at element i contains the positive and hard-negative 
        document for the question at element i. The batch dimension gives us 
        perfect correspondence between questions and their documents -- however 
        this introduces interleaved positive and negative documents if flattened.

        The usual contrastive loss formulation has a [batch_size x batch_size] 
        tensor where the diagonal entries correspond are the positive matches. 
        If you had a document tensor with shape [batch_size, max_sequence_length] 
        of only positive documents, this would be the case. Even with hard-negatives,
        people will often keep the positive entries in the first `batch_size` 
        columns of the [batch_size x 2*batch_size] loss tensor. For simplicity 
        of data-processing logic and avoidance of expensive redist operations, 
        we use the interleaved documents which requires the labels to be the 
        even entries [0, 2, 4, ...] rather than the first [0, ..., B] entries. 
        """

        ctx_input_ids = self.flatten_ctx_input(ctx_input_ids)
        ctx_attention_mask = self.flatten_ctx_input(ctx_attention_mask)
        ctx_token_type_ids = self.flatten_ctx_input(ctx_token_type_ids)

        ctx_embds, ctx_embds_pooled = self.ctx_encoder(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask,
            segment_ids=ctx_token_type_ids,
        )
        # User can specify to not use FFN pooler on-top of CLS token, and
        # directly use CLS hidden representation otherwise
        if not self.question_encoder.pooler:
            q_embds_pooled = q_embds[:, 0]

        if not self.ctx_encoder.pooler:
            ctx_embds_pooled = ctx_embds[:, 0]

        scores = self.get_similarity_scores(q_embds_pooled, ctx_embds_pooled)
        return (scores, q_embds_pooled, ctx_embds_pooled)
