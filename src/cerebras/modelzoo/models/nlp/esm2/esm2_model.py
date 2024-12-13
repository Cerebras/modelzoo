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

from typing import Optional

import torch

from cerebras.modelzoo.models.nlp.bert.bert_model import (
    BertModel,
    BertModelConfig,
)


class Esm2ModelConfig(BertModelConfig):
    token_dropout: bool = False
    mask_token_id: Optional[int] = None
    norm_first: bool = True

    def post_init(self, context):
        super().post_init(context)

        if self.token_dropout and self.mask_token_id is None:
            raise ValueError(
                "mask_token_id parameter must be provided when token_dropout is "
                "enabled."
            )


class Esm2Model(BertModel):
    def __init__(self, config: Esm2ModelConfig):
        super().__init__(config)

        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    # ESM-2 may use token dropout. This function overrides the default BERT
    # embedding computation, and is used in the forward() fn.
    def compute_input_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        segment_ids=None,
    ):
        if not self.token_dropout:
            embeddings = self.embedding_layer(
                input_ids,
                position_ids=position_ids,
                segment_ids=segment_ids,
            )
        else:
            embeddings = self.embedding_layer.compute_token_embeddings(
                input_ids
            )

            embeddings.masked_fill_(
                (input_ids == self.mask_token_id).unsqueeze(-1), 0.0
            )
            mask_ratio_train = (
                0.15 * 0.8
            )  # Hardcoded as the ratio used in all ESM model training runs
            batch_size, seq_length, hidden_size = embeddings.shape
            src_lengths = torch.sum(attention_mask.to(embeddings.dtype), dim=-1)
            num_masked = torch.sum(
                (input_ids == self.mask_token_id).to(embeddings.dtype), dim=-1
            )
            one_minus_mask_ratio_observed = 1 - num_masked / src_lengths
            # Decompose one two-dimensional broadcastOp to two single dimension broadcastOps,
            # due to layout limitation for two-dimensional broadcastOp.
            # First broadcast [batch_size] to [batch_size, seq_length]
            # Then broadcast [batch_size, seq_length] to [batch_size, seq_length, hidden_size]
            broadcasted_mask_ratio = one_minus_mask_ratio_observed.unsqueeze(
                -1
            ).expand(batch_size, seq_length)
            broadcasted_mask_ratio = broadcasted_mask_ratio.unsqueeze(
                -1
            ).expand(batch_size, seq_length, hidden_size)
            embeddings = (
                embeddings * (1 - mask_ratio_train) / broadcasted_mask_ratio
            ).to(embeddings.dtype)

            if self.embedding_layer.position_embeddings is not None:
                assert (
                    self.embedding_layer.embedding_size
                    == self.embedding_layer.positional_embedding_size
                ), "embedding size and positional embedding size should be same"
                embeddings = (
                    embeddings
                    + self.embedding_layer.compute_positional_embeddings(
                        input_ids, position_ids, 0, embeddings.dtype
                    )
                )
            if (
                segment_ids is not None
                and self.embedding_layer.segment_embeddings is not None
            ):
                assert (
                    self.embedding_layer.embedding_size
                    == self.embedding_layer.segment_embedding_size
                ), "embedding size and segment embedding should be same"
                embeddings = (
                    embeddings
                    + self.embedding_layer.compute_segment_embeddings(
                        segment_ids
                    )
                )
        return embeddings
