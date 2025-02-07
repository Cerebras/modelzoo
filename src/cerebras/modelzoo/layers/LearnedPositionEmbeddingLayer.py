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

import torch
import torch.nn as nn

from cerebras.modelzoo.layers.create_initializer import create_initializer


class LearnedPositionEmbeddingLayer(nn.Module):
    def __init__(
        self,
        max_position_embeddings=None,
        positional_embedding_size=None,
        pad_token_id=None,
        position_embedding_offset=0,
        position_embeddings_initializer='uniform',
        mask_padding_in_positional_embed=False,
        device=None,
    ) -> None:
        super(LearnedPositionEmbeddingLayer, self).__init__()

        num_pos_embeddings = max_position_embeddings + position_embedding_offset
        if mask_padding_in_positional_embed:
            num_pos_embeddings += (
                1  # Corresponds to extra embedding for padding token
            )

        if num_pos_embeddings >= 2**16:
            raise ValueError(
                "`num_pos_embeddings` over 65535 is not supported for learned position embedding."
            )
        self.embed = nn.Embedding(
            num_pos_embeddings,
            positional_embedding_size,
            device=device,
        )

        self.position_embeddings_initializer = position_embeddings_initializer
        self.position_embedding_offset = position_embedding_offset
        self.pad_token_id = pad_token_id
        self.mask_padding_in_positional_embed = mask_padding_in_positional_embed

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        create_initializer(self.position_embeddings_initializer)(
            self.embed.weight.data
        )

        if self.pad_token_id:
            self.embed.weight.data[self.embed.padding_idx].zero_()

    def forward(self, input_ids, position_ids=None, past_length=0, dtype=None):
        return self.compute_positional_embeddings(
            input_ids, position_ids, past_length, dtype
        )

    def compute_positional_embeddings(
        self, input_ids, position_ids=None, past_length=0, dtype=None
    ):
        import cerebras.pytorch as cstorch

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = "cpu" if cstorch.use_cs() else input_ids.device

        if position_ids is not None:
            assert (
                position_ids.size() == input_shape
            ), "position_ids must have shape [batch_size, seq_length]"

        if position_ids is None:
            if not self.mask_padding_in_positional_embed:
                position_ids = (
                    cstorch.make_constant(
                        torch.arange(
                            past_length,
                            input_shape[-1] + past_length,
                            device=device,
                        )
                    )
                    + self.position_embedding_offset
                ).expand((batch_size, -1))
            else:
                mask = input_ids.ne(self.pad_token_id)
                unmasked_position_ids = cstorch.make_constant(
                    torch.arange(
                        past_length + 1,
                        input_shape[-1] + past_length + 1,
                        device=device,
                    )
                )
                # WARNING: the following line assumes that padding tokens
                # always appear at the end of a sequence.
                position_ids = (
                    torch.where(mask, unmasked_position_ids, 0)
                    + self.position_embedding_offset
                )
                position_ids = position_ids.expand((batch_size, -1))

        position_embeddings = self.embed(position_ids)

        return position_embeddings
