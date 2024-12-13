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

import torch
import torch.nn as nn


class FixedPositionEmbeddingLayer(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        positional_embedding_size,
        min_timescale,
        max_timescale,
    ) -> None:
        super(FixedPositionEmbeddingLayer, self).__init__()

        self.fpe = self.create_fix_pos_embedding(
            max_position_embeddings,
            positional_embedding_size,
            min_timescale,
            max_timescale,
        )
        self.max_position_embeddings = max_position_embeddings

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        pass

    def forward(self, length, position_ids=None, dtype=None):
        """Compute positional encoding using fixed embeddings.

        Args:
            length (int): Length of sequence to generate positional encoding for.
                Cannot be larger than max_position_embeddings.
            position_ids (Tensor): position ids with shape ``[batch_size, seq_length]``.
            dtype: Type of output positional encoding.

        Returns:
            Position embedding output with shape ``[batch_size, seq_length, positional_embedding_size]``.
        """
        positional_embed = self.fpe.to(dtype=dtype)
        if position_ids is None:
            assert (
                length <= self.max_position_embeddings
            ), "FixedPositionEmbeddingLayer: length cannot be larger than max_position_embeddings"
            if length != self.max_position_embeddings:
                positional_embed = positional_embed[:length]
        else:
            assert (
                len(positional_embed.shape) == 2
            ), "positional_embed must be a 2D tensor."
            batch_size = position_ids.shape[0]
            seq_length = position_ids.shape[1]
            max_position_embeddings = positional_embed.shape[0]
            embedding_size = positional_embed.shape[1]

            bc_positional_embed = positional_embed[None, :, :].broadcast_to(
                batch_size, max_position_embeddings, embedding_size
            )
            bc_position_ids = (
                position_ids[:, :, None]
                .broadcast_to(batch_size, seq_length, embedding_size)
                .to(torch.long)
            )
            positional_embed = torch.gather(
                bc_positional_embed, 1, bc_position_ids
            )
        return positional_embed

    @staticmethod
    def create_fix_pos_embedding(
        seq_len, embed_len, min_timescale, max_timescale
    ):
        """
        adapted from: https://github.com/tensorflow/tensor2tensor/blob\
            /1843c72d1d5faf4c085bb198b5dde0908f4081d0/tensor2tensor/layers\
            /common_attention.py#L407
        """
        position = torch.arange(seq_len, dtype=torch.float32)
        num_timescales = embed_len // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (float(num_timescales) - 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32)
            * -log_timescale_increment
        )
        scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
            inv_timescales, 0
        )
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=1
        )
        signal = torch.reshape(signal, (seq_len, 2, num_timescales))
        signal = torch.transpose(signal, 1, 2)
        signal = torch.reshape(signal, (seq_len, 2 * num_timescales))
        signal = torch.nn.functional.pad(
            signal,
            (0, embed_len % 2, 0, 0),
        )

        return torch.nn.Parameter(signal, requires_grad=False)
