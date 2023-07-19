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

from modelzoo.common.pytorch.layers.RelativePositionEmbeddingLayer import (
    RelativePositionEmbeddingLayer,
)
from modelzoo.common.pytorch.model_utils.create_initializer import (
    create_initializer,
)


class AlibiPositionEmbeddingLayer(nn.Module):
    """Alibi Position Embedding Layer, Symmetric case with bidirectional supported

    alibi bias as in paper: https://arxiv.org/abs/2108.12409

    Args:
        num_heads (int): number of attention heads.
        slopes (Tensor): slope values to use for alibi heads. Shape: [num_heads, 1]. Default to `None`.
        alibi_trainable_slopes (bool): whether the alibi slopes are trainable parameters.
        slopes_initializer (str): initializer for alibi slopes if it's trainable. Defaults to ``xavier_uniform``.
        alibi_implementation (str): variant name for alibi implementation. Currently
            accepts ``embedding`` and ``expand``. Defaults to ``expand``.
    Returns:
        position_bias (Tensor): Relative position bias, to be used in attention masking
    """

    def __init__(
        self,
        num_heads,
        slopes=None,
        alibi_trainable_slopes=False,
        slopes_initializer="xavier_uniform",
        alibi_implementation="expand",
    ):
        super(AlibiPositionEmbeddingLayer, self).__init__()

        _SUPPORTED_ALIBI_IMPLEMENTATIONS = ["embedding", "expand"]
        assert (
            alibi_implementation in _SUPPORTED_ALIBI_IMPLEMENTATIONS
        ), f"Alibi implementation {alibi_implementation} is not supported."
        assert slopes is None, "Customized slope is not supported yet."

        self.num_heads = num_heads
        self.alibi_trainable_slopes = alibi_trainable_slopes
        self.use_embedding_implementation = alibi_implementation == "embedding"
        if not slopes:
            if self.alibi_trainable_slopes:
                slopes = torch.zeros([num_heads, 1])
                self.slopes_initializer = slopes_initializer
            else:
                slopes = torch.tensor(
                    AlibiPositionEmbeddingLayer._get_alibi_slopes(num_heads)
                ).unsqueeze(-1)
        else:
            if self.alibi_trainable_slopes:
                self.slopes_initializer = slopes_initializer

        self.slopes = nn.parameter.Parameter(
            slopes, requires_grad=self.alibi_trainable_slopes
        )

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        if self.alibi_trainable_slopes:
            create_initializer(self.slopes_initializer)(self.slopes.data)

    def forward(
        self, seq_length, key_length, past_kv=None,
    ):
        """Return the position bias based on the alibi slopes.

        Args:
            seq_length (int): the length of query tokens.
            key_length (int): the length of key tokens.

        Returns:
            Position bias tensor with shape [num_heads, query_length, key_length]
        """
        position_bias = self._compute_alibi_bias(seq_length, key_length)
        # if key and values are already calculated we want only
        # the last query position bias
        if past_kv is not None:
            position_bias = position_bias[:, :, -seq_length, :]

        return position_bias

    @staticmethod
    def _get_alibi_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + AlibiPositionEmbeddingLayer._get_alibi_slopes(
                    2 * closest_power_of_2
                )[0::2][: n - closest_power_of_2]
            )

    def _alibi_implementation_embedding(self, seq_length, key_length, slopes):
        # 1D tensor range(key_length): [0, 1, ... key_length - 1]
        range_k = torch.arange(
            key_length, dtype=torch.int32, device=slopes.device
        )

        # Compute bias for each head: slopes[head_index] * [0, 1, ... key_length - 1]
        # Shape: (key_length, num_heads)
        bias = slopes.permute([1, 0]) * range_k.unsqueeze(-1) * -1.0

        # Construct the broadcasting with compute_raw_relative_positions from RelativePositionEmbedding
        # Shape: (seq_length, key_length)
        relative_position = RelativePositionEmbeddingLayer.compute_raw_relative_positions(
            seq_length, key_length, device=slopes.device
        )
        # casting to int32 to bypass the wgt kernel gather limitation
        relative_position = torch.abs(relative_position).to(torch.int32)

        # Use embedding as a 2D to 3D broadcast.
        # Shape: (seq_length, key_length, num_heads)
        bias = nn.functional.embedding(relative_position, bias)

        # Transpose to the expected output order.
        # Shape: (num_heads, seq_length, key_length)
        bias = bias.permute([2, 0, 1])
        return bias

    def _alibi_implementation_expand(self, seq_length, key_length, slopes):
        relative_position = RelativePositionEmbeddingLayer.compute_raw_relative_positions(
            seq_length, key_length, device=slopes.device
        )
        relative_position = (
            torch.abs(relative_position)
            .unsqueeze(0)
            .expand(self.num_heads, -1, -1)
        )
        alibi = (slopes * -1.0).unsqueeze(1) * relative_position
        return alibi

    def _compute_alibi_bias(self, seq_length, key_length, slopes=None):
        if slopes is None:
            slopes = self.slopes

        if self.use_embedding_implementation:
            return self._alibi_implementation_embedding(
                seq_length, key_length, slopes
            )
        else:
            return self._alibi_implementation_expand(
                seq_length, key_length, slopes
            )
