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

from modelzoo.common.pytorch.model_utils.create_initializer import (
    create_initializer,
)


class RelativePositionEmbeddingLayer(nn.Module):
    """Relative Position Embedding Layer
    Args:
        relative_attention_bias (Tensor): Tensor with relative attention weights.
            Shape: [`num_relative_attention_buckets`, `num_heads`]. Defaults set to None.
        num_relative_attention_buckets (int): Used to calculate relative position bias.
        bidirectional_relative_attention (bool): Whether attention is bidirectional.
        relative_attn_bias_initializer (bool): Relative Attention bias initializer.
            Defaults to ``xavier_uniform``.
    Returns:
        position_bias (Tensor): Relative position bias, to be used in attention masking
    """

    def __init__(
        self,
        num_heads,
        relative_attention_bias=None,
        num_relative_attention_buckets=32,
        bidirectional_relative_attention=False,
        relative_attn_bias_initializer="xavier_uniform",
    ):
        super(RelativePositionEmbeddingLayer, self).__init__()

        if relative_attention_bias:
            self.relative_attention_bias = relative_attention_bias
        else:
            # If no relative attention bias weights are provided when we create them.
            self.relative_attention_bias = nn.Embedding(
                num_relative_attention_buckets, num_heads
            )

        self.num_relative_attention_buckets = num_relative_attention_buckets
        self.bidirectional_relative_attention = bidirectional_relative_attention

        self.relative_attn_bias_initializer = relative_attn_bias_initializer

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        weight_initializer = create_initializer(
            self.relative_attn_bias_initializer
        )
        weight_initializer(self.relative_attention_bias.weight.data)

    def forward(
        self, seq_length, key_length, past_kv=None,
    ):
        position_bias = self._compute_bias(seq_length, key_length)

        # if key and values are already calculated we want only
        # the last query position bias
        if past_kv is not None:
            position_bias = position_bias[:, :, -seq_length, :]

        return position_bias

    def _compute_bias(self, query_length, key_length):
        """Compute binned relative position bias.
        Args:
            query_length (int): length of the query tensor.
            key_length (int): length of the key tensor.
        Returns:
            values (Tensor): computed values for position bias.
        """
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]

        # shape (query_length, key_length)
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional_relative_attention,
            num_buckets=self.num_relative_attention_buckets,
        )

        # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # shape (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1])
        return values

    def _relative_position_bucket(
        self,
        relative_position,
        bidirectional=True,
        num_buckets=32,
        max_distance=128,
    ):
        """
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the attended-to position.
        If ``self.bidirectional_relative_attention`` = False, then positive relative
        positions are invalid. We use smaller buckets for small absolute
        relative positions and larger buckets for larger absolute relative positions.
        All relative positions >= max_distance map to the same bucket.
        All relative positions <= -max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences
        than the model has been trained on.
        Args:
            relative_position (Tensor): Tensor with relative positions.
            bidirectional (bool): Whether attention is bidirectional
            num_buckets (int): number of buckets for relative positions
            max_distance (int): Used in order to calculate relative position buckets.
        Returns:
            a Tensor with the same shape as ``relative_position``,
            containing int32 values in the range [0, self.num_relative_attention_buckets).
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative position is in the range [0, inf)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # other half of the buckets are for logarithmically bigger bins
        # in position up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )

        # cast to int32_t because WS weight host gather can only handle int32
        return relative_buckets.to(torch.int32)
