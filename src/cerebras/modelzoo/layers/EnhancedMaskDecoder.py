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

from torch import Tensor

from .TransformerEncoderLayer import TransformerEncoderLayer


class EnhancedMaskDecoder(TransformerEncoderLayer):
    r"""Enhanced Mask Decoder (EMD) layer as described in the paper "DEBERTA:
    Decoding-Enhanced BERT with disentangled attention" (https://arxiv.org/pdf/2006.03654.pdf).
    Unlike the standard TransformerEncoderLayer which restricts the queries,
    keys, and values to be the same vector, EMD allows the query vector
    to be different from the key & value vector.
    """

    def __init__(self, **params):
        super().__init__(**params)
        assert (
            self.norm_first is False
        ), "EnhancedMaskDecoder doesn't support norm_first=True"

    def forward(
        self,
        src: Tensor,
        query: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        self_attn_position_bias: Optional[Tensor] = None,
        **extra_args,
    ) -> Tensor:
        r"""
        The layer introduces a new optional argument (`query`) to the forward function.
        When this argument is provided, the specified query is used in the attention
        computation (instead of src) and in the attention block's residual connection.
        """
        x = src
        if query is None:
            query = x

        x = self.norm1(
            query
            + self._sa_block(
                x,
                query=query,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                self_attn_position_bias=self_attn_position_bias,
                **extra_args,
            )
        )
        x = self.norm2(x + self.ffn(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        query: Optional[Tensor],
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        self_attn_position_bias: Optional[Tensor],
        **extra_args,
    ) -> Tensor:
        if query is None:
            query = x
        x = self.self_attn(
            query,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            position_bias=self_attn_position_bias,
            need_weights=False,
            **extra_args,
        )
        return self.dropout1(x)
