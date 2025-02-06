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

"""
Adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
"""

from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from cerebras.modelzoo.layers.AdaLayerNorm import AdaLayerNorm
from cerebras.modelzoo.layers.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)
from cerebras.modelzoo.layers.TransformerDecoderLayer import (
    TransformerDecoderLayer,
)

SelfAttnKV = Tuple[Tensor, Tensor]
CrossAttnKV = Tuple[Tensor, Tensor]
SelfAndCrossAttnKV = Tuple[Tensor, Tensor, Tensor, Tensor]


class DiTDecoderLayer(TransformerDecoderLayer):
    def __init__(self, gate_res=True, **kwargs):
        super(DiTDecoderLayer, self).__init__(**kwargs)
        d_model = kwargs["d_model"]

        self.gate_res = gate_res
        if gate_res:
            self.gate_msa = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=True),
                nn.Unflatten(dim=1, unflattened_size=(1, -1)),
            )
            self.gate_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=True),
                nn.Unflatten(dim=1, unflattened_size=(1, -1)),
            )

        self.__reset_gate_parameters()

    def __reset_gate_parameters(self):
        # zero initialize
        if self.gate_res:
            self.gate_msa[1].bias.data.zero_()
            self.gate_msa[1].weight.data.zero_()
            self.gate_mlp[1].bias.data.zero_()
            self.gate_mlp[1].weight.data.zero_()

    def reset_parameters(self):
        super().reset_parameters()
        self.__reset_gate_parameters()

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
        past_kv: Optional[Union[SelfAttnKV, SelfAndCrossAttnKV]] = None,
        cache_present_kv: bool = False,
        self_attn_position_bias: Optional[Tensor] = None,
        cross_attn_position_bias: Optional[Tensor] = None,
        **extra_args,
    ) -> Union[Tensor, Tuple[Tensor, Union[SelfAttnKV, SelfAndCrossAttnKV]]]:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        assert (
            past_kv is None and not cache_present_kv
        ), "Cannot provide past_kv because inference is not supported yet."
        res = x = tgt
        if self.norm_first:
            if isinstance(self.norm1, nn.LayerNorm):
                x = self.norm1(x)
            elif isinstance(self.norm1, AdaLayerNorm):
                x = self.norm1(x, memory)

            attn1_out = self._sa_block(
                x,
                tgt_mask,
                tgt_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                past_kv=past_kv[:2] if past_kv is not None else None,
                cache_present_kv=cache_present_kv,
                self_attn_position_bias=self_attn_position_bias,
                **extra_args,
            )
            res = x = (
                res
                + (self.gate_msa(memory) if self.gate_res else 1.0)
                * attn1_out[0]
            )

            if self.add_cross_attention:
                attn2_out = self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    past_kv=past_kv[2:] if past_kv is not None else None,
                    cache_present_kv=cache_present_kv,
                    cross_attn_position_bias=cross_attn_position_bias,
                    **extra_args,
                )

                res = x = x + attn2_out[0]

            if isinstance(self.norm3, nn.LayerNorm):
                x = self.norm3(x)
            elif isinstance(self.norm3, AdaLayerNorm):
                x = self.norm3(x, memory)
            x = self.ffn(x)
            x = res + (self.gate_mlp(memory) if self.gate_res else 1.0) * x
        else:
            attn1_out = self._sa_block(
                x,
                tgt_mask,
                tgt_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                past_kv=past_kv[:2] if past_kv is not None else None,
                cache_present_kv=cache_present_kv,
                self_attn_position_bias=self_attn_position_bias,
                **extra_args,
            )

            x = (
                res
                + (self.gate_msa(memory) if self.gate_res else 1.0)
                * attn1_out[0]
            )

            if isinstance(self.norm1, nn.LayerNorm):
                x = self.norm1(x)
            elif isinstance(self.norm1, AdaLayerNorm):
                x = self.norm1(x, memory)

            if self.add_cross_attention:
                attn2_out = self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    past_kv=past_kv[2:] if past_kv is not None else None,
                    cache_present_kv=cache_present_kv,
                    cross_attn_position_bias=cross_attn_position_bias,
                    **extra_args,
                )
                x = self.norm2(x + attn2_out[0])

            res = x
            x = self.ffn(x)
            x = res + (self.gate_mlp(memory) if self.gate_res else 1.0) * x

            if isinstance(self.norm3, nn.LayerNorm):
                x = self.norm3(x)
            elif isinstance(self.norm3, AdaLayerNorm):
                x = self.norm3(x, memory)

        if not cache_present_kv:
            return x
        else:
            present_kv = (
                attn1_out[1]
                if not self.add_cross_attention
                else attn1_out[1] + attn2_out[1]
            )
            return x, present_kv
