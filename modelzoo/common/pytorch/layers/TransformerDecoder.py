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

from typing import List, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from modelzoo.common.pytorch.layers.utils import _get_clones
from modelzoo.common.pytorch.model_utils.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)

SelfAttnKV = Tuple[Tensor, Tensor]
SelfAndCrossAttnKV = Tuple[Tensor, Tensor, Tensor, Tensor]


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # Re-initialize all layers to get new set of weights for each layer
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.norm:
            if hasattr(self.norm, 'bias'):
                self.norm.bias.data.zero_()
            self.norm.weight.data.fill_(1.0)

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        sparse_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        self_attn_position_bias: Optional[Tensor] = None,
        cross_attn_position_bias: Optional[Tensor] = None,
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
        past_kv: Optional[List[Union[SelfAttnKV, SelfAndCrossAttnKV]]] = None,
        cache_present_kv: bool = False,
    ) -> Union[
        Tensor, Tuple[Tensor, List[Union[SelfAttnKV, SelfAndCrossAttnKV]]]
    ]:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (optional).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            past_kv: Past keys and values for each of the decoder layers (optional).
            cache_present_kv: Specifies if the present keys and values
                must be cached and returned. (optional).

        Shape:
            see the docs in Transformer class.
        """
        assert (
            past_kv is None and not cache_present_kv
        ), "Cannot provide past_kv because inference is not supported yet."

        output = tgt
        present_kv = []

        for layer_idx, mod in enumerate(self.layers):
            output = mod(
                output,
                memory=memory,
                # Alternate between dense and fixed sparse attention,
                # This is used in GPT-3 model.
                tgt_mask=sparse_mask
                if layer_idx % 2 != 0 and sparse_mask is not None
                else tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                past_kv=None if past_kv is None else past_kv[layer_idx],
                cache_present_kv=cache_present_kv,
                self_attn_position_bias=self_attn_position_bias,
                cross_attn_position_bias=cross_attn_position_bias,
            )
            if cache_present_kv:
                present_kv.append(output[1])
                output = output[0]

        if self.norm is not None:
            output = self.norm(output)

        if cache_present_kv:
            return (output, present_kv)
        else:
            return output
