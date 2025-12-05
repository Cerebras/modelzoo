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

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from cerebras.modelzoo.layers.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)
from cerebras.modelzoo.layers.utils import _get_clones
from cerebras.modelzoo.trainer import summarize_scalar

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

    def __init__(
        self,
        decoder_layer: Union[nn.Module, List[nn.Module]],
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        if isinstance(decoder_layer, list):
            if len(decoder_layer) != num_layers:
                raise ValueError(
                    f"The list of nn.Modules supplied to the TransformerDecoder"
                    f" block has length {len(decoder_layer)} while num_layers "
                    f"was set to {num_layers}. These two numbers are supposed "
                    f"to be the same. Check your config."
                )
            self.layers = nn.ModuleList(decoder_layer)
        else:
            self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.moe_enabled = self.layers[0].moe_enabled
        self.router_selection_nonlinearity = self.layers[
            0
        ].router_selection_nonlinearity
        # Re-initialize all layers to get new set of weights for each layer
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.norm:
            if hasattr(self.norm, 'bias') and hasattr(self.norm.bias, 'data'):
                self.norm.bias.data.zero_()
            if hasattr(self.norm, 'weight') and hasattr(
                self.norm.weight, 'data'
            ):
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
        extract_layer_idx: Optional[int] = None,
        expert_hash_idx: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        special_token_meta: Dict[str, Tensor] = None,
        **extra_args,
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
            self_attn_position_bias: the tensor containing position bias to apply in self-attention,
                can be obtained from relative or alibi position embeddings.
            cross_attn_position_bias: similar to self_attn_position_bias,
                this is the tensor containing position bias to apply in cross-attention.
            rotary_position_embedding_helper (Optional[RotaryPositionEmbeddingHelper]):
                A helper class to apply rotary embedding on the input tensor.
            past_kv: Past keys and values for each of the decoder layers (optional).
            cache_present_kv: Specifies if the present keys and values
                must be cached and returned. (optional).
            extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
                Applies decoder layers up to (and including) `extract_layer_idx`
                instead of all decoder layers.
                For ex: extract_layer_idx=3 would run fwd pass from decoder_block_0 to decoder_block_3
                and return outputs from decoder_block_3.
                If `extract_layer_idx` = None and `norm` != None, then
                the output returned would be decoder_block_{self.num_layers-1} -> norm -> output (return)
            expert_hash_idx: Optional tensor for mixture-of-experts models
                with hash-based routing. Tensor contains the expert ID for
                each token in the batch based on a hashing calculation.

        Shape:
            see the docs in Transformer class.
        """
        assert (
            past_kv is None and not cache_present_kv
        ), "Cannot provide past_kv because inference is not supported yet."

        output = tgt
        present_kv = []
        _is_extract_idx_was_none = extract_layer_idx is None
        if extract_layer_idx == None:
            extract_layer_idx = self.num_layers - 1

        routing_weights = []
        expert_masks = []
        for layer_idx in range(extract_layer_idx + 1):
            mod = self.layers[layer_idx]
            if isinstance(tgt_mask, list):
                # tgt_mask provided by the user is a list corresponding to the
                # mask for each layer.
                tgt_mask_i = tgt_mask[layer_idx]
            elif sparse_mask is not None and layer_idx % 2 != 0:
                # Alternate between dense and fixed sparse attention,
                # This is used in GPT-3 model.
                tgt_mask_i = sparse_mask
            else:
                tgt_mask_i = tgt_mask

            from cerebras.modelzoo.common.utils.model.transformer_utils import (
                SparseAttentionMask,
            )

            if isinstance(tgt_mask_i, SparseAttentionMask):
                mod.self_attn.sparse_attn_mask_ranges = (
                    tgt_mask_i.sparsity_annotation
                )
                tgt_mask_i = tgt_mask_i.mask_tensor

            output = mod(
                output,
                memory=memory,
                tgt_mask=tgt_mask_i,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                past_kv=None if past_kv is None else past_kv[layer_idx],
                cache_present_kv=cache_present_kv,
                self_attn_position_bias=self_attn_position_bias,
                cross_attn_position_bias=cross_attn_position_bias,
                layer_idx=layer_idx,
                expert_hash_idx=expert_hash_idx,
                position_ids=position_ids,
                special_token_meta=special_token_meta,
                **extra_args,
            )

            if self.moe_enabled:
                layer_routing_weights = output[1]
                num_experts = layer_routing_weights.shape[-1]
                routing_weights.append(layer_routing_weights)
                layer_expert_masks = output[2]
                expert_masks.append(layer_expert_masks)
                if cache_present_kv:
                    present_kv.append(output[3])
                output = output[0]

                # Log the entropy of the probabilities output from the routing.
                # We add an epsilon of 1e-5 for small probabilities, and we
                # normalize to maximum entropy for the given number of experts.
                entropy = (
                    layer_routing_weights
                    * -torch.log(layer_routing_weights + 1e-5)
                ).sum() / (layer_routing_weights.numel() / num_experts)
                max_entropy = torch.log(
                    torch.tensor(num_experts, dtype=layer_routing_weights.dtype)
                )
                entropy /= max_entropy
                summarize_scalar(
                    f"expert_stats/entropy_l{layer_idx}",
                    entropy,
                )

            else:
                if cache_present_kv:
                    present_kv.append(output[1])
                    output = output[0]

        if self.norm is not None and _is_extract_idx_was_none:
            output = self.norm(output)

        if self.moe_enabled:
            if cache_present_kv:
                return output, routing_weights, expert_masks, present_kv
            else:
                return output, routing_weights, expert_masks
        else:
            if cache_present_kv:
                return (output, present_kv)
            else:
                return output
