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

from typing import List, Optional

import torch.nn as nn
from torch import Tensor

from cerebras.modelzoo.layers.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)
from cerebras.modelzoo.layers.utils import _get_clones


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``False`` (disabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(
        self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False
    ):
        super(TransformerEncoder, self).__init__()

        assert not enable_nested_tensor, "Nested tensors are not supported."
        self.layers = _get_clones(encoder_layer, num_layers)
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
            if hasattr(self.norm, "bias") and hasattr(self.norm.bias, "data"):
                self.norm.bias.data.zero_()
            if hasattr(self.norm, "weight") and hasattr(
                self.norm.weight, "data"
            ):
                self.norm.weight.data.fill_(1.0)

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
        self_attn_position_bias: Optional[Tensor] = None,
        extract_layer_idx: Optional[int] = None,
        intermediate_layers_indices: Optional[List[int]] = None,
        **extra_args,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            rotary_position_embedding_helper (Optional[RotaryPositionEmbeddingHelper]):
                A helper class to apply rotary embedding on the input tensor.
            self_attn_position_bias: the tensor containing position bias to apply in self-attention,
                can be obtained from relative or alibi position embeddings.
            extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
                Applies encoder layers up to (and including) `extract_layer_idx`
                instead of all encoder layers.
                For ex: extract_layer_idx=3 would run fwd pass from encoder_block_0 to encoder_block_3
                and return outputs from encoder_block_3.
                If `extract_layer_idx` = None and `norm` != None, then
                the output returned would be encoder_block_{self.num_layers-1} -> norm -> output (return)
            intermediate_layers_indices: list of layer indices in range [0, self.num_layers) (zero-indexed) to return

        Shape:
            see the docs in Transformer class.
        """

        _is_extract_idx_was_none = extract_layer_idx is None
        if extract_layer_idx is None:
            extract_layer_idx = self.num_layers - 1

        assert (
            extract_layer_idx < self.num_layers
        ), f"extract_layer_idx should be < {self.num_layers}, got {extract_layer_idx}"

        output = src

        intermediate_layers = []
        for i in range(extract_layer_idx + 1):
            if (
                intermediate_layers_indices is not None
                and i in intermediate_layers_indices
            ):
                intermediate_layers.append(output)

            mod = self.layers[i]
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                self_attn_position_bias=self_attn_position_bias,
                **extra_args,
            )

        if self.norm is not None and _is_extract_idx_was_none:
            output = self.norm(output)

        if intermediate_layers_indices is not None:
            return output, intermediate_layers
        else:
            return output
