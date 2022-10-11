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

from typing import Callable, Optional, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from modelzoo.common.pytorch.layers.AttentionLayer import MultiheadAttention
from modelzoo.common.pytorch.layers.FeedForwardNetwork import FeedForwardNetwork
from modelzoo.common.pytorch.model_utils.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)


class TransformerDecoderLayer(nn.Module):
    r"""
    TransformerDecoderLayer is made up of self-attn, multihead-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multihead-attention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: gelu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        attention_dropout_rate: Attention dropout rate. If None, defaults to dropout.
        use_projection_bias_in_attention: Add bias to Q,K,V projections
            in the Attention layer. Defaults to False.
        attention_type: Should be in ["scaled_dot_product", "dot_product"]
        add_cross_attention: If ``True``, adds cross-attention layer between encoder/decoder,
            otherwise, only self-attention is used in the decoder (GPT-style models should set to ``False``)
        use_ffn_bias_in_attention: Add bias in the concluding FFN
            in the Attention layer. Defaults to False.
        use_ffn_bias: Add bias in all dense layers of the decoder's ffn sublayer
        attention_initializer: Attention layer initializer. Defaults to "xavier_uniform".
        ffn_initializer: FFN layer initializer. Defaults to "xavier_uniform".

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        device=None,
        add_cross_attention: bool = True,
        attention_dropout_rate=None,
        attention_type="scaled_dot_product",
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        use_ffn_bias=False,
        attention_initializer="xavier_uniform",
        ffn_initializer="xavier_uniform",
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()

        assert batch_first, "Currently, only batch_first=True is supported"
        self.add_cross_attention = add_cross_attention
        if attention_dropout_rate is None:
            attention_dropout_rate = dropout

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout_rate,
            batch_first=batch_first,
            attention_type=attention_type,
            use_projection_bias=use_projection_bias_in_attention,
            use_ffn_bias=use_ffn_bias_in_attention,
            attention_initializer=attention_initializer,
            device=device,
        )

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, device=device,)
        self.dropout1 = Dropout(dropout)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, device=device,)

        if self.add_cross_attention:
            self.multihead_attn = MultiheadAttention(
                d_model,
                nhead,
                dropout=attention_dropout_rate,
                batch_first=batch_first,
                attention_type=attention_type,
                use_projection_bias=use_projection_bias_in_attention,
                use_ffn_bias=use_ffn_bias_in_attention,
                attention_initializer=attention_initializer,
                device=device,
            )
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, device=device,)
            self.dropout2 = Dropout(dropout)

        self.ffn = FeedForwardNetwork(
            input_unit=d_model,
            layers_units=[dim_feedforward, d_model],
            layers_activation=[activation, None],
            layers_dropout_rates=[None, dropout],
            use_bias=use_ffn_bias,
            kernel_initializer=ffn_initializer,
            bias_initializer="zeros",
            device=device,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        self.self_attn._reset_parameters()
        self.ffn._reset_parameters()
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm3.bias.data.zero_()
        self.norm3.weight.data.fill_(1.0)
        if self.add_cross_attention:
            self.multihead_attn._reset_parameters()
            self.norm2.bias.data.zero_()
            self.norm2.weight.data.fill_(1.0)

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
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
                rotary_position_embedding_helper,
            )
            if self.add_cross_attention:
                x = x + self._mha_block(
                    self.norm2(x), memory, memory_mask, memory_key_padding_mask
                )
            x = x + self.ffn(self.norm3(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    tgt_mask,
                    tgt_key_padding_mask,
                    rotary_position_embedding_helper,
                )
            )
            if self.add_cross_attention:
                x = self.norm2(
                    x
                    + self._mha_block(
                        x, memory, memory_mask, memory_key_padding_mask
                    )
                )
            x = self.norm3(x + self.ffn(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            rotary_position_embedding_helper=rotary_position_embedding_helper,
        )
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout2(x)
