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

from typing import Callable, Optional, Tuple, Type, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from modelzoo.common.pytorch.layers.AttentionHelper import get_attention_module
from modelzoo.common.pytorch.layers.FeedForwardNetwork import FeedForwardNetwork
from modelzoo.common.pytorch.model_utils.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)

SelfAttnKV = Tuple[Tensor, Tensor]
CrossAttnKV = Tuple[Tensor, Tensor]
SelfAndCrossAttnKV = Tuple[Tensor, Tensor, Tensor, Tensor]


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
        norm_layer: the normalization class that will be used before/after FF layers (default=nn.LayerNorm)
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        attention_dropout_rate: Attention dropout rate. If None, defaults to dropout.
        attention_softmax_fp32: Use FP32 softmax in attention block.
        use_projection_bias_in_attention: Add bias to Q,K,V projections
            in the Attention layer. Defaults to False.
        attention_type: Should be in ["scaled_dot_product", "dot_product"]
        attention_inner_dim (int):  Number of output units in attention query/key/value projection. Defaults to d_model
        add_cross_attention: If ``True``, adds cross-attention layer between encoder/decoder,
            otherwise, only self-attention is used in the decoder (GPT-style models should set to ``False``)
        use_ffn_bias_in_attention: Add bias in the concluding FFN
            in the Attention layer. Defaults to False.
        use_ffn_bias: Add bias in all dense layers of the decoder's ffn sublayer
        attention_initializer: Attention layer initializer. Defaults to "xavier_uniform".
        attention_q_initializer: Query projection kernel initializer. If not
            specified, the query will be initialized via ``attention_initializer``
        attention_output_layer_initializer: attention output layer projection initializer. If not
            specified, the output will be initialized via ``attention_initializer``
        ffn_initializer: FFN layer initializer. Defaults to "xavier_uniform".
        ffn_output_layer_initializer: If not None, initialize the last FFN layer
            with this initializer. Defaults to None.
        use_ff_layer1_dropout: If ``True``, dropout will be enabled after the first feed forward layer. Default: True
        use_ff_layer2_dropout = If ``True``, dropout will be enabled after the second feed forward layer. Default: True
        ffn_dropout_rate: Controls dropout rate of FF's first layer. If None, defaults to dropout.

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
        norm_layer: Type[nn.Module] = LayerNorm,
        norm_first: bool = False,
        attention_module_str="aiayn_attention",
        extra_attention_params={},
        device=None,
        add_cross_attention: bool = True,
        attention_dropout_rate: Optional[float] = None,
        attention_softmax_fp32: Optional[bool] = True,
        attention_type="scaled_dot_product",
        attention_inner_dim=None,
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        use_ffn_bias=False,
        attention_initializer="xavier_uniform",
        attention_q_initializer=None,
        attention_output_layer_initializer=None,
        ffn_initializer="xavier_uniform",
        ffn_output_layer_initializer=None,
        use_ff_layer1_dropout: bool = True,
        use_ff_layer2_dropout: bool = True,
        ffn_dropout_rate: Optional[float] = None,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()

        assert batch_first, "Currently, only batch_first=True is supported"
        self.add_cross_attention = add_cross_attention
        if attention_dropout_rate is None:
            attention_dropout_rate = dropout

        AttentionModule = get_attention_module(
            attention_module_str, extra_attention_params
        )
        self.self_attn = AttentionModule(
            d_model,
            nhead,
            inner_dim=attention_inner_dim,
            dropout=attention_dropout_rate,
            batch_first=batch_first,
            attention_type=attention_type,
            softmax_dtype_fp32=attention_softmax_fp32,
            use_projection_bias=use_projection_bias_in_attention,
            use_ffn_bias=use_ffn_bias_in_attention,
            attention_initializer=attention_initializer,
            attention_q_initializer=attention_q_initializer,
            output_layer_initializer=attention_output_layer_initializer,
            device=device,
            **extra_attention_params,
        )

        self.norm_first = norm_first
        self.norm1 = norm_layer(d_model, eps=layer_norm_eps, device=device,)
        self.dropout1 = Dropout(dropout)
        self.norm3 = norm_layer(d_model, eps=layer_norm_eps, device=device,)

        if self.add_cross_attention:
            self.multihead_attn = AttentionModule(
                d_model,
                nhead,
                inner_dim=attention_inner_dim,
                dropout=attention_dropout_rate,
                batch_first=batch_first,
                attention_type=attention_type,
                softmax_dtype_fp32=attention_softmax_fp32,
                use_projection_bias=use_projection_bias_in_attention,
                use_ffn_bias=use_ffn_bias_in_attention,
                attention_initializer=attention_initializer,
                attention_q_initializer=attention_q_initializer,
                output_layer_initializer=attention_output_layer_initializer,
                device=device,
                **extra_attention_params,
            )
            self.norm2 = norm_layer(d_model, eps=layer_norm_eps, device=device,)
            self.dropout2 = Dropout(dropout)

        if ffn_dropout_rate is None:
            ffn_dropout_rate = dropout

        self.ffn = FeedForwardNetwork(
            input_unit=d_model,
            layers_units=[dim_feedforward, d_model],
            layers_activation=[activation, None],
            layers_dropout_rates=[
                ffn_dropout_rate if use_ff_layer1_dropout else None,
                dropout if use_ff_layer2_dropout else None,
            ],
            use_bias=use_ffn_bias,
            kernel_initializer=ffn_initializer,
            output_layer_initializer=ffn_output_layer_initializer,
            bias_initializer="zeros",
            device=device,
        )
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        self.self_attn.reset_parameters()
        self.ffn.reset_parameters()
        if hasattr(self.norm1, 'bias'):
            self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        if hasattr(self.norm3, 'bias'):
            self.norm3.bias.data.zero_()
        self.norm3.weight.data.fill_(1.0)
        if self.add_cross_attention:
            self.multihead_attn.reset_parameters()
            if hasattr(self.norm2, 'bias'):
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
        past_kv: Optional[Union[SelfAttnKV, SelfAndCrossAttnKV]] = None,
        cache_present_kv: bool = False,
        self_attn_position_bias: Optional[Tensor] = None,
        cross_attn_position_bias: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Union[SelfAttnKV, SelfAndCrossAttnKV]]]:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            past_kv: Past keys and values for self attention and (if applicable) cross
                attention modules. Key/value tensors have shape
                ``[batch_size, num_heads, seq_length, embed_dim / num_heads]``. (optional).
            cache_present_kv: Specifies if the present keys and values
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        assert (
            past_kv is None and not cache_present_kv
        ), "Cannot provide past_kv because inference is not supported yet."

        x = tgt
        if self.norm_first:
            attn1_out = self._sa_block(
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                past_kv=past_kv[:2] if past_kv is not None else None,
                cache_present_kv=cache_present_kv,
                self_attn_position_bias=self_attn_position_bias,
            )

            x = x + attn1_out[0]

            if self.add_cross_attention:
                attn2_out = self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    past_kv=past_kv[2:] if past_kv is not None else None,
                    cache_present_kv=cache_present_kv,
                    cross_attn_position_bias=cross_attn_position_bias,
                )

                x = x + attn2_out[0]
            x = x + self.ffn(self.norm3(x))
        else:
            attn1_out = self._sa_block(
                x,
                tgt_mask,
                tgt_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                past_kv=past_kv[:2] if past_kv is not None else None,
                cache_present_kv=cache_present_kv,
                self_attn_position_bias=self_attn_position_bias,
            )

            x = self.norm1(x + attn1_out[0])
            if self.add_cross_attention:
                attn2_out = self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    past_kv=past_kv[2:] if past_kv is not None else None,
                    cache_present_kv=cache_present_kv,
                    cross_attn_position_bias=cross_attn_position_bias,
                )
                x = self.norm2(x + attn2_out[0])
            x = self.norm3(x + self.ffn(x))

        if not cache_present_kv:
            return x
        else:
            present_kv = (
                attn1_out[1]
                if not self.add_cross_attention
                else attn1_out[1] + attn2_out[1]
            )
            return x, present_kv

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
        past_kv: Optional[SelfAttnKV] = None,
        cache_present_kv: bool = False,
        self_attn_position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        attn_out = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            rotary_position_embedding_helper=rotary_position_embedding_helper,
            past_kv=past_kv,
            cache_present_kv=cache_present_kv,
            position_bias=self_attn_position_bias,
        )

        if cache_present_kv:
            out, present_kv = attn_out
        else:
            out = attn_out
        out = (self.dropout1(out),)
        if cache_present_kv:
            out += (present_kv,)
        return out

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        past_kv: Optional[CrossAttnKV] = None,
        cache_present_kv: bool = False,
        cross_attn_position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        attn_out = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            past_kv=past_kv,
            cache_present_kv=cache_present_kv,
            past_kv_self_attn=False,
            position_bias=cross_attn_position_bias,
        )

        if cache_present_kv:
            x, present_kv = attn_out
        else:
            x = attn_out

        out = (self.dropout2(x),)
        if cache_present_kv:
            out += (present_kv,)
        return out
