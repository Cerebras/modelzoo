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

import math
from typing import Callable, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, LayerNorm

import cerebras.pytorch as cstorch
from cerebras.modelzoo.layers.AttentionHelper import get_attention_module
from cerebras.modelzoo.layers.FeedForwardNetwork import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
)
from cerebras.modelzoo.layers.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)
from cerebras.modelzoo.layers.StochasticDepth import StochasticDepth


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multihead attention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: gelu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_layer: the normalization class that will be used before/after FF layers (default=nn.LayerNorm)
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        attention_dropout_rate: Attention dropout rate. If None, defaults to dropout.
        use_projection_bias_in_attention: Add bias to Q,K,V projections
            in the Attention layer. Defaults to False.
        attention_type: Should be in ["scaled_dot_product", "dot_product"]
        scale_qk_dot_by_d (bool): If ``True`` scales QK^T dot product by d(=hidden/d_head) instead of sqrt(d).
        attention_softmax_fp32: Use FP32 softmax in attention block.
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
        layerscale_value: initial value to use for LayerScale in vision transformers. Defaults to None.
        gate_attention: If ``True``, the self attention output gated via the tanh of a learned parameter. Default: ``False``.
        gate_ffn: If ``True``, the feedforward output is gated via the tanh of a learned parameter. Default: ``False``.
        stochastic_depth_drop_prob: drop probability for stochastic depth per sample (when applied in main path of residual blocks.
        stochastic_depth_mode: should be in ["batch", "row"].

    Example:
        When ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
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
        attention_module: Union[str, nn.Module] = "aiayn_attention",
        extra_attention_params={},
        device=None,
        attention_dropout_rate: Optional[float] = None,
        attention_type="scaled_dot_product",
        scale_qk_dot_by_d=False,
        attention_logits_alpha=1.0,
        q_projection_scale=1.0,
        k_projection_scale=1.0,
        v_projection_scale=1.0,
        output_projection_scale=1.0,
        attention_softmax_fp32: Optional[bool] = True,
        attention_inner_dim=None,
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        use_ffn_bias=False,
        attention_initializer="xavier_uniform",
        attention_q_initializer=None,
        attention_output_layer_initializer=None,
        attention_logit_softcapping=None,
        ffn_initializer="xavier_uniform",
        ffn_output_layer_initializer=None,
        use_ff_layer1_dropout: bool = True,
        use_ff_layer2_dropout: bool = True,
        ffn_dropout_rate: Optional[float] = None,
        layerscale_value: Optional[float] = None,
        gate_attention: bool = False,
        gate_ffn: bool = False,
        stochastic_depth_drop_prob: Optional[float] = 0.0,
        stochastic_depth_mode: Optional[str] = "batch",
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()

        assert batch_first, "Currently, only batch_first=True is supported"
        if attention_dropout_rate is None:
            attention_dropout_rate = dropout

        AttentionModule = get_attention_module(
            attention_module, extra_attention_params
        )

        self.self_attn = AttentionModule(
            d_model,
            nhead,
            inner_dim=attention_inner_dim,
            dropout=attention_dropout_rate,
            batch_first=batch_first,
            attention_type=attention_type,
            scale_qk_dot_by_d=scale_qk_dot_by_d,
            attention_logits_alpha=attention_logits_alpha,
            q_projection_scale=q_projection_scale,
            k_projection_scale=k_projection_scale,
            v_projection_scale=v_projection_scale,
            output_projection_scale=output_projection_scale,
            softmax_dtype_fp32=attention_softmax_fp32,
            use_projection_bias=use_projection_bias_in_attention,
            use_ffn_bias=use_ffn_bias_in_attention,
            attention_initializer=attention_initializer,
            attention_q_initializer=attention_q_initializer,
            output_layer_initializer=attention_output_layer_initializer,
            logit_softcapping=attention_logit_softcapping,
            device=device,
            **extra_attention_params,
        )

        self.layerscale_value = layerscale_value
        if self.layerscale_value is not None:
            self.layer_scale1 = nn.Parameter(
                self.layerscale_value * torch.ones(d_model)
            )
            self.layer_scale2 = nn.Parameter(
                self.layerscale_value * torch.ones(d_model)
            )

        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob
        self.stochastic_depth_mode = stochastic_depth_mode

        if self.stochastic_depth_drop_prob > 0.0:
            self.drop_path = StochasticDepth(
                self.stochastic_depth_drop_prob, self.stochastic_depth_mode
            )
        else:
            self.drop_path = nn.Identity()

        if ffn_dropout_rate is None:
            ffn_dropout_rate = dropout

        self.ffn = FeedForwardNetwork(
            FeedForwardNetworkConfig(
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
        )

        self.norm_first = norm_first
        self.norm1 = norm_layer(
            d_model,
            eps=layer_norm_eps,
            device=device,
        )
        self.norm2 = norm_layer(
            d_model,
            eps=layer_norm_eps,
            device=device,
        )
        self.dropout1 = Dropout(dropout)

        self.gate_attention = gate_attention
        self.gate_ffn = gate_ffn
        if self.gate_attention:
            self.gate_attn_weight = nn.Parameter(torch.ones(1) * math.pi / 4)
        if self.gate_ffn:
            self.gate_ffn_weight = nn.Parameter(torch.ones(1) * math.pi / 4)

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        self.self_attn.reset_parameters()
        if self.layerscale_value is not None:
            self.layer_scale1.data.fill_(self.layerscale_value)
            self.layer_scale2.data.fill_(self.layerscale_value)

        # initialized following the llama3.2 implementation: https://github.com/huggingface/transformers/blob/e1b150862e66e16acf951edfa13206ffcd1032be/src/transformers/models/mllama/modeling_mllama.py#L345
        if self.gate_attention:
            self.gate_attn_weight.data.fill_(math.pi / 4)
        if self.gate_ffn:
            self.gate_ffn_weight.data.fill_(math.pi / 4)

        self.ffn.reset_parameters()
        if hasattr(self.norm1, 'bias') and hasattr(self.norm1.bias, "data"):
            self.norm1.bias.data.zero_()
        if hasattr(self.norm1, 'weight') and hasattr(self.norm1.weight, "data"):
            self.norm1.weight.data.fill_(1.0)
        if hasattr(self.norm2, 'bias') and hasattr(self.norm2.bias, "data"):
            self.norm2.bias.data.zero_()
        if hasattr(self.norm2, 'weight') and hasattr(self.norm1.weight, "data"):
            self.norm2.weight.data.fill_(1.0)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
        self_attn_position_bias: Optional[Tensor] = None,
        **extra_args,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            rotary_position_embedding_helper (Optional[RotaryPositionEmbeddingHelper]):
                A helper class to apply rotary embedding on the input tensor.
            self_attn_position_bias: the tensor containing position bias to apply in self-attention,
                can be obtained from relative or alibi position embeddings.

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                self_attn_position_bias=self_attn_position_bias,
                **extra_args,
            )
            x = x + self._ffn_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    rotary_position_embedding_helper=rotary_position_embedding_helper,
                    self_attn_position_bias=self_attn_position_bias,
                    **extra_args,
                )
            )
            x = self.norm2(x + self._ffn_block(x))

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
        self_attn_position_bias: Optional[Tensor] = None,
        **extra_args,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            rotary_position_embedding_helper=rotary_position_embedding_helper,
            position_bias=self_attn_position_bias,
            need_weights=False,
            **extra_args,
        )
        x = self.dropout1(x)
        if self.gate_attention:
            x = (
                self.gate_attn_weight.to(cstorch.amp.get_half_dtype())
                .broadcast_to(x.shape)
                .tanh()
                * x
            )
        if self.layerscale_value is not None:
            x = self.layer_scale1.to(cstorch.amp.get_half_dtype()) * x
        x = self.drop_path(x)
        return x

    # ffn block
    def _ffn_block(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.ffn(x)
        if self.gate_ffn:
            x = (
                self.gate_ffn_weight.to(cstorch.amp.get_half_dtype())
                .broadcast_to(x.shape)
                .tanh()
                * x
            )
        if self.layerscale_value is not None:
            x = self.layer_scale2.to(cstorch.amp.get_half_dtype()) * x
        x = self.drop_path(x)
        return x
