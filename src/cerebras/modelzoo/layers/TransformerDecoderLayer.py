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

from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Identity, LayerNorm

from cerebras.modelzoo.layers.AttentionHelper import get_attention_module
from cerebras.modelzoo.layers.FeedForwardNetwork import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
    MoEConfig,
)
from cerebras.modelzoo.layers.MemoryTokenHelpers import (
    create_mem_token_attn_module,
)
from cerebras.modelzoo.layers.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)
from cerebras.modelzoo.layers.SparseMoEBlock import SparseMoEBlock
from cerebras.modelzoo.layers.utils import reset_norm

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
        scale_qk_dot_by_d (bool): If ``True`` scales QK^T dot product by d(=hidden/d_head) instead of sqrt(d).
        attention_logit_alpha (float): Scales the QK^T dot product. Used to stabilize logits in muP training.
        attention_inner_dim (int):  Number of output units in attention query/key/value projection. Defaults to d_model
        attention_qk_norm_layer: the normalization string to create the class for QK normalization,
            as proposed in https://arxiv.org/pdf/2309.14322.pdf (default=nn.Identity)
        attention_qk_norm_eps: the eps value in qk layer normalization components (default=1e-5).
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
        moe_params: A dict of MoE params including num_experts, top_k and load_balancing_loss_coef
        disable_self_attention (bool):  If ``True``, Disables the self-attention and only uses cross-attention
            in the decoder layer. Defaults to ``False``,
        cross_attention_gate_attention (bool):  If ``True``, the output of the cross-attention attention is multiplied by a learnable
            gate parameter with tanh() activation. Defaults to ``False``.
        cross_attention_gate_mlp (bool):  If ``True``, the output of the cross-attention mlp is multiplied by a learnable
            gate parameter with tanh() activation. Defaults to ``False``.

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
        norm_first_sandwich: bool = False,
        attention_module: Union[str, nn.Module] = "aiayn_attention",
        extra_attention_params={},
        extra_ffn_params={},
        device=None,
        add_cross_attention: bool = True,
        attention_dropout_rate: Optional[float] = None,
        attention_softmax_fp32: Optional[bool] = True,
        attention_type="scaled_dot_product",
        scale_qk_dot_by_d=False,
        attention_logits_alpha: Optional[float] = 1.0,
        q_projection_scale=1.0,
        k_projection_scale=1.0,
        v_projection_scale=1.0,
        output_projection_scale=1.0,
        scale_qk_dot_by_layer_idx=False,
        attention_inner_dim=None,
        attention_qk_norm_layer: Type[nn.Module] = Identity,
        attention_qk_norm_eps: float = 1e-5,
        cross_attention_kv_dim=None,
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
        moe_params=MoEConfig(),
        memory_tokens_config=None,
        disable_self_attention=False,
        cross_attention_gate_attention=False,
        cross_attention_gate_mlp=False,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()

        assert batch_first, "Currently, only batch_first=True is supported"
        self.add_cross_attention = add_cross_attention
        self.disable_self_attention = disable_self_attention
        self.cross_attention_gate_attention = cross_attention_gate_attention
        self.cross_attention_gate_mlp = cross_attention_gate_mlp

        if self.cross_attention_gate_attention:
            self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))

        if self.cross_attention_gate_mlp:
            self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

        if attention_dropout_rate is None:
            attention_dropout_rate = dropout
        AttentionModule = get_attention_module(
            attention_module, extra_attention_params
        )
        if (
            memory_tokens_config is not None
            and memory_tokens_config.add_qkv_memory_weights
        ):
            # Wrap the attention module so that a separate set of
            # weights is added for memory tokens
            AttentionModule = create_mem_token_attn_module(AttentionModule)

        if not disable_self_attention:
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
                attention_qk_norm_layer=attention_qk_norm_layer,
                attention_qk_norm_eps=attention_qk_norm_eps,
                attention_initializer=attention_initializer,
                attention_q_initializer=attention_q_initializer,
                output_layer_initializer=attention_output_layer_initializer,
                scale_qk_dot_by_layer_idx=scale_qk_dot_by_layer_idx,
                logit_softcapping=attention_logit_softcapping,
                device=device,
                **extra_attention_params,
            )

        self.norm_first = norm_first
        if norm_first_sandwich:
            assert self.norm_first, (
                "When norm_first_sandwich is enabled, norm_first must be "
                "enabled too"
            )
        self.norm_first_sandwich = norm_first_sandwich
        self.norm1 = norm_layer(
            d_model,
            eps=layer_norm_eps,
            device=device,
        )
        self.dropout1 = Dropout(dropout)
        self.norm3 = norm_layer(
            d_model,
            eps=layer_norm_eps,
            device=device,
        )

        if norm_first_sandwich:
            self.norm1_post = norm_layer(
                d_model,
                eps=layer_norm_eps,
                device=device,
            )
            self.norm3_post = norm_layer(
                d_model,
                eps=layer_norm_eps,
                device=device,
            )

        if self.add_cross_attention:
            if cross_attention_kv_dim is None:
                cross_attention_kv_dim = d_model
            self.multihead_attn = AttentionModule(
                d_model,
                nhead,
                kdim=cross_attention_kv_dim,
                vdim=cross_attention_kv_dim,
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
                attention_qk_norm_layer=attention_qk_norm_layer,
                attention_qk_norm_eps=attention_qk_norm_eps,
                attention_initializer=attention_initializer,
                attention_q_initializer=attention_q_initializer,
                output_layer_initializer=attention_output_layer_initializer,
                scale_qk_dot_by_layer_idx=scale_qk_dot_by_layer_idx,
                logit_softcapping=attention_logit_softcapping,
                device=device,
                **extra_attention_params,
            )
            if not self.disable_self_attention and self.add_cross_attention:
                self.norm2 = norm_layer(
                    d_model,
                    eps=layer_norm_eps,
                    device=device,
                )
            self.dropout2 = Dropout(dropout)

        if ffn_dropout_rate is None:
            ffn_dropout_rate = dropout

        ffn_config = FeedForwardNetworkConfig(
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
            moe_params=moe_params,
            **extra_ffn_params,
        )

        total_experts = moe_params.num_experts
        if moe_params.num_shared_experts:
            total_experts += moe_params.num_shared_experts
        self.moe_enabled = total_experts > 1
        self.router_selection_nonlinearity = (
            moe_params.router_selection_nonlinearity
        )
        if self.moe_enabled:
            self.ffn = SparseMoEBlock(ffn_config)
        else:
            self.ffn = FeedForwardNetwork(ffn_config)

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        if not self.disable_self_attention:
            self.self_attn.reset_parameters()
        self.ffn.reset_parameters()

        reset_norm(self.norm1)

        if self.cross_attention_gate_attention:
            self.cross_attn_attn_gate.data.fill_(0.0)

        if self.cross_attention_gate_mlp:
            self.cross_attn_mlp_gate.data.fill_(0.0)

        if self.norm3 is not None:
            reset_norm(self.norm3)

        if self.norm_first_sandwich:
            reset_norm(self.norm1_post)

            if self.norm3_post is not None:
                reset_norm(self.norm3_post)

        if self.add_cross_attention:
            self.multihead_attn.reset_parameters()
            if not self.disable_self_attention:
                reset_norm(self.norm2)

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
        layer_idx: Optional[int] = None,
        expert_hash_idx: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        special_token_meta: Dict[str, Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tensor] = None,
        **extra_args,
    ) -> Union[Tensor, Tuple[Tensor, Union[SelfAttnKV, SelfAndCrossAttnKV]]]:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            rotary_position_embedding_helper (Optional[RotaryPositionEmbeddingHelper]):
                A helper class to apply rotary embedding on the input tensor.
            past_kv: Past keys and values for self attention and (if applicable) cross
                attention modules. Key/value tensors have shape
                ``[batch_size, num_heads, seq_length, embed_dim / num_heads]``. (optional).
            cache_present_kv: Specifies if the present keys and values
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. (optional).
            self_attn_position_bias: the tensor containing position bias to apply in self-attention,
                can be obtained from relative or alibi position embeddings.
            expert_hash_idx: tensor containing mixture-of-experts expert
                selection indices for each token in the batch. Only used with
                MoE with hash-based routing enabled (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        assert (
            past_kv is None and not cache_present_kv
        ), "Cannot provide past_kv because inference is not supported yet."

        ffn_extra_args = {}
        ffn_extra_args["layer_idx"] = layer_idx
        if (
            extra_args
            and ("token_modality_idx" in extra_args)
            and (extra_args["token_modality_idx"] is not None)
        ):
            ffn_extra_args["token_modality_idx"] = extra_args[
                "token_modality_idx"
            ]
            extra_args.pop('token_modality_idx')
        if self.moe_enabled and expert_hash_idx is not None:
            ffn_extra_args["expert_hash_idx"] = expert_hash_idx

        # Skip this layer if no self-attention and no cross-attention
        if self.disable_self_attention and memory is None:
            return tgt

        x = tgt
        if self.norm_first:
            if not self.disable_self_attention:
                attn1_out = self._sa_block(
                    self.norm1(x),
                    tgt_mask,
                    tgt_key_padding_mask,
                    rotary_position_embedding_helper=rotary_position_embedding_helper,
                    past_kv=past_kv[:2] if past_kv is not None else None,
                    cache_present_kv=cache_present_kv,
                    self_attn_position_bias=self_attn_position_bias,
                    layer_idx=layer_idx,
                    position_ids=position_ids,
                    special_token_meta=special_token_meta,
                    **extra_args,
                )
                post_attn1 = attn1_out[0]

                if self.norm_first_sandwich:
                    post_attn1 = self.norm1_post(post_attn1)

                x = x + post_attn1

            if self.add_cross_attention:
                attn2_out = self._mha_block(
                    (
                        self.norm2(x)
                        if not self.disable_self_attention
                        else self.norm1(x)
                    ),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    past_kv=past_kv[2:] if past_kv is not None else None,
                    cache_present_kv=cache_present_kv,
                    cross_attn_position_bias=cross_attn_position_bias,
                    layer_idx=layer_idx,
                    position_ids=position_ids,
                    special_token_meta=special_token_meta,
                    **extra_args,
                )

                if self.cross_attention_gate_attention:
                    cross_attn_out = (
                        attn2_out[0]
                        * self.cross_attn_attn_gate.broadcast_to(
                            attn2_out[0].shape
                        ).tanh()
                    )
                else:
                    cross_attn_out = attn2_out[0]

                x = x + cross_attn_out

            ffn_output = (
                self.ffn(self.norm3(x), **ffn_extra_args)
                if self.norm3 is not None
                else self.ffn(x, **ffn_extra_args)
            )
            if self.moe_enabled:
                (ffn_output, routing_weights, expert_mask) = ffn_output

            post_ffn_output = ffn_output
            if self.norm_first_sandwich:
                post_ffn_output = self.norm3_post(post_ffn_output)

            if self.cross_attention_gate_mlp:
                post_ffn_output = (
                    post_ffn_output
                    * self.cross_attn_mlp_gate.broadcast_to(
                        post_ffn_output.shape
                    ).tanh()
                )

            if full_text_row_masked_out_mask is not None:
                post_ffn_output = full_text_row_masked_out_mask * post_ffn_output  # type: ignore

            x = x + post_ffn_output
        else:
            if self.disable_self_attention:
                raise NotImplementedError(
                    "norm_first=False and disable_self_attention=True is not supported in TransformerDecoderLayer"
                )

            attn1_out = self._sa_block(
                x,
                tgt_mask,
                tgt_key_padding_mask,
                rotary_position_embedding_helper=rotary_position_embedding_helper,
                past_kv=past_kv[:2] if past_kv is not None else None,
                cache_present_kv=cache_present_kv,
                self_attn_position_bias=self_attn_position_bias,
                layer_idx=layer_idx,
                position_ids=position_ids,
                special_token_meta=special_token_meta,
                **extra_args,
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
                    layer_idx=layer_idx,
                    position_ids=position_ids,
                    special_token_meta=special_token_meta,
                    **extra_args,
                )
                x = self.norm2(x + attn2_out[0])
            ffn_output = self.ffn(x, **ffn_extra_args)
            if self.moe_enabled:
                (ffn_output, routing_weights, expert_mask) = ffn_output

            x = (
                self.norm3(x + ffn_output)
                if self.norm3 is not None
                else x + ffn_output
            )

        if not self.moe_enabled:
            if not cache_present_kv:
                return x
            else:
                present_kv = (
                    attn1_out[1]
                    if not self.add_cross_attention
                    else attn1_out[1] + attn2_out[1]
                )
            return x, present_kv
        else:
            if not cache_present_kv:
                return x, routing_weights, expert_mask
            else:
                present_kv = (
                    attn1_out[1]
                    if not self.add_cross_attention
                    else attn1_out[1] + attn2_out[1]
                )
            return x, routing_weights, expert_mask, present_kv

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
        layer_idx: Optional[int] = None,
        **extra_args,
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
            layer_idx=layer_idx,
            **extra_args,
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
        layer_idx: Optional[int] = None,
        **extra_args,
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
            layer_idx=layer_idx,
            **extra_args,
        )

        if cache_present_kv:
            x, present_kv = attn_out
        else:
            x = attn_out

        out = (self.dropout2(x),)
        if cache_present_kv:
            out += (present_kv,)
        return out
