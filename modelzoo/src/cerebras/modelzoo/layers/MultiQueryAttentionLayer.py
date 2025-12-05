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

import logging

import torch
import torch.nn as nn

from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.pytorch.utils.num_kv_groups import groups_annotater

from .AttentionLayer import MultiheadAttention


class MultiQueryAttention(MultiheadAttention):
    """Implements the Multi-Query Attention Layer from
        `Fast Transformer Decoding: One Write-Head is All You Need
        <https://arxiv.org/abs/1911.02150>`

    Args:
        embed_dim (int): Number of input units in each projection output
        num_heads (int): Number of attention heads.
        inner_dim (int): Number of output units in attention query/key/value projection. Defaults to ``embed_dim``.
        dropout (float): Dropout rate for key-query weights. Defaults to 0.0.
        batch_first (bool): If True, then the input and output tensors are
            provided as (batch, seq, feature), otherwise the format will be
            (seq, batch, feature). Default: True (batch, seq, feature).
        add_bias_kv (bool): If specified, adds bias to the key and value sequences at dim=0. Default: False.
        add_zero_attn (bool): If specified, adds a new batch of zeros to the key and value
            sequences at dim=1. Default: False
        kdim (int):  Number of output units in key projection
        vdim (int):  Number of output units in  projection
        use_projection_bias (bool): Whether to use bias in the key, query, and
            value projections.
        use_ffn_bias (bool): Whether to use bias in the output projection.
        attention_qk_norm_layer (nn.Module): Norm layer for applying qk normalization
        attention_qk_norm_eps (float): epsilon for norm layer for applying qk normalization
        attention_initializer (str): Projection kernel initializer. Defaults to
            ``xavier_uniform``.
        attention_q_initializer: Query projection kernel initializer. If not
            specified, the query will be initialized via ``attention_initializer``
        output_layer_initializer (str or initializer): If not None, use this
            initializer for the output transform layer. Defaults to None.
        bias_initializer (str): Bias initializer. Defaults to ``zeros``.
        attention_type (str): The attention variant to execute. Currently
            accepts ``dot_product`` and ``scaled_dot_product``. Defaults to
            ``scaled_dot_product``.
        softmax_dtype_fp32 (bool): Use an FP32 softmax implementation.
        attention_kernel (str | None): Kernel to use. Uses ``default`` if None.
            See accepted values below.
                ``None`` - Default implementation.
                ``fast_attention`` - Experimental optimized implementation.
        device (optional): Device to create the model parameters on, can be a cuda device or CS device.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        inner_dim=None,
        dropout=0.0,
        batch_first=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        use_projection_bias=None,
        use_ffn_bias=False,
        attention_qk_norm_layer=None,
        attention_qk_norm_eps=1e-5,
        attention_initializer="xavier_uniform",
        attention_q_initializer=None,
        output_layer_initializer=None,
        bias_initializer="zeros",
        attention_type="scaled_dot_product",
        scale_qk_dot_by_d=False,
        attention_logits_alpha=1.0,
        q_projection_scale=1.0,
        k_projection_scale=1.0,
        v_projection_scale=1.0,
        output_projection_scale=1.0,
        softmax_dtype_fp32=True,
        attention_kernel=None,
        scale_qk_dot_by_layer_idx=False,
        logit_softcapping=None,
        device=None,
        # MQA specific
        num_kv_groups=1,
    ):
        super(MultiQueryAttention, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inner_dim=inner_dim,
            dropout=dropout,
            batch_first=batch_first,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            vdim=vdim,
            kdim=kdim,
            use_projection_bias=use_projection_bias,
            use_ffn_bias=use_ffn_bias,
            attention_qk_norm_layer=attention_qk_norm_layer,
            attention_qk_norm_eps=attention_qk_norm_eps,
            attention_initializer=attention_initializer,
            attention_q_initializer=attention_q_initializer,
            output_layer_initializer=output_layer_initializer,
            bias_initializer=bias_initializer,
            attention_type=attention_type,
            scale_qk_dot_by_d=scale_qk_dot_by_d,
            attention_logits_alpha=attention_logits_alpha,
            q_projection_scale=q_projection_scale,
            k_projection_scale=k_projection_scale,
            v_projection_scale=v_projection_scale,
            output_projection_scale=output_projection_scale,
            softmax_dtype_fp32=softmax_dtype_fp32,
            attention_kernel=attention_kernel,
            scale_qk_dot_by_layer_idx=scale_qk_dot_by_layer_idx,
            logit_softcapping=logit_softcapping,
            device=device,
        )

        self.head_dim = self.inner_dim // self.num_heads
        self.num_kv_groups = num_kv_groups
        self.per_group_num_heads = self.num_heads // self.num_kv_groups
        self.using_groups = groups_annotater(self.num_kv_groups)

        assert (
            self.num_heads % self.num_kv_groups == 0
        ), f"num_heads has to be a multiple of num_kv_groups but got {self.num_heads} and {self.num_kv_groups}"

        # assuming only 1 head for key and value projections
        self.proj_k_dense_layer = nn.Linear(
            self.kdim,
            self.num_kv_groups * self.head_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_v_dense_layer = nn.Linear(
            self.vdim,
            self.num_kv_groups * self.head_dim,
            bias=use_projection_bias,
            device=device,
        )
        # reset newly initialized parameters
        self.__reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # bias initialization
        bias_initializer = create_initializer(self.bias_initializer)
        if self.use_projection_bias:
            bias_initializer(self.proj_k_dense_layer.bias.data)
            bias_initializer(self.proj_v_dense_layer.bias.data)

        # k, v projections
        weight_initializer = create_initializer(self.initializer)
        weight_initializer(self.proj_k_dense_layer.weight.data)
        weight_initializer(self.proj_v_dense_layer.weight.data)

    def construct_key_vector(
        self,
        k,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        # linear projection
        k = self.get_key_projection(
            k,
            special_token_meta=special_token_meta,
        )  # [batch_size, seq_length, self.num_kv_groups * self.head_dim]

        if self.num_kv_groups == 1:
            return torch.unsqueeze(
                k, 2
            )  # [batch_size, seq_length, 1, kv_channels]

        batch_size, seq_length, _ = k.shape
        # [batch_size, seq_length, self.num_kv_groups, self.head_dim]
        k = k.reshape(batch_size, seq_length, self.num_kv_groups, self.head_dim)

        return k

    def construct_value_vector(
        self,
        v,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        # linear projection
        v = self.get_value_projection(
            v, special_token_meta=special_token_meta
        )  # [batch_size, seq_length, self.num_kv_groups * self.head_dim]

        if self.num_kv_groups == 1:
            return torch.unsqueeze(
                v, 1
            )  # [batch_size, 1, seq_length, kv_channels]

        batch_size, seq_length, _ = v.shape
        v = v.reshape(batch_size, seq_length, self.num_kv_groups, self.head_dim)
        v = v.transpose(2, 1)

        # [batch_size, self.num_kv_groups, seq_length, self.head_dim]
        return v

    def expand_kv_over_group_dim(self, x):
        # expand k/v over dimension
        batch_size, _, seq_length, _ = x.shape

        x = x.unsqueeze(
            2
        )  # [batch_size, self.num_kv_groups, 1, seq_length, self.head_dim]
        # expand over per_group_num_heads
        x = x.expand(
            batch_size,
            self.num_kv_groups,
            self.per_group_num_heads,
            seq_length,
            self.head_dim,
        )
        x = x.reshape(batch_size, self.num_heads, seq_length, self.head_dim)
        return x

    def calculate_attention_logits(self, q, k, layer_idx):
        if self.num_kv_groups > 1:
            k = self.expand_kv_over_group_dim(k)

        return super().calculate_attention_logits(q, k, layer_idx)

    def calculate_attention_output(
        self, attention_scores, v, special_token_meta=None
    ):
        if self.num_kv_groups > 1:
            v = self.expand_kv_over_group_dim(v)

        return super().calculate_attention_output(
            attention_scores, v, special_token_meta
        )

    def check_extra_params(params):
        if "num_kv_groups" not in params:
            params["num_kv_groups"] = 1
            logging.warning(
                "num_kv_groups is not set in the yaml, it is set to 1 by default. "
                "Please provide a value if this is not intended."
            )
