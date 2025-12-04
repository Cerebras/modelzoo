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

import torch

from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.utils import _get_clones

from .AttentionLayer import MultiheadAttention


class LinformerAttention(MultiheadAttention):
    """
        Applies the attention mechanism according to the linformer (https://arxiv.org/pdf/2006.04768.pdf) paper.
        key vectors and value vectors are reduced on the seq_length level before computing softmax.
    Args:
        # Multihead Attention Arguments
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
        attention_initializer (str): Projection kernel initializer. Defaults to
            ``xavier_uniform``.
        attention_q_initializer: Query projection kernel initializer. If not
            specified, the query will be initialized via ``attention_initializer``
        output_layer_initializer (str or initializer): If not None, use this
            initializer for the output transform layer. Defaults to None.
        attention_type (str): The attention variant to execute. Currently
            accepts ``dot_product`` and ``scaled_dot_product``. Defaults to
            ``scaled_dot_product``.
        softmax_dtype_fp32 (bool): Use an FP32 softmax implementation.
        attention_kernel (str | None): Kernel to use. Uses ``default`` if None.
            See accepted values below.
                ``None`` - Default implementation.
                ``fast_attention`` - Experimental optimized implementation.

        # Linformer Attention Keyword Args
        kv_len (int): sequence length of the k,v to reduce from
        reduced_kv_len_dim (int): sequence length of the k,v to reduce to
        projection_kernel_initializer (str or initializer): Use this
            initializer for the projection kernel of keys and values. Defaults to `xavier_uniform`.
        param_share_style (str): The share-style of the projection kernels. 4 options:
            1. "no-share", every layer, head, has 2 kernels for key&value projection
            2. "share-headwise", every layer has 2 kernels for key&value projection,
                the kernel is shared across heads in each layer
            3. "share-kv", every layer has 1 kernel for both key and value projection
                the kernel is shared across heads in each layer
            4. "share-layerwise", the entire transformer model has one kernel.
                the kernel is tied through tie_kernel_projections() and used across
                layers and heads, key&value share the same kernel as well
            for details please refer to section `4 Model` of the paper
    """

    def __init__(
        self,
        # Multihead Attention Position Args
        embed_dim,
        num_heads,
        # Multihead Attention Keyword Args
        inner_dim=None,
        dropout=0.0,
        batch_first=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        use_projection_bias=None,
        use_ffn_bias=False,
        attention_initializer="xavier_uniform",
        attention_q_initializer=None,
        output_layer_initializer=None,
        attention_type="scaled_dot_product",
        scale_qk_dot_by_d=False,
        logit_softcapping=None,
        attention_logits_alpha=1.0,
        q_projection_scale=1.0,
        k_projection_scale=1.0,
        v_projection_scale=1.0,
        output_projection_scale=1.0,
        softmax_dtype_fp32=True,
        attention_kernel=None,
        device=None,
        # Linformer Attention Keyword Args
        kv_len=256,
        reduced_kv_len_dim=64,
        projection_kernel_initializer="xavier_uniform",
        param_share_style="no-share",
    ):

        super(LinformerAttention, self).__init__(
            embed_dim,
            num_heads,
            inner_dim=inner_dim,
            dropout=dropout,
            batch_first=batch_first,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            use_projection_bias=use_projection_bias,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            attention_q_initializer=attention_q_initializer,
            output_layer_initializer=output_layer_initializer,
            attention_type=attention_type,
            scale_qk_dot_by_d=scale_qk_dot_by_d,
            attention_logits_alpha=attention_logits_alpha,
            q_projection_scale=q_projection_scale,
            k_projection_scale=k_projection_scale,
            v_projection_scale=v_projection_scale,
            output_projection_scale=output_projection_scale,
            softmax_dtype_fp32=softmax_dtype_fp32,
            attention_kernel=attention_kernel,
            device=device,
        )

        self.kv_len = kv_len
        self.param_share_style = param_share_style

        if self.param_share_style == "no-share":
            kernel = torch.nn.Linear(
                self.kv_len, reduced_kv_len_dim, bias=use_projection_bias
            )
            # Ei
            self.key_projection_kernel = _get_clones(kernel, self.num_heads)
            # Fi
            self.value_projection_kernel = _get_clones(kernel, self.num_heads)
        elif self.param_share_style == "share-headwise":
            # E
            self.key_projection_kernel = torch.nn.Linear(
                self.kv_len, reduced_kv_len_dim, bias=use_projection_bias
            )
            # F
            self.value_projection_kernel = torch.nn.Linear(
                self.kv_len, reduced_kv_len_dim, bias=use_projection_bias
            )
        elif self.param_share_style == "share-kv":
            # E = F
            self.key_projection_kernel = torch.nn.Linear(
                self.kv_len, reduced_kv_len_dim, bias=use_projection_bias
            )
            self.value_projection_kernel = self.key_projection_kernel
        elif self.param_share_style == "share-layerwise":
            # tie projections kernels after transformer layers are instantiated and deepcopied
            self.key_projection_kernel = None
            self.value_projection_kernel = None

        self.linformer_initializer = projection_kernel_initializer

        self.__reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # child
        weight_initializer = create_initializer(self.linformer_initializer)
        if self.param_share_style == "no-share":
            for kernel in self.key_projection_kernel:
                weight_initializer(kernel.weight.data)
            for kernel in self.value_projection_kernel:
                weight_initializer(kernel.weight.data)
        elif self.param_share_style == "share-headwise":
            weight_initializer(self.key_projection_kernel.weight.data)
            weight_initializer(self.value_projection_kernel.weight.data)
        elif self.param_share_style == "share-kv":
            weight_initializer(self.key_projection_kernel.weight.data)

        # If `share-layerwise`, initialization should be done at model level

    def construct_query_vector(
        self,
        q,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        # linear projection
        q = self.proj_q_dense_layer(q)

        # 4D attn_mask [batch_size, 1, 1, len]
        if attn_mask is not None:
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(torch.bool)
            # 2D [batch_size, seq_len]
            attn_mask = attn_mask.squeeze().unsqueeze(-1)
            q = q.masked_fill_(attn_mask, 0.0)

        # split into heads
        q = self._split_heads(q, rotary=True)
        return q

    def construct_key_vector(
        self,
        k,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        # linear projection
        k = self.proj_k_dense_layer(k)

        # 4D attn_mask [batch_size, 1, 1, len]
        if attn_mask is not None:
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(torch.bool)
            # 2D [batch_size, seq_len]
            attn_mask = attn_mask.squeeze().unsqueeze(-1)
            k = k.masked_fill_(attn_mask, 0.0)

        # split into heads
        k = self._split_heads(k, rotary=True)
        return k

    def construct_value_vector(
        self,
        v,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        # linear projection
        v = self.proj_v_dense_layer(v)

        # 4D attn_mask [batch_size, 1, 1, len]
        if attn_mask is not None:
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(torch.bool)
            # 2D [batch_size, seq_len]
            attn_mask = attn_mask.squeeze().unsqueeze(-1)
            v = v.masked_fill_(attn_mask, 0.0)

        # split into heads
        v = self._split_heads(v, rotary=False)
        return v

    def process_k_before_logits_calc(self, k):
        # K has shape [batch_size, num_heads, seq_length, head_dim]
        assert (
            k.shape[-2] == self.kv_len
        ), "Linformer key dimension mismatch, seq_length should be at dim -2"
        # outshape [batch_size, num_heads, head_dim, seq_length]
        k = k.transpose(-2, -1)
        if self.param_share_style == "no-share":
            k_by_heads = []
            for i in range(self.num_heads):
                k_headwise = k[:, i, :, :]
                k_by_heads.append(self.key_projection_kernel[i](k_headwise))
            k = torch.stack(k_by_heads, dim=1)
            k = k.transpose(-2, -1)
        elif (
            self.param_share_style == "share-headwise"
            or self.param_share_style == "share-kv"
            or self.param_share_style == "share-layerwise"
        ):
            if self.param_share_style == "share-layerwise":
                assert (
                    self.key_projection_kernel is not None
                ), "Should call tie_kernel_projections to pass external projection kernel"
            k = self.key_projection_kernel(k)
            k = k.transpose(-2, -1)

        # K has shape [batch_size, num_heads, reduced_kv_len_dim, head_dim]
        return k

    def process_v_before_logits_calc(self, v):
        # V has shape [batch_size, num_heads, seq_length, head_dim]
        assert (
            v.shape[-2] == self.kv_len
        ), "Linformer key dimension mismatch, seq_length should be at dim -2"
        # outshape [batch_size, num_heads, head_dim, seq_length]
        v = v.transpose(-2, -1)
        if self.param_share_style == "no-share":
            v_by_heads = []
            for i in range(self.num_heads):
                v_headwise = v[:, i, :, :]
                v_by_heads.append(self.value_projection_kernel[i](v_headwise))
            v = torch.stack(v_by_heads, dim=1)
            v = v.transpose(-2, -1)
        elif (
            self.param_share_style == "share-headwise"
            or self.param_share_style == "share-kv"
            or self.param_share_style == "share-layerwise"
        ):
            if self.param_share_style == "share-layerwise":
                assert (
                    self.value_projection_kernel is not None
                ), "Should call tie_kernel_projections to pass external projection kernel"
            v = self.value_projection_kernel(v)
            v = v.transpose(-2, -1)

        # V has shape [batch_size, num_heads, reduced_kv_len_dim, head_dim]
        return v

    def process_attention_mask(self, attn_mask, past_kv, q):
        return None

    def process_key_padding_mask(self, key_padding_mask, attn_mask, past_kv, q):
        return None

    def check_extra_params(params):
        condition_exist = (
            "kv_len" in params
            and "reduced_kv_len_dim" in params
            and "param_share_style" in params
        )
        assert (
            condition_exist
        ), "extra params need to contain necessary elements `kv_len`, `reduced_kv_len_dim` and `param_share_style` for linformer attention"

    def tie_kernel_projections(self, kernel_projection):
        """
        Used to tie external kernel for layerwise sharing
        """
        assert (
            self.param_share_style == "share-layerwise"
        ), "Only tie kernel projections across layers in layerwise sharing style"
        self.key_projection_kernel = kernel_projection
        self.value_projection_kernel = self.key_projection_kernel
