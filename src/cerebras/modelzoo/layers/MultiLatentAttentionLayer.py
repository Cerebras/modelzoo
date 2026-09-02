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
Multi-head Latent Attention (MLA) from DeepSeek-V2
(https://arxiv.org/abs/2405.04434).

MLA replaces the standard K/V projections with a low-rank compression scheme:
  - A small "latent" vector c_KV is projected from the input (kv_lora_rank dims).
  - K (nope part) and V are projected back up from c_KV.
  - A separate, uncompressed K_rope is projected directly from the input and
    receives decoupled RoPE, while K_nope and Q_nope bypass RoPE entirely.
  - Q can optionally be compressed similarly via q_lora_rank.

This dramatically reduces KV-cache size during inference while maintaining
model quality.

Projection dimensions at a glance:
    kv_down_proj  : embed_dim  → kv_lora_rank
    kv_up_proj    : kv_lora_rank → num_heads*(qk_nope_head_dim + v_head_dim)
    k_rope_proj   : embed_dim  → qk_rope_head_dim  (shared across all heads)
    q_proj        : embed_dim  → num_heads*(qk_nope_head_dim + qk_rope_head_dim)
                    (used when q_lora_rank is None)
    q_down_proj   : embed_dim  → q_lora_rank
    q_up_proj     : q_lora_rank → num_heads*(qk_nope_head_dim + qk_rope_head_dim)
                    (used when q_lora_rank is not None)
    proj_output   : num_heads*v_head_dim → embed_dim

RoPE is applied only to the last qk_rope_head_dim dimensions of Q and K.
Set `rotary_dim: <qk_rope_head_dim>` in the model config to match.
"""

import logging

import torch
import torch.nn as nn

from cerebras.modelzoo.layers.create_initializer import create_initializer

from .AttentionLayer import MultiheadAttention


class MultiLatentAttention(MultiheadAttention):
    """Multi-head Latent Attention (MLA) from DeepSeek-V2.

    Args:
        embed_dim (int): Model hidden size (d_model).
        num_heads (int): Number of attention heads.
        kv_lora_rank (int): Latent rank for KV compression. Default: 512.
        q_lora_rank (int | None): Latent rank for Q compression. ``None``
            uses a standard full-rank Q projection. Default: ``None``.
        qk_nope_head_dim (int): Per-head dimension for the non-RoPE (nope)
            portion of Q and K. Default: 128.
        qk_rope_head_dim (int): Per-head dimension for the decoupled RoPE
            portion of Q and K. The model's ``rotary_dim`` must match this.
            Default: 64.
        v_head_dim (int): Per-head dimension for V. May differ from
            qk_nope_head_dim. Default: 128.
        kv_norm_layer (nn.Module class | None): Norm class (not instance)
            applied to the compressed KV latent before up-projection, e.g.
            ``RMSNorm``. ``None`` skips the norm. Default: ``None``.
        kv_norm_eps (float): Epsilon for kv_norm_layer. Default: 1e-5.
        q_norm_layer (nn.Module class | None): Same as kv_norm_layer but
            applied to the Q latent (only relevant when q_lora_rank is set).
            Default: ``None``.
        q_norm_eps (float): Epsilon for q_norm_layer. Default: 1e-5.

        All remaining keyword arguments are forwarded to ``MultiheadAttention``
        unchanged (dropout, use_projection_bias, initializers, etc.).

    Note:
        ``inner_dim`` passed by ``TransformerDecoderLayer`` is ignored; MLA
        derives its own effective inner_dim as
        ``num_heads * (qk_nope_head_dim + qk_rope_head_dim)``.

        During a single forward pass, the KV down-projection is computed
        twice (once for K, once for V) because ``construct_key_vector`` and
        ``construct_value_vector`` are called independently by the base
        ``forward``. This is correct but slightly redundant; future work can
        fuse them by overriding ``forward``.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        # MLA-specific parameters
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_norm_layer=None,
        kv_norm_eps=1e-5,
        q_norm_layer=None,
        q_norm_eps=1e-5,
        # Standard MultiheadAttention parameters
        inner_dim=None,
        dropout=0.0,
        batch_first=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        use_projection_bias=False,
        use_ffn_bias=False,
        use_sink_token=False,
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
    ):
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # Pass inner_dim = num_heads * qk_head_dim to base class so that
        # existing infrastructure (e.g. layer_idx scaling) uses the right dim.
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inner_dim=num_heads * qk_head_dim,
            dropout=dropout,
            batch_first=batch_first,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            use_projection_bias=use_projection_bias,
            use_ffn_bias=use_ffn_bias,
            use_sink_token=use_sink_token,
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

        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        # Replace base-class Q/K/V projections (wrong dims for MLA)
        del self.proj_q_dense_layer
        del self.proj_k_dense_layer
        del self.proj_v_dense_layer

        # ---------- KV path ----------
        # Step 1: compress input to low-rank latent
        self.kv_down_proj = nn.Linear(
            self.embed_dim,
            kv_lora_rank,
            bias=use_projection_bias,
            device=device,
        )
        # Optional norm on the KV latent (stabilises large-scale training)
        self.kv_norm = (
            kv_norm_layer(kv_lora_rank, eps=kv_norm_eps, device=device)
            if kv_norm_layer is not None
            else nn.Identity()
        )
        # Step 2: project latent → k_nope + v concatenated
        #   first  num_heads*qk_nope_head_dim columns → K nope
        #   last   num_heads*v_head_dim columns        → V
        self.kv_up_proj = nn.Linear(
            kv_lora_rank,
            num_heads * qk_nope_head_dim + num_heads * v_head_dim,
            bias=use_projection_bias,
            device=device,
        )
        # Decoupled K for RoPE: projected directly from input, shared across
        # all query heads (broadcast during key construction).
        self.k_rope_proj = nn.Linear(
            self.embed_dim,
            qk_rope_head_dim,
            bias=use_projection_bias,
            device=device,
        )

        # ---------- Q path ----------
        if q_lora_rank is None:
            # Full-rank Q projection: embed_dim → num_heads * qk_head_dim
            self.q_proj = nn.Linear(
                self.embed_dim,
                num_heads * qk_head_dim,
                bias=use_projection_bias,
                device=device,
            )
            self.q_down_proj = None
            self.q_norm = None
            self.q_up_proj = None
        else:
            self.q_proj = None
            self.q_down_proj = nn.Linear(
                self.embed_dim,
                q_lora_rank,
                bias=use_projection_bias,
                device=device,
            )
            self.q_norm = (
                q_norm_layer(q_lora_rank, eps=q_norm_eps, device=device)
                if q_norm_layer is not None
                else nn.Identity()
            )
            self.q_up_proj = nn.Linear(
                q_lora_rank,
                num_heads * qk_head_dim,
                bias=use_projection_bias,
                device=device,
            )

        # ---------- Output projection ----------
        # Replaces base class proj_output_dense_layer.
        # Input dim is num_heads * v_head_dim (not inner_dim).
        del self.proj_output_dense_layer
        self.proj_output_dense_layer = nn.Linear(
            num_heads * v_head_dim,
            self.embed_dim,
            bias=use_ffn_bias,
            device=device,
        )

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        weight_init = create_initializer(self.initializer)
        q_init = create_initializer(self.query_initializer)
        out_init = create_initializer(self.output_initializer)
        bias_init = create_initializer(self.bias_initializer)

        def _init_linear(layer, w_init):
            w_init(layer.weight.data)
            if layer.bias is not None:
                bias_init(layer.bias.data)

        _init_linear(self.kv_down_proj, weight_init)
        _init_linear(self.kv_up_proj, weight_init)
        _init_linear(self.k_rope_proj, weight_init)

        if self.q_proj is not None:
            _init_linear(self.q_proj, q_init)
        else:
            _init_linear(self.q_down_proj, q_init)
            _init_linear(self.q_up_proj, q_init)

        _init_linear(self.proj_output_dense_layer, out_init)

    # ------------------------------------------------------------------
    # Override Q / K / V construction
    # ------------------------------------------------------------------

    def construct_query_vector(
        self,
        q,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        batch_size, seq_length = q.shape[:2]

        if self.q_proj is not None:
            q_out = self.q_proj(q) * self.q_projection_scale
        else:
            q_latent = self.q_down_proj(q)
            q_latent = self.q_norm(q_latent)
            q_out = self.q_up_proj(q_latent) * self.q_projection_scale

        # Return in rotary format: [B, S, H, qk_head_dim]
        return q_out.view(batch_size, seq_length, self.num_heads, self.qk_head_dim)

    def construct_key_vector(
        self,
        k,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        batch_size, seq_length = k.shape[:2]

        # Compressed K nope via the KV latent
        kv_latent = self.kv_down_proj(k)
        kv_normed = self.kv_norm(kv_latent)
        kv_up = self.kv_up_proj(kv_normed) * self.k_projection_scale

        # [B, S, H, qk_nope_head_dim]
        k_nope = kv_up[..., : self.num_heads * self.qk_nope_head_dim]
        k_nope = k_nope.view(
            batch_size, seq_length, self.num_heads, self.qk_nope_head_dim
        )

        # Decoupled K rope: project from input, share across heads
        # k_rope_proj output: [B, S, qk_rope_head_dim]
        k_rope = self.k_rope_proj(k)
        # Broadcast to all heads: [B, S, H, qk_rope_head_dim]
        k_rope = k_rope.unsqueeze(2).expand(
            batch_size, seq_length, self.num_heads, self.qk_rope_head_dim
        )

        # Concatenate: [B, S, H, qk_head_dim]  (nope first, rope last)
        return torch.cat([k_nope, k_rope], dim=-1)

    def construct_value_vector(
        self,
        v,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        batch_size, seq_length = v.shape[:2]

        # Re-run KV compression (input is the same tensor as k in self-attn)
        kv_latent = self.kv_down_proj(v)
        kv_normed = self.kv_norm(kv_latent)
        kv_up = self.kv_up_proj(kv_normed) * self.v_projection_scale

        # Extract V portion: [B, S, H * v_head_dim]
        v_out = kv_up[..., self.num_heads * self.qk_nope_head_dim :]
        # [B, S, H, v_head_dim] → [B, H, S, v_head_dim]
        v_out = v_out.view(
            batch_size, seq_length, self.num_heads, self.v_head_dim
        ).transpose(1, 2)
        return v_out

    # ------------------------------------------------------------------
    # Override RoPE to apply only to the rope slice of Q/K
    # ------------------------------------------------------------------

    def apply_rotary_position_embedding(
        self,
        vector,
        rotary_position_embedding_helper,
        offset_length,
        constant_pos_mask=None,
        position_ids=None,
        rope_cache_tag=None,
    ):
        """Apply RoPE only to the last ``qk_rope_head_dim`` dimensions.

        ``vector`` has shape ``[B, S, H, qk_head_dim]`` (rotary format from
        ``construct_*_vector``).  After this method it becomes
        ``[B, H, S, qk_head_dim]`` (standard attention format).
        """
        if rotary_position_embedding_helper is None:
            return vector.transpose(1, 2)

        # Split nope / rope slices
        nope = vector[..., : self.qk_nope_head_dim]  # [B, S, H, nope_dim]
        rope = vector[..., self.qk_nope_head_dim :]   # [B, S, H, rope_dim]

        # rotate_tensor expects & returns [B, S, H, rope_dim]
        rope = rotary_position_embedding_helper.rotate_tensor(
            rope,
            offset=offset_length,
            constant_pos_mask=constant_pos_mask,
            position_ids=position_ids,
            rope_cache_tag=rope_cache_tag,
        )

        # Transpose both to [B, H, S, dim] and recombine
        nope = nope.transpose(1, 2)
        rope = rope.transpose(1, 2)
        return torch.cat([nope, rope], dim=-1)  # [B, H, S, qk_head_dim]

    # ------------------------------------------------------------------
    # Override logit scaling to use qk_head_dim instead of inner_dim/H
    # ------------------------------------------------------------------

    def calculate_attention_logits(self, q, k, layer_idx=None):
        if self.attention_type == "scaled_dot_product":
            q = q * torch.tensor(
                1.0 / float(self.qk_head_dim) ** 0.5, dtype=q.dtype
            )

        if self.scale_qk_dot_by_layer_idx:
            q = q * torch.tensor(1.0 / float(layer_idx + 1), dtype=q.dtype)

        logits = self.attention_logits_alpha * self.using_kernel(torch.matmul)(
            q, k.transpose(-1, -2)
        )

        if self.logit_softcapping is not None:
            logits = (
                torch.tanh(logits / self.logit_softcapping)
                * self.logit_softcapping
            )
        return logits

    # ------------------------------------------------------------------
    # Override attention output to handle v_head_dim != qk_head_dim
    # ------------------------------------------------------------------

    def calculate_attention_output(
        self, attention_scores, v, special_token_meta=None
    ):
        # attention_scores: [B, H, Lq, Lk]
        # v:                [B, H, Lk, v_head_dim]
        attn_out = self.using_kernel(torch.matmul)(
            attention_scores, v
        )  # [B, H, Lq, v_head_dim]

        # Combine heads: [B, Lq, H * v_head_dim]
        batch_size, num_heads, seq_length, _ = attn_out.shape
        attn_out = attn_out.transpose(1, 2).reshape(
            batch_size, seq_length, num_heads * self.v_head_dim
        )

        # Output projection: num_heads * v_head_dim → embed_dim
        return (
            self.proj_output_dense_layer(attn_out) * self.output_projection_scale
        )

    # ------------------------------------------------------------------
    # Config validation (called by get_attention_module)
    # ------------------------------------------------------------------

    def check_extra_params(params):
        """Validate and fill defaults for MLA extra_attention_params."""
        defaults = {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }
        for key, default in defaults.items():
            if key not in params:
                params[key] = default
                logging.warning(
                    f"MultiLatentAttention: '{key}' not set in "
                    f"extra_attention_params, defaulting to {default}."
                )
