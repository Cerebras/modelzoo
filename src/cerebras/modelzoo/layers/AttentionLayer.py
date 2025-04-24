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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.pytorch.utils.kernel import kernel_annotater
from cerebras.pytorch.utils.num_kv_groups import groups_annotater


class MultiheadAttention(nn.Module):
    """Multi-head attention layer. Adapted from:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention.

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
        kdim (int):  Number of input units in the key projection
        vdim (int):  Number of input units in the value projection
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
        scale_qk_dot_by_d (bool): If ``True`` scales QK^T dot product by d(=hidden/d_head) instead of sqrt(d).
        attention_logits_alpha (float): Scales the QK^T dot product. Used to stabilize logits in muP training.
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
        use_projection_bias=False,
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
    ):
        _SUPPORTED_ATTENTION_TYPES = [
            "dot_product",
            "scaled_dot_product",
            "scaled_cosine",
        ]
        assert (
            attention_type in _SUPPORTED_ATTENTION_TYPES
        ), f"Attention type {attention_type} is not supported."
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}."

        if inner_dim is not None:
            assert (
                inner_dim % num_heads == 0
            ), "inner_dim must be divisible by num_heads."

        assert batch_first, "Currently, only batch_first=True is supported"
        assert not add_bias_kv, "add_bias_kv=True is not supported."
        assert not add_zero_attn, "add_zero_attn=True is not supported."
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.inner_dim = inner_dim if inner_dim is not None else embed_dim

        self.num_heads = num_heads
        self.attention_type = attention_type

        self.use_projection_bias = use_projection_bias
        self.use_ffn_bias = use_ffn_bias

        self.proj_q_dense_layer = nn.Linear(
            self.embed_dim,
            self.inner_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_k_dense_layer = nn.Linear(
            self.kdim,
            self.inner_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_v_dense_layer = nn.Linear(
            self.vdim,
            self.inner_dim,
            bias=use_projection_bias,
            device=device,
        )

        self.q_norm = None
        self.k_norm = None
        if attention_qk_norm_layer is not None and not isinstance(
            attention_qk_norm_layer, nn.Identity
        ):
            head_dim = self.inner_dim // self.num_heads
            self.q_norm = attention_qk_norm_layer(
                head_dim,
                eps=attention_qk_norm_eps,
                device=device,
            )
            self.k_norm = attention_qk_norm_layer(
                head_dim,
                eps=attention_qk_norm_eps,
                device=device,
            )

        if self.attention_type == "scaled_cosine":
            self.logits_scale = nn.Parameter(
                torch.log(10 * torch.ones((self.num_heads, 1, 1)))
            )

        self.dropout_layer = nn.Dropout(dropout)

        self.proj_output_dense_layer = nn.Linear(
            self.inner_dim,
            self.embed_dim,
            bias=use_ffn_bias,
            device=device,
        )

        # handle initialization
        output_initializer = attention_initializer
        if output_layer_initializer is not None:
            output_initializer = output_layer_initializer

        self.initializer = attention_initializer
        self.query_initializer = self.initializer
        if attention_q_initializer is not None:
            self.query_initializer = attention_q_initializer
        self.output_initializer = output_initializer
        self.bias_initializer = bias_initializer
        self.softmax_dtype_fp32 = softmax_dtype_fp32
        if attention_kernel:
            attention_kernel = attention_kernel.upper()
        self.using_kernel = kernel_annotater(attention_kernel)
        self.using_groups = groups_annotater(None)

        self.scale_qk_dot_by_d = scale_qk_dot_by_d
        self.attention_logits_alpha = attention_logits_alpha
        self.q_projection_scale = q_projection_scale
        self.k_projection_scale = k_projection_scale
        self.v_projection_scale = v_projection_scale
        self.output_projection_scale = output_projection_scale
        self.scale_qk_dot_by_layer_idx = scale_qk_dot_by_layer_idx
        self.logit_softcapping = logit_softcapping

        self.sparse_attn_mask_ranges = None

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        # bias initialization
        bias_initializer = create_initializer(self.bias_initializer)
        if self.use_projection_bias:
            bias_initializer(self.proj_q_dense_layer.bias.data)
            bias_initializer(self.proj_k_dense_layer.bias.data)
            bias_initializer(self.proj_v_dense_layer.bias.data)
        if self.use_ffn_bias:
            bias_initializer(self.proj_output_dense_layer.bias.data)

        # q projection
        weight_initializer = create_initializer(self.query_initializer)
        weight_initializer(self.proj_q_dense_layer.weight.data)

        # k, v projections
        weight_initializer = create_initializer(self.initializer)
        weight_initializer(self.proj_k_dense_layer.weight.data)
        weight_initializer(self.proj_v_dense_layer.weight.data)

        # output projections
        weight_initializer = create_initializer(self.output_initializer)
        weight_initializer(self.proj_output_dense_layer.weight.data)

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        average_attn_weights=True,
        past_kv=None,
        cache_present_kv=False,
        past_kv_self_attn=True,
        position_bias=None,
        rotary_position_embedding_helper=None,
        layer_idx=None,
        position_ids=None,
        special_token_meta=None,
        **extra_args,
    ):
        """Applies the attention mechanism to queries ``q``, keys ``k`` and values ``v``.

        Args:
            q (Tensor): Queries, shape ``[batch_size, seq_length, embed_dim]``.
            k (Tensor): Keys, shape ``[batch_size, seq_length, embed_dim]``.
            v (Tensor): Values, shape ``[batch_size, seq_length, embed_dim]``.
            attn_mask (Tensor): Attention mask. Can be 2D of shape
                ``[batch_size, seq_length]``, or 3D of shape
                ``[batch, query_length, seq_length]``.
            key_padding_mask (Tensor): If specified, a mask of shape (N, S) indicating
                which elements within key to ignore for the purpose of attention
                (i.e. treat as “padding”). Defaults to None.
            need_weights (bool): If specified, returns attn_output_weights in addition
                to attn_outputs. Default: False.
            average_attn_weights (bool): If true, indicates that the returned attn_weights
                should be averaged across heads. Otherwise, attn_weights are provided
                separately per head. Note that this flag only has an effect when
                need_weights=True. Default: True (i.e. average weights across heads)
            past_kv (tuple(tensor, tensor)): Past keys and values. Tensors have shape
                ``[batch_size, num_heads, seq_length, embed_dim / num_heads]``.
                The 0th and 1st tensor contain the past keys and values, respectively.
                Defaults to ``None``.
            cache_present_kv (bool): Specifies if the present keys and values
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. Defaults to ``False``.
            past_kv_self_attn (bool): Specifies whether the past keys & values should be
                used for self-attention (true) or cross-attention (false). Ignored if
                past_kv is not provided. Default: True
            position_bias (Tensor): Tensor containing position bias to apply in attention
                with shape ``[num_heads, query_length, key_length]``.
            rotary_position_embedding_helper (Optional[RotaryPositionEmbeddingHelper]):
                A helper class to apply rotary embedding on the input tensor.

        Returns:
            Attention output tensor with shape ``[batch_size, seq_length, embed_dim]``.
        """
        assert not (
            rotary_position_embedding_helper and position_bias
        ), "Cannot specify both rotary and relative position embeddings, pick one!"

        assert (
            past_kv is None and not cache_present_kv
        ), "Cannot provide past_kv because inference is not supported yet."

        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = q.shape[:2]
        real_seq_length = seq_length

        assert (
            real_seq_length > 1
        ), "Sequence length 1 is currently unsupported."

        constant_pos_mask = None
        if extra_args and ("constant_pos_mask" in extra_args):
            constant_pos_mask = extra_args["constant_pos_mask"]

        # construct query, key and value vector with a linear projection and split into heads
        q = self.construct_query_vector(
            q,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            special_token_meta=special_token_meta,
        )
        k = self.construct_key_vector(
            k,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            special_token_meta=special_token_meta,
        )
        v = self.construct_value_vector(
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            special_token_meta=special_token_meta,
        )

        # Work with KV cache before RoPE modification of keys
        k, v = self.process_past_kv(past_kv, past_kv_self_attn, k, v)
        present_kv = self.construct_present_kv(cache_present_kv, k, v)

        offset_length, real_seq_length = self.get_sequence_length(
            past_kv, real_seq_length
        )

        # Scale k for muP transfer before Transpose to get around the compile issue
        if (
            self.scale_qk_dot_by_d
            and self.attention_type == "scaled_dot_product"
        ):
            depth = self.inner_dim // self.num_heads
            k = k * torch.tensor(1 / float(depth) ** 0.5, dtype=k.dtype)

        # rotary embedding helper
        k_rotated = self.apply_rotary_position_embedding(
            k,
            rotary_position_embedding_helper,
            offset_length,
            constant_pos_mask=constant_pos_mask,
            position_ids=position_ids,
        )
        q_rotated = self.apply_rotary_position_embedding(
            q,
            rotary_position_embedding_helper,
            offset_length,
            constant_pos_mask=constant_pos_mask,
            position_ids=position_ids,
        )
        # q, k now have shape [batch_size, num_heads, seq_length, head_dim]

        if (
            rotary_position_embedding_helper is not None
            and not rotary_position_embedding_helper.is_rel_distance_default
        ):
            # We are using capped/grouped relative distances for RoPE so we need to
            # compute a separate set of QK to get logit values outside the sliding window region
            q_pos_id, k_pos_id = (
                rotary_position_embedding_helper.get_distant_pos_id_vectors(
                    device=q.device
                )
            )
            k_distant = self.apply_rotary_position_embedding(
                k,
                rotary_position_embedding_helper,
                offset_length,
                constant_pos_mask=constant_pos_mask,
                position_ids=k_pos_id,
                rope_cache_tag="key_distant",
            )
            q_distant = self.apply_rotary_position_embedding(
                q,
                rotary_position_embedding_helper,
                offset_length,
                constant_pos_mask=constant_pos_mask,
                position_ids=q_pos_id,
                rope_cache_tag="query_distant",
            )
            q_distant = self.process_q_before_logits_calc(q_distant)
            k_distant = self.process_k_before_logits_calc(k_distant)

        q = self.process_q_before_logits_calc(q_rotated)
        k = self.process_k_before_logits_calc(k_rotated)
        v = self.process_v_before_logits_calc(v)

        logits = self.calculate_attention_logits(q, k, layer_idx)

        if (
            rotary_position_embedding_helper is not None
            and not rotary_position_embedding_helper.is_rel_distance_default
        ):
            # Compute logits outside the sliding window region and combine with original logits
            logits_distant = self.calculate_attention_logits(
                q_distant, k_distant, layer_idx
            )
            mask_local, mask_distant = (
                rotary_position_embedding_helper.get_attn_region_masks(
                    shape=q_distant.shape,
                    device=q_distant.device,
                )
            )
            logits = mask_local * logits + mask_distant * logits_distant

        attn_mask_processed = self.process_attention_mask(attn_mask, past_kv, q)
        key_padding_mask_processed = self.process_key_padding_mask(
            key_padding_mask, attn_mask, past_kv, q
        )

        attention_bias = self.combine_masks(
            attn_mask_processed, key_padding_mask_processed
        )

        logits = self.apply_position_bias(logits, position_bias)
        logits = self.apply_attention_bias(logits, attention_bias)

        attention_scores = self.calculate_attention_scores(logits)
        attention_output = self.calculate_attention_output(
            attention_scores,
            v,
            special_token_meta=special_token_meta,
        )

        if cache_present_kv:
            return attention_output, present_kv

        if not need_weights:
            return attention_output
        else:
            if average_attn_weights:
                attention_scores = torch.mean(attention_scores, dim=1).squeeze()
            return (
                attention_output,
                attention_scores,
            )

    def _split_heads(self, x, rotary):
        """Split x into different heads, and transpose the resulting value. The
        tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
            x: A tensor with shape ``[batch_size, seq_length, hidden_size]``.

        Returns:
            If rotary is true, a tensor with shape
            ``[batch_size, seq_length, num_heads, hidden_size/num_heads]``
            else, a tensor with shape
            ``[batch_size, num_heads, seq_length, hidden_size/num_heads]``
        """
        batch_size, seq_length, hidden_size = x.shape
        depth = hidden_size // self.num_heads

        # Transpose the result if not rotary
        if rotary:
            return x.view(batch_size, seq_length, self.num_heads, depth)
        return x.view(batch_size, seq_length, self.num_heads, depth).transpose(
            1, 2
        )

    def _combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
            x: A tensor with shape
            ``[batch_size, num_heads, seq_length, embed_dim/num_heads]``.

        Returns:
            A tensor with shape ``[batch_size, seq_length, embed_dim]``.
        """
        batch_size, num_heads, seq_length, depth = x.shape
        return x.transpose(1, 2).reshape(
            batch_size, seq_length, num_heads * depth
        )

    def get_query_projection(self, q, special_token_meta=None):
        # linear projection
        return self.proj_q_dense_layer(q)

    def get_key_projection(self, k, special_token_meta=None):
        # linear projection
        return self.proj_k_dense_layer(k)

    def get_value_projection(self, v, special_token_meta=None):
        # linear projection
        return self.proj_v_dense_layer(v)

    def construct_query_vector(
        self,
        q,
        attn_mask=None,
        key_padding_mask=None,
        special_token_meta=None,
    ):
        q = (
            self.get_query_projection(q, special_token_meta=special_token_meta)
            * self.q_projection_scale
        )
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
        k = (
            self.get_key_projection(k, special_token_meta)
            * self.k_projection_scale
        )
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
        v = (
            self.get_value_projection(v, special_token_meta)
            * self.v_projection_scale
        )
        # split into heads
        v = self._split_heads(v, rotary=False)
        return v

    def get_sequence_length(self, past_kv, real_seq_length):
        offset_length = 0
        if past_kv is not None:
            offset_length = past_kv[0].shape[-2]
            real_seq_length += offset_length
        return offset_length, real_seq_length

    def apply_rotary_position_embedding(
        self,
        vector,
        rotary_position_embedding_helper,
        offset_length,
        constant_pos_mask=None,
        position_ids=None,
        rope_cache_tag=None,
    ):
        if rotary_position_embedding_helper:
            vector = rotary_position_embedding_helper.rotate_tensor(
                vector,
                offset=offset_length,
                constant_pos_mask=constant_pos_mask,
                position_ids=position_ids,
                rope_cache_tag=rope_cache_tag,
            )
        vector = vector.transpose(1, 2)
        return vector

    def process_q_before_logits_calc(self, q):
        # May get overriden but other attention schemas
        return q

    def process_k_before_logits_calc(self, k):
        # May get overriden but other attention schemas
        return k

    def process_v_before_logits_calc(self, v):
        # May get overriden but other attention schemas
        return v

    def process_past_kv(self, past_kv, past_kv_self_attn, k, v):
        if past_kv is not None:
            k_past, v_past = past_kv[0], past_kv[1]
            if past_kv_self_attn:
                k = torch.cat([k_past, k], dim=-2)
                v = torch.cat([v_past, v], dim=-2)
            else:
                k, v = k_past, v_past
        return k, v

    def construct_present_kv(self, cache_present_kv, k, v):
        present_kv = None
        if cache_present_kv:
            present_kv = (k, v)
        return present_kv

    def calculate_attention_logits(self, q, k, layer_idx=None):
        if self.attention_type == "scaled_dot_product":
            depth = self.inner_dim // self.num_heads
            q = q * torch.tensor(
                1 / float(depth) ** 0.5,
                dtype=q.dtype,
            )
        elif self.attention_type == "scaled_cosine":
            q = F.normalize(q, p=2.0, dim=-1)
            k = F.normalize(k, p=2.0, dim=-1)

        if self.scale_qk_dot_by_layer_idx:
            q = q * torch.tensor(
                1 / float(layer_idx + 1),
                dtype=q.dtype,
            )

        # calculate dot product attention
        logits = self.attention_logits_alpha * self.using_groups(
            self.using_kernel(torch.matmul)
        )(
            q, k.transpose(-1, -2)
        )  # (B, H, Lq, E) * (B, H, E, Lk) -> (B, H, Lq, Lk)

        if self.attention_type == "scaled_cosine":
            logits_scale = torch.clamp(
                self.logits_scale, max=math.log(1.0 / 0.01)
            ).exp()
            logits = logits * logits_scale

        if self.logit_softcapping is not None:
            logits = (
                torch.tanh(logits / self.logit_softcapping)
                * self.logit_softcapping
            )

        return logits

    def process_attention_mask(self, attn_mask, past_kv, q):
        attn_mask_reshaped = None

        # apply attention mask
        if attn_mask is not None:
            # 2D [query_length, sequence_length]
            # 3D [batch_size, query_length, sequence_length]
            # 4D [batch_size, num_heads, query_length, sequence_length]
            assert len(attn_mask.shape) in [
                2,
                3,
                4,
            ], "Only 2D, 3D or 4D masks are supported for now"

            if (
                not attn_mask.is_floating_point()
                and not attn_mask.dtype == torch.bool
            ):
                attn_mask = attn_mask.to(torch.bool)

            # for broadcasting over all heads
            num_heads = 1
            if len(attn_mask.shape) == 2:
                if past_kv is not None:
                    past_mask = torch.zeros(
                        (q.shape[0], past_kv.shape[-2]),
                        dtype=attn_mask.dtype,
                    )
                    attn_mask = torch.cat([past_mask, attn_mask], axis=-1)
                query_length, all_seq_length = attn_mask.shape
                # for broadcasting over all batches
                batch_size = 1
            elif len(attn_mask.shape) == 3:
                if past_kv is not None:
                    past_mask = torch.zeros(
                        (q.shape[0], q.shape[-2], past_kv.shape[-2]),
                        dtype=attn_mask.dtype,
                    )
                    attn_mask = torch.cat([past_mask, attn_mask], axis=-1)
                batch_size, query_length, all_seq_length = attn_mask.shape
            else:
                num_heads = attn_mask.shape[1]
                if past_kv is not None:
                    past_mask = torch.zeros(
                        (q.shape[0], num_heads, q.shape[-2], past_kv.shape[-2]),
                        dtype=attn_mask.dtype,
                    )
                    attn_mask = torch.cat([past_mask, attn_mask], axis=-1)
                (
                    batch_size,
                    num_heads,
                    query_length,
                    all_seq_length,
                ) = attn_mask.shape

            # compute the attention_bias based on the mask.
            attn_mask_reshaped = attn_mask.view(
                batch_size, num_heads, query_length, all_seq_length
            )

        return attn_mask_reshaped

    def process_key_padding_mask(self, key_padding_mask, attn_mask, past_kv, q):
        key_padding_mask_reshaped = None

        if key_padding_mask is not None:
            if (
                not key_padding_mask.is_floating_point()
                and not key_padding_mask.dtype == torch.bool
            ):
                key_padding_mask = key_padding_mask.to(torch.bool)

            num_heads = 1
            query_length = 1
            if len(key_padding_mask.shape) == 2:
                if past_kv is not None:
                    past_mask = torch.zeros(
                        (q.shape[0], past_kv.shape[-2]),
                        dtype=key_padding_mask.dtype,
                    )
                    key_padding_mask = torch.cat(
                        [past_mask, key_padding_mask], axis=-1
                    )
                batch_size, all_seq_length = key_padding_mask.shape
            elif len(key_padding_mask.shape) == 3:
                if past_kv is not None:
                    past_mask = torch.zeros(
                        (q.shape[0], q.shape[-2], past_kv.shape[-2]),
                        dtype=key_padding_mask.dtype,
                    )
                    key_padding_mask = torch.cat(
                        [past_mask, key_padding_mask], axis=-1
                    )
                (
                    batch_size,
                    query_length,
                    all_seq_length,
                ) = key_padding_mask.shape
            else:
                num_heads = key_padding_mask.shape[1]
                if past_kv is not None:
                    past_mask = torch.zeros(
                        (q.shape[0], num_heads, q.shape[-2], past_kv.shape[-2]),
                        dtype=key_padding_mask.dtype,
                    )
                    key_padding_mask = torch.cat(
                        [past_mask, key_padding_mask], axis=-1
                    )
                (
                    batch_size,
                    num_heads,
                    query_length,
                    all_seq_length,
                ) = key_padding_mask.shape

            # compute the attention_bias based on the mask.
            key_padding_mask_reshaped = key_padding_mask.view(
                batch_size, num_heads, query_length, all_seq_length
            )

        return key_padding_mask_reshaped

    def combine_masks(self, attn_mask_reshaped, key_padding_mask_reshaped):
        attention_bias = None

        if (
            attn_mask_reshaped is not None
            and key_padding_mask_reshaped is not None
        ):
            # Need to broadcast over dimensions before merging
            (
                attn_mask_reshaped,
                key_padding_mask_reshaped,
            ) = torch.broadcast_tensors(
                attn_mask_reshaped, key_padding_mask_reshaped
            )

            # Need to merge attention mask and key padding mask:
            attn_mask_is_float = attn_mask_reshaped.is_floating_point()
            key_padding_is_float = key_padding_mask_reshaped.is_floating_point()

            if attn_mask_is_float and key_padding_is_float:
                attention_bias = attn_mask_reshaped + key_padding_mask_reshaped
            elif attn_mask_is_float:
                mask_neg_inf = torch.tensor(
                    float("-inf"), dtype=attn_mask_reshaped.dtype
                )
                attention_bias = attn_mask_reshaped.masked_fill(
                    key_padding_mask_reshaped, mask_neg_inf
                )
            elif key_padding_is_float:
                mask_neg_inf = torch.tensor(
                    float("-inf"), dtype=key_padding_mask_reshaped.dtype
                )
                attention_bias = key_padding_mask_reshaped.masked_fill(
                    attn_mask_reshaped, mask_neg_inf
                )
            else:
                attention_bias = attn_mask_reshaped.logical_or(
                    key_padding_mask_reshaped
                )
        elif attn_mask_reshaped is not None:
            attention_bias = attn_mask_reshaped
        elif key_padding_mask_reshaped is not None:
            attention_bias = key_padding_mask_reshaped

        return attention_bias

    def apply_attention_bias(self, logits, attention_bias):
        if attention_bias is not None:
            if attention_bias.dtype == torch.bool:
                final_attention_bias = torch.zeros_like(
                    attention_bias, dtype=logits.dtype
                )
                mask_neg_inf = torch.tensor(
                    float("-inf"), dtype=final_attention_bias.dtype
                )
                final_attention_bias.masked_fill_(attention_bias, mask_neg_inf)
                attention_bias = final_attention_bias
            logits += attention_bias.type_as(logits).broadcast_to(logits.shape)
        return logits

    def apply_position_bias(self, logits, position_bias):
        # Add relative position bias, if any
        if position_bias is not None:
            logits += position_bias.type_as(logits).broadcast_to(logits.shape)
        return logits

    def calculate_attention_scores(self, logits):
        from cerebras.modelzoo.common.utils.model.attn_mask_ranges import (
            mask_range,
        )

        @mask_range(self.sparse_attn_mask_ranges)
        def apply_softmax():
            if self.softmax_dtype_fp32 and logits.dtype != torch.float32:
                attention_scores = nn.functional.softmax(
                    logits.float(), dim=-1
                ).type_as(logits)
            else:
                attention_scores = nn.functional.softmax(logits, dim=-1)
            return attention_scores

        attention_scores = apply_softmax()
        attention_scores = self.dropout_layer(attention_scores)
        return attention_scores

    def get_attention_output_projection(
        self, attention_output, special_token_meta=None
    ):
        return self.proj_output_dense_layer(attention_output)

    def calculate_attention_output(
        self, attention_scores, v, special_token_meta=None
    ):
        # Shape: (batch_size, num_heads, query_length, embed_dim / num_heads)
        attention_output = self.using_kernel(torch.matmul)(attention_scores, v)

        # Recombine heads --> [batch_size, seq_length, embed_dim].
        attention_output = self._combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = (
            self.get_attention_output_projection(
                attention_output, special_token_meta=special_token_meta
            )
            * self.output_projection_scale
        )

        return attention_output

    def check_extra_params(params):
        assert (
            k in {"attention_kernel"} for k in params.keys()
        ), "Overflow extra params for attention module `MultiheadAttention`"
