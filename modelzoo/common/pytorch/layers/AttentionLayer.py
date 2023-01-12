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
import torch.nn as nn

from modelzoo.common.pytorch.model_utils.create_initializer import (
    create_initializer,
)


class MultiheadAttention(nn.Module):
    """Multi-head attention layer. Adapted from:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention

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
        attention_initializer="xavier_uniform",
        attention_q_initializer=None,
        output_layer_initializer=None,
        bias_initializer="zeros",
        attention_type="scaled_dot_product",
        softmax_dtype_fp32=True,
        device=None,
    ):
        _SUPPORTED_ATTENTION_TYPES = ["dot_product", "scaled_dot_product"]
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
        assert kdim is None, "kdim should be set to None."
        assert vdim is None, "vdim should be set to None."
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.inner_dim = inner_dim if inner_dim is not None else embed_dim

        self.num_heads = num_heads
        self.scale_dot_product = attention_type == "scaled_dot_product"
        self.neg_inf = -1e4

        self.use_projection_bias = use_projection_bias
        self.use_ffn_bias = use_ffn_bias

        self.proj_q_dense_layer = nn.Linear(
            self.embed_dim,
            self.inner_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_k_dense_layer = nn.Linear(
            self.embed_dim,
            self.inner_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_v_dense_layer = nn.Linear(
            self.embed_dim,
            self.inner_dim,
            bias=use_projection_bias,
            device=device,
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.proj_output_dense_layer = nn.Linear(
            self.inner_dim, self.embed_dim, bias=use_ffn_bias, device=device,
        )

        # handle initialization
        output_initializer = attention_initializer
        if output_layer_initializer is not None:
            output_initializer = output_layer_initializer

        self.initializer = attention_initializer
        self.query_initalizer = self.initializer
        if attention_q_initializer is not None:
            self.query_initalizer = attention_q_initializer
        self.output_initializer = output_initializer
        self.bias_initializer = bias_initializer
        self.softmax_dtype_fp32 = softmax_dtype_fp32

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
        weight_initializer = create_initializer(self.query_initalizer)
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
                used for self-attention (true) of cross-attention (false). Ignored if
                past_kv is not provided. Default: True
            TODO: the following param doesn't seem to be used anywhere. remove it?
            training (bool): Training the model if ``True``. Needed to call the
                ``dropout`` (after softmax) in the appropriate mode.
            position_bias (Tensor): Tensor containing position bias to apply in attention.

        Returns:
            If ``cache_present_kv`` is ``False``, no entry for present keys and values
            is provided.
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

        # construct query, key and value vector with a linear projection and split into heads
        q = self.construct_query_vector(
            q, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        k = self.construct_key_vector(
            k, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        v = self.construct_value_vector(
            v, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        offset_length, real_seq_length = self.get_sequence_length(
            past_kv, real_seq_length
        )

        # rotary embedding helper
        k = self.apply_rotary_position_embedding(
            k, rotary_position_embedding_helper, real_seq_length, offset_length
        )
        q = self.apply_rotary_position_embedding(
            q, rotary_position_embedding_helper, real_seq_length, offset_length
        )
        # q, k now have shape [batch_size, num_heads, seq_length, head_dim]

        q = self.process_q_before_logits_calc(q)
        k = self.process_k_before_logits_calc(k)
        v = self.process_v_before_logits_calc(v)

        k, v = self.process_past_kv(past_kv, past_kv_self_attn, k, v)

        present_kv = self.construct_present_kv(cache_present_kv, k, v)

        logits = self.calculate_attention_logits(q, k)

        attn_mask_processed = self.process_attention_mask(attn_mask, past_kv, q)
        key_padding_mask_processed = self.process_key_padding_mask(
            key_padding_mask, attn_mask, past_kv, q
        )

        attention_bias = self.combine_masks(
            attn_mask_processed, key_padding_mask_processed
        )

        logits = self.apply_attention_bias(logits, attention_bias)
        logits = self.apply_position_bias(logits, position_bias)

        attention_scores = self.calculate_attention_scores(logits)
        attention_output = self.calculate_attention_output(attention_scores, v)

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

    def construct_query_vector(self, q, attn_mask=None, key_padding_mask=None):
        # linear projection
        q = self.proj_q_dense_layer(q)
        # split into heads
        q = self._split_heads(q, rotary=True)
        return q

    def construct_key_vector(self, k, attn_mask=None, key_padding_mask=None):
        # linear projection
        k = self.proj_k_dense_layer(k)
        # split into heads
        k = self._split_heads(k, rotary=True)
        return k

    def construct_value_vector(self, v, attn_mask=None, key_padding_mask=None):
        # linear projection
        v = self.proj_v_dense_layer(v)
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
        real_seq_length,
        offset_length,
    ):
        if rotary_position_embedding_helper:
            vector = rotary_position_embedding_helper.rotate_tensor(
                vector, real_seq_length, offset=offset_length
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

    def calculate_attention_logits(self, q, k):
        if self.scale_dot_product:
            depth = self.embed_dim // self.num_heads
            q = q * torch.tensor(1 / float(depth) ** 0.5, dtype=q.dtype,)

        # calculate dot product attention
        logits = torch.matmul(
            q, k.transpose(-1, -2)
        )  # (B, H, Lq, E) * (B, H, E, Lk) -> (B, H, Lq, Lk)
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
                        (q.shape[0], past_kv.shape[-2]), dtype=attn_mask.dtype,
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

        # apply key padding mask
        if key_padding_mask is not None:
            # 2D [batch_size, sequence_length]
            assert (
                len(key_padding_mask.shape) == 2
            ), "Only 2D key_padding_mask is supported for now"
            if (
                not key_padding_mask.is_floating_point()
                and not key_padding_mask.dtype == torch.bool
            ):
                key_padding_mask = key_padding_mask.to(torch.bool)

            # for broadcasting over all heads and queries
            if past_kv is not None:
                past_mask = torch.zeros(
                    (q.shape[0], past_kv.shape[-2]), dtype=attn_mask.dtype,
                )
                key_padding_mask = torch.cat(
                    [past_mask, key_padding_mask], axis=-1
                )
            batch_size, all_seq_length = key_padding_mask.shape

            # compute the attention_bias based on the mask.
            key_padding_mask_reshaped = key_padding_mask.view(
                batch_size, 1, 1, all_seq_length
            )
            # Need to expand key_padding_mask's head dimension if we need to merge masks
            if attn_mask is not None:
                key_padding_mask_reshaped = key_padding_mask_reshaped.expand(
                    batch_size, self.num_heads, 1, all_seq_length
                )

        return key_padding_mask_reshaped

    def combine_masks(self, attn_mask_reshaped, key_padding_mask_reshaped):
        attention_bias = None

        if (
            attn_mask_reshaped is not None
            and key_padding_mask_reshaped is not None
        ):
            # Need to merge attention mask and key padding mask:
            attn_mask_is_float = attn_mask_reshaped.is_floating_point()
            key_padding_is_float = key_padding_mask_reshaped.is_floating_point()

            if attn_mask_is_float and key_padding_is_float:
                attention_bias = torch.minimum(
                    attn_mask_reshaped, key_padding_mask_reshaped
                )
            elif attn_mask_is_float:
                attention_bias = attn_mask_reshaped.masked_fill(
                    key_padding_mask_reshaped, self.neg_inf
                )
            elif key_padding_is_float:
                attention_bias = key_padding_mask_reshaped.masked_fill(
                    attn_mask_reshaped, self.neg_inf
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
                final_attention_bias.masked_fill_(attention_bias, self.neg_inf)
                attention_bias = final_attention_bias
            logits += attention_bias
        return logits

    def apply_position_bias(self, logits, position_bias):
        # Add relative position bias, if any
        if position_bias is not None:
            logits += position_bias.unsqueeze(0)
        return logits

    def calculate_attention_scores(self, logits):
        if self.softmax_dtype_fp32 and logits.dtype != torch.float32:
            attention_scores = nn.functional.softmax(
                logits.float(), dim=-1
            ).type_as(logits)
        else:
            attention_scores = nn.functional.softmax(logits, dim=-1)
        attention_scores = self.dropout_layer(attention_scores)
        return attention_scores

    def calculate_attention_output(self, attention_scores, v):
        # Shape: (batch_size, num_heads, query_length, embed_dim / num_heads)
        attention_output = torch.matmul(attention_scores, v)

        # Recombine heads --> [batch_size, seq_length, embed_dim].
        attention_output = self._combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.proj_output_dense_layer(attention_output)

        return attention_output

    def check_extra_params(params):
        assert (
            params == {}
        ), "Overflow extra params for attention module `MultiheadAttention`, should be empty mapping"
