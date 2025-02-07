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

# This code is adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/deberta/modeling_deberta.py
# and
# https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/disentangled_attention.py
#
# Copyright 2023 Cerebras Systems.
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
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

from cerebras.modelzoo.layers.create_initializer import create_initializer

from .AttentionLayer import MultiheadAttention


class DisentangledAttention(MultiheadAttention):
    """
        Applies the disenntangled attention mechanism according to the deberta (https://arxiv.org/pdf/2006.03654.pdf) paper.
        Each word is represented using two vectors that encode its content and position, respectively, and the attention
        weights among words are computed using disentangled matrices on their contents and relative positions, respectively.
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

        # Disentangled Attention Keyword Args
            pos_att_type (set of str): Types of disentangled attentions to compute.
                Supports the following options:
                    * content to position attention ("c2p")
                    * position to content attention ("p2c")
                    * position to position attention ("p2p")
                Default: {"c2p", "p2c"}
            share_att_key (bool): If enabled (true), the content projection matrix
                is shared with the position projection matrix. This feature was
                introduced in DeBerta v2. Default: False.
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
        # Disentangled Attention Keyword Args
        pos_att_type={"c2p", "p2c"},
        share_att_key=False,
    ):

        super(DisentangledAttention, self).__init__(
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

        self.share_att_key = share_att_key
        self.pos_att_type = pos_att_type
        self.pos_dropout = nn.Dropout(dropout)

        if not self.share_att_key:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_key_proj = nn.Linear(
                    self.embed_dim, self.inner_dim, bias=True
                )
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_query_proj = nn.Linear(self.embed_dim, self.inner_dim)

        self.__reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        if not self.share_att_key:
            weight_initializer = create_initializer(self.initializer)
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                weight_initializer(self.pos_key_proj.weight.data)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                weight_initializer(self.pos_query_proj.weight.data)

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
        relative_position_embeddings=None,
        relative_positions=None,
        relative_position_embed_size=None,
        relative_pos_key_only=None,
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
                used for self-attention (true) of cross-attention (false). Ignored if
                past_kv is not provided. Default: True
            position_bias (Tensor): Tensor containing position bias to apply in attention.
            rotary_position_embedding_helper: Helper module for applying rotary embeddings
            relative_position_embeddings (tensor): Required tensor containing relative position
                embeddings. Shape ``[2*relative_position_embed_size, embed_dim]``
            relative_positions (tensor): Required tensor contaning relative positions.
                Shape ``[1, 1, query_seq_length, key_seq_length]``
            relative_position_embed_size (int): Size of relative position embeddings
                which corresponds to either the number of max relative positions or
                the number of relative position buckets (depending on if bucketing
                is enabled).
            relative_pos_key_only (tensor): Required only if
                `query_seq_length != key_seq_length`. Contains the relative
                positions between keys only. Shape ``[1, 1, key_seq_length, key_seq_length]``


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

        assert (
            relative_position_embeddings is not None
        ), "DisentangledAttentionLayer requires relative_position_embeddings to be specified"
        assert (
            relative_positions is not None
        ), "DisentangledAttentionLayer requires relative_positions to be specified"
        assert (
            relative_position_embed_size is not None
        ), "DisentangledAttentionLayer requires relative_position_embed_size to be specified"

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
            k, rotary_position_embedding_helper, offset_length
        )
        q = self.apply_rotary_position_embedding(
            q, rotary_position_embedding_helper, offset_length
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
        disentangled_attention_bias = self.disentangled_attention_bias(
            q,
            k,
            relative_positions,
            relative_position_embeddings,
            relative_position_embed_size,
            relative_pos_key_only,
        )
        logits += disentangled_attention_bias

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

    def calculate_attention_logits(self, q, k):
        if self.attention_type == "scaled_dot_product":
            depth = self.embed_dim // self.num_heads
            scale_factor = 1 + len(
                self.pos_att_type
            )  # scale_factor = 1 for vanilla attention
            q = q * torch.tensor(
                1 / float(depth * scale_factor) ** 0.5,
                dtype=q.dtype,
            )

        # calculate dot product attention
        logits = self.using_kernel(torch.matmul)(
            q, k.transpose(-1, -2)
        )  # (B, H, Lq, E) * (B, H, E, Lk) -> (B, H, Lq, Lk)
        return logits

    def disentangled_attention_bias(
        self,
        q,
        k,
        relative_pos,
        rel_embeddings,
        relative_position_embed_size,
        relative_pos_key_only=None,
    ):
        assert relative_pos.dim() == 4, "Relative postion ids must be of dim 4"
        # relative_pos shape: (1, 1, Lq, Lk)
        # rel_embeddings shape: (2*relative_position_embed_size, E)

        batch_size, num_heads, query_seq_length, depth = q.shape
        key_seq_length = k.shape[2]

        if key_seq_length != query_seq_length:
            assert (
                relative_pos_key_only is not None
            ), "DisentangledAttentionLayer requires relative_pos_key_only to be specified when the query & key sequence lengths aren't the same."

        att_span = min(
            max(query_seq_length, key_seq_length), relative_position_embed_size
        )  # as implemented in HF (different from original MSFT implementation)
        relative_pos = relative_pos.long().to(q.device)
        if relative_pos_key_only is not None:
            relative_pos_key_only = relative_pos_key_only.long().to(q.device)

        rel_embeddings = rel_embeddings[
            relative_position_embed_size
            - att_span : relative_position_embed_size
            + att_span,
            :,
        ].unsqueeze(0)
        rel_embeddings = self.pos_dropout(rel_embeddings)
        rel_embeddings_length = rel_embeddings.shape[1]  # 2K
        # rel_embeddings shape: (1, 2K, E)
        if self.share_att_key:
            pos_query_layer = self.proj_q_dense_layer(
                rel_embeddings
            )  # (1, 2K, E')
            pos_query_layer = self._split_heads(
                pos_query_layer, rotary=False
            )  # (1, H, 2K, D)

            pos_key_layer = self.proj_k_dense_layer(rel_embeddings)
            pos_key_layer = self._split_heads(pos_key_layer, rotary=False)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.pos_key_proj(rel_embeddings)
                pos_key_layer = self._split_heads(pos_key_layer, rotary=False)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.pos_query_proj(rel_embeddings)
                pos_query_layer = self._split_heads(
                    pos_query_layer, rotary=False
                )

        if self.attention_type == "scaled_dot_product":
            scale_factor = 1 + len(
                self.pos_att_type
            )  # scale_factor = 1 for vanilla attention
            scale = torch.tensor(
                1 / float(depth * scale_factor) ** 0.5,
                dtype=q.dtype,
            )

        score = 0

        # content->position
        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)

        if 'c2p' in self.pos_att_type:
            q_normalized = (
                q * scale if self.attention_type == "scaled_dot_product" else q
            )
            c2p_att = torch.matmul(
                q_normalized, pos_key_layer.transpose(-1, -2)
            )
            c2p_index = c2p_pos.squeeze(0).expand(
                [batch_size, num_heads, query_seq_length, key_seq_length]
            )
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_index)
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or (
            'p2p' in self.pos_att_type and query_seq_length != key_seq_length
        ):
            r_pos = (
                relative_pos
                if key_seq_length == query_seq_length
                else relative_pos_key_only
            )
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_seq_length != key_seq_length:
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                pos_index = (
                    pos_index + key_seq_length
                ) % key_seq_length  # DEVIATION: this fixes negative index bug

        if 'p2c' in self.pos_att_type:
            k_normalized = (
                k * scale if self.attention_type == "scaled_dot_product" else k
            )
            p2c_att = torch.matmul(
                k_normalized, pos_query_layer.transpose(-1, -2)
            )
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).expand(
                    [batch_size, num_heads, key_seq_length, key_seq_length]
                ),
            ).transpose(-1, -2)
            if query_seq_length != key_seq_length:
                p2c_att = torch.gather(
                    p2c_att,
                    dim=-2,
                    index=pos_index.expand(
                        (
                            batch_size,
                            num_heads,
                            query_seq_length,
                            key_seq_length,
                        )
                    ),
                )
            score += p2c_att

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:, :, att_span:, :]
            pos_query_normalized = (
                pos_query * scale
                if self.attention_type == "scaled_dot_product"
                else pos_query
            )  # DEVIATION: original implementations forget to scale p2p score
            p2p_att = torch.matmul(
                pos_query_normalized, pos_key_layer.transpose(-1, -2)
            )
            p2p_att = p2p_att.expand(
                (batch_size, num_heads, att_span, rel_embeddings_length)
            )
            if query_seq_length != key_seq_length:
                p2p_att = torch.gather(
                    p2p_att,
                    dim=-2,
                    index=pos_index.expand(
                        (batch_size, num_heads)
                        + (pos_index.size(-2), p2p_att.size(-1))
                    ),
                )
            p2p_att = torch.gather(
                p2p_att,
                dim=-1,
                index=c2p_pos.expand(
                    [
                        batch_size,
                        num_heads,
                        query_seq_length,
                        relative_pos.size(-1),
                    ]
                ),
            )
            score += p2p_att

        return score

    def check_extra_params(extra_params):
        supported_extra_params = {
            "pos_att_type",
            "share_att_key",
            "attention_kernel",
        }
        for param_name in extra_params:
            assert (
                param_name in supported_extra_params
            ), "{} is not a supported extra parameter for disentangled attention".format(
                param_name
            )
