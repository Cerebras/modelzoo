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

from modelzoo.common.pytorch.layers import (
    EmbeddingLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from modelzoo.common.pytorch.layers.utils import apply_position_bias
from modelzoo.common.pytorch.model_utils.norms import get_norm
from modelzoo.common.pytorch.model_utils.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)
from modelzoo.transformers.pytorch.gpt2.sparse_mask import (
    create_fixed_sparse_attention_mask,
)
from modelzoo.transformers.pytorch.transformer_utils import (
    build_broadcastable_attention_mask,
    make_sparse_mask_broadcastable,
)


class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 model with LM head
    """

    def __init__(
        self,
        # Embedding
        vocab_size=50257,
        max_position_embeddings=1024,
        embd_pdrop=0.1,
        position_embedding_type="learned",
        position_embedding_offset=0,
        hidden_size=768,
        share_embedding_weights=True,
        embedding_layer_norm=False,
        num_relative_attention_buckets=32,
        rotary_dim=None,
        # Encoder
        num_hidden_layers=12,
        dropout_rate=0.1,
        norm_type="layernorm",
        layer_norm_epsilon=1.0e-5,
        # Encoder - Attention
        num_heads=12,
        attention_type="scaled_dot_product",
        attention_module="aiayn_attention",
        extra_attention_params={},
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        attention_dropout_rate=0.1,
        attention_softmax_fp32=True,
        attention_kernel=None,
        fixed_sparse_attention=None,
        # Encoder - ffn
        filter_size=3072,
        nonlinearity="gelu",
        use_ffn_bias=True,
        # Task-specific
        use_bias_in_output=False,
        initializer_range=0.02,
        embedding_initializer=None,
        initializer=None,
        output_layer_initializer=None,
        # muP (maximal update parameterization)  parameters
        output_logits_scale=None,
        embeddings_scale=1.0,
        scale_qk_dot_by_d=False,
        alibi_trainable_slopes=False,
        scale_qk_dot_by_layer_idx=False,
    ):
        super(GPT2LMHeadModel, self).__init__()

        # std deviation for weight initialization
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.share_embedding_weights = share_embedding_weights
        self.embedding_layer_norm = embedding_layer_norm
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.embeddings_scale = embeddings_scale

        if initializer is None:
            attention_initializer = {
                "name": "truncated_normal",
                "mean": 0.0,
                "std": self.initializer_range,
            }
            ffn_initializer = {
                "name": "truncated_normal",
                "mean": 0.0,
                "std": self.initializer_range,
            }
            if output_layer_initializer is None:
                output_layer_initializer = {
                    "name": "truncated_normal",
                    "mean": 0.0,
                    "std": self.initializer_range
                    / math.sqrt(2 * self.num_hidden_layers),
                }
        else:
            attention_initializer = initializer
            ffn_initializer = initializer

        if embedding_initializer is None:
            embedding_initializer = {
                "name": "truncated_normal",
                "mean": 0.0,
                "std": self.initializer_range,
            }

        norm_class = get_norm(norm_type)

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            embeddings_initializer=embedding_initializer,
            position_embedding_type=position_embedding_type,
            position_embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
            position_embedding_offset=position_embedding_offset,
        )

        if self.embedding_layer_norm:
            self.embedding_ln_f = norm_class(
                hidden_size, eps=layer_norm_epsilon
            )

        if position_embedding_type == "rotary":
            if rotary_dim is None:
                rotary_dim = hidden_size // num_heads
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3
            assert (
                rotary_dim <= hidden_size / num_heads
            ), "Rotary dimensions should be <= hidden size divided by number of attention heads."
            assert (
                rotary_dim % 2 == 0
            ), "Rotary dimension must be an even number."
        embedding_helper = self.embedding_layer.position_embedding_helper(
            num_heads=num_heads,
            num_relative_attention_buckets=num_relative_attention_buckets,
            rotary_dim=rotary_dim,
            alibi_trainable_slopes=alibi_trainable_slopes,
        )

        self.rotary_pe_helper = None
        self.relative_pe_helper = None
        if isinstance(embedding_helper, RotaryPositionEmbeddingHelper):
            self.rotary_pe_helper = embedding_helper
        else:
            self.relative_pe_helper = embedding_helper

        self.drop_embd = nn.Dropout(embd_pdrop)

        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_layer=norm_class,
            norm_first=True,
            extra_attention_params={
                "attention_kernel": attention_kernel,
                **extra_attention_params,
            },
            add_cross_attention=False,
            attention_type=attention_type,
            scale_qk_dot_by_d=scale_qk_dot_by_d,
            scale_qk_dot_by_layer_idx=scale_qk_dot_by_layer_idx,
            attention_module=attention_module,
            attention_dropout_rate=attention_dropout_rate,
            attention_softmax_fp32=attention_softmax_fp32,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            attention_output_layer_initializer=output_layer_initializer,
            ffn_initializer=ffn_initializer,
            ffn_output_layer_initializer=output_layer_initializer,
            use_ff_layer1_dropout=False,
        )
        self.output_logits_scale = output_logits_scale

        # Final LayerNorm
        self.ln_f = norm_class(hidden_size, eps=layer_norm_epsilon)

        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_hidden_layers, norm=self.ln_f,
        )

        if fixed_sparse_attention is not None:
            self.fixed_sparsity_mask = create_fixed_sparse_attention_mask(
                max_sequence_length=max_position_embeddings,
                n_heads=num_heads,
                **fixed_sparse_attention,
            )
        else:
            self.fixed_sparsity_mask = None

        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=use_bias_in_output
        )

        self.tie_weights()

        self.__reset_parameters()

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.transformer_decoder.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Init final norm layer
        if hasattr(self.ln_f, "bias"):
            self.ln_f.bias.data.zero_()
        self.ln_f.weight.data.fill_(1.0)

        # Initialize LM head
        if not self.share_embedding_weights:
            self.lm_head.weight.data.normal_(
                mean=0.0, std=self.initializer_range
            )
        if self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()

    def tie_weights(self):
        if not self.share_embedding_weights:
            return

        output_embedding = self.get_output_embeddings()
        input_embedding = self.get_input_embeddings()
        output_embedding.weight = input_embedding.weight

        if getattr(output_embedding, "bias", None) is not None:
            output_embedding.bias.data = nn.functional.pad(
                output_embedding.bias.data,
                (
                    0,
                    output_embedding.weight.shape[0]
                    - output_embedding.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embedding, "out_features") and hasattr(
            input_embedding, "num_embeddings"
        ):
            output_embedding.out_features = input_embedding.num_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embedding_layer.get_input_embeddings()

    def compute_input_embeddings(self, input_ids):
        hidden_states = self.embedding_layer(input_ids)
        if self.embedding_layer_norm:
            hidden_states = self.embedding_ln_f(hidden_states)
        hidden_states *= torch.tensor(
            float(self.embeddings_scale), dtype=hidden_states.dtype
        )
        hidden_states = self.drop_embd(hidden_states)
        return hidden_states

    def forward(
        self, input_ids=None, attention_mask=None,
    ):
        hidden_states = self.compute_input_embeddings(input_ids)

        causal_attention_mask = build_broadcastable_attention_mask(
            attention_mask,
            build_causal=True,
            device=input_ids.device,
            dtype=hidden_states.dtype,
        )

        # Fixed sparse attention, used in GPT-3 model
        sparse_attention_mask = None
        if self.fixed_sparsity_mask is not None:
            sparse_attention_mask = make_sparse_mask_broadcastable(
                self.fixed_sparsity_mask,
                attention_mask,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                revert_mask=False,
            )

        # Helpers on alibi/relative position embeddings bias
        length = input_ids.shape[1]
        self_attn_position_bias = apply_position_bias(
            self.relative_pe_helper, length, length
        )

        hidden_states = self.transformer_decoder(
            hidden_states,
            tgt_mask=causal_attention_mask,
            sparse_mask=sparse_attention_mask,
            rotary_position_embedding_helper=self.rotary_pe_helper,
            self_attn_position_bias=self_attn_position_bias,
        )

        lm_logits = self.lm_head(hidden_states)

        # scale lm_logits for muP transfer
        if self.output_logits_scale:
            lm_logits *= torch.tensor(
                float(self.output_logits_scale), dtype=lm_logits.dtype,
            )

        return lm_logits
