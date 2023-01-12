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

import torch.nn as nn

from modelzoo.common.pytorch.layers import (
    EmbeddingLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
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
        position_embedding_type="learned",
        hidden_size=768,
        share_embedding_weights=True,
        # Encoder
        num_hidden_layers=12,
        dropout_rate=0.1,
        layer_norm_epsilon=1.0e-5,
        # Encoder - Attention
        num_heads=12,
        attention_type="scaled_dot_product",
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        attention_dropout_rate=0.1,
        attention_softmax_fp32=True,
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
        # Loss
        loss_weight=1.0,
        fixed_sparse_attention=None,
    ):
        super(GPT2LMHeadModel, self).__init__()

        # std deviation for weight initialization
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.share_embedding_weights = share_embedding_weights
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type

        assert (
            self.position_embedding_type != "rotary"
        ), f"GPT2 models don't support rotary position embedding."

        if initializer is None:
            attention_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": self.initializer_range
                / math.sqrt(2 * self.num_hidden_layers),
            }
            ffn_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": self.initializer_range,
            }
        else:
            attention_initializer = initializer
            ffn_initializer = initializer

        if embedding_initializer is None:
            embedding_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": self.initializer_range,
            }

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            embeddings_initializer=embedding_initializer,
            position_embedding_type=position_embedding_type,
            position_embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
        )

        self.drop = nn.Dropout(dropout_rate)

        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=True,
            add_cross_attention=False,
            attention_type=attention_type,
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

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

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

        self.__reset_parameters()

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.transformer_decoder.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Init final norm layer
        self.ln_f.bias.data.zero_()
        self.ln_f.weight.data.fill_(1.0)

        # Initialize LM head
        self.lm_head.weight.data.normal_(mean=0.0, std=self.initializer_range)
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

    def forward(
        self, input_ids=None, attention_mask=None, labels=None,
    ):
        hidden_states = self.embedding_layer(input_ids)
        hidden_states = self.drop(hidden_states)

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

        hidden_states = self.transformer_decoder(
            hidden_states,
            tgt_mask=causal_attention_mask,
            sparse_mask=sparse_attention_mask,
        )

        lm_logits = self.lm_head(hidden_states)

        return lm_logits
