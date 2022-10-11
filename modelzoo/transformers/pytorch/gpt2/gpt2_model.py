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

from modelzoo.common.pytorch.layers.EmbeddingLayer import EmbeddingLayer
from modelzoo.common.pytorch.layers.TransformerDecoder import TransformerDecoder
from modelzoo.common.pytorch.layers.TransformerDecoderLayer import (
    TransformerDecoderLayer,
)
from modelzoo.common.pytorch.model_utils.GPTLMHeadModelLoss import (
    GPTLMHeadModelLoss,
)
from modelzoo.transformers.pytorch.transformer_utils import (
    create_autoregressive_mask,
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
        # Encoder - ffn
        filter_size=3072,
        nonlinearity="gelu",
        use_ffn_bias=True,
        # Task-specific
        use_bias_in_output=False,
        initializer_range=0.02,
        # Loss
        loss_weight=1.0,
    ):
        super(GPT2LMHeadModel, self).__init__()

        # std deviation for weight initialization
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.share_embedding_weights = share_embedding_weights
        self.max_position_embeddings = max_position_embeddings

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            embeddings_initializer={
                "name": "normal",
                "std": self.initializer_range,
            },
            position_embedding_type=position_embedding_type,
            position_embeddings_initializer={
                "name": "normal",
                "std": self.initializer_range,
            },
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
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer={
                "name": "normal",
                "std": self.initializer_range
                / math.sqrt(2 * self.num_hidden_layers),
            },
            ffn_initializer={"name": "normal", "std": self.initializer_range,},
        )

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_hidden_layers, norm=self.ln_f,
        )

        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=use_bias_in_output
        )
        self.loss_fn = GPTLMHeadModelLoss(vocab_size, loss_weight,)

        self._reset_parameters()

    def _reset_parameters(self):
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

        self.auto_regressive_mask = create_autoregressive_mask(
            max_sequence_length=self.max_position_embeddings,
            device=input_ids.device,
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        causal_attention_mask, _ = torch.broadcast_tensors(
            self.auto_regressive_mask, extended_attention_mask
        )
        causal_attention_mask = causal_attention_mask * -1e4

        hidden_states = self.transformer_decoder(
            hidden_states, tgt_mask=causal_attention_mask,
        )

        lm_logits = self.lm_head(hidden_states)

        loss = self.loss_fn(lm_logits, labels, attention_mask,)

        return loss, lm_logits
