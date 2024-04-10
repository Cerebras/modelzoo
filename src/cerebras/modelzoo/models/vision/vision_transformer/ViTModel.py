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

from torch import nn

from cerebras.modelzoo.layers import (
    TransformerEncoder,
    TransformerEncoderLayer,
    ViTEmbeddingLayer,
)
from cerebras.modelzoo.models.nlp.bert.bert_model import BertPooler


class ViTEncoder(nn.Module):
    def __init__(
        self,
        # Embedding
        hidden_size=768,
        # Encoder
        num_hidden_layers=12,
        layer_norm_epsilon=1.0e-5,
        # Encoder Attn
        num_heads=12,
        attention_module="aiayn_attention",
        extra_attention_params={},
        attention_type="scaled_dot_product",
        attention_softmax_fp32=True,
        dropout_rate=0.0,
        nonlinearity="gelu",
        pooler_nonlinearity=None,
        attention_dropout_rate=0.0,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        # Encoder ffn
        filter_size=3072,
        use_ffn_bias=True,
        # Task-specific
        use_final_layer_norm=True,
        initializer_range=0.02,
        default_initializer=None,
        attention_initializer=None,
        ffn_initializer=None,
        pooler_initializer=None,
        norm_first=True,
        use_encoder_pooler_layer=False,
    ):
        super(ViTEncoder, self).__init__()

        self.initializer_range = initializer_range

        if default_initializer is None:
            default_initializer = {
                "name": "truncated_normal",
                "std": self.initializer_range,
                "mean": 0.0,
                "a": self.initializer_range * -2.0,
                "b": self.initializer_range * 2.0,
            }

        if attention_initializer is None:
            attention_initializer = default_initializer
        if ffn_initializer is None:
            ffn_initializer = default_initializer
        if pooler_initializer is None:
            pooler_initializer = default_initializer

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=norm_first,
            attention_module=attention_module,
            attention_dropout_rate=attention_dropout_rate,
            attention_type=attention_type,
            attention_softmax_fp32=attention_softmax_fp32,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            ffn_initializer=ffn_initializer,
            use_ff_layer1_dropout=False,
            use_ff_layer2_dropout=True,
        )

        final_ln_f = None
        if use_final_layer_norm:
            final_ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_hidden_layers, norm=final_ln_f
        )

        if pooler_nonlinearity is None:
            pooler_nonlinearity = nonlinearity

        self.pooler = (
            BertPooler(
                hidden_size,
                pooler_norm=False,
                layer_norm_epsilon=layer_norm_epsilon,
                use_bias=use_ffn_bias,
                activation=pooler_nonlinearity,
                dropout=None,
                initializer=pooler_initializer,
            )
            if use_encoder_pooler_layer
            else None
        )

    def reset_parameters(self):
        self.transformer_encoder.reset_parameters()
        if self.pooler is not None:
            self.pooler.reset_parameters()

    def forward(self, input_embeddings, extract_layer_idx=None):
        # no mask required for now
        hidden_states = self.transformer_encoder(
            input_embeddings, extract_layer_idx=extract_layer_idx
        )

        pooled_states = None
        if self.pooler is not None:
            pooled_states = self.pooler(hidden_states)
        else:
            pooled_states = hidden_states[:, 0]

        return hidden_states, pooled_states


class ViTModel(nn.Module):
    def __init__(
        self,
        # Embedding
        embedding_dropout_rate=0.0,
        hidden_size=768,
        use_post_embed_layer_norm=False,
        use_embed_proj_bias=True,
        # Encoder
        num_hidden_layers=12,
        layer_norm_epsilon=1.0e-5,
        # Encoder Attn
        num_heads=12,
        attention_module="aiayn_attention",
        extra_attention_params={},
        attention_type="scaled_dot_product",
        attention_softmax_fp32=True,
        dropout_rate=0.0,
        nonlinearity="gelu",
        pooler_nonlinearity=None,
        attention_dropout_rate=0.0,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        # Encoder ffn
        filter_size=3072,
        use_ffn_bias=True,
        # Task-specific
        use_final_layer_norm=True,
        initializer_range=0.02,
        default_initializer=None,
        projection_initializer=None,
        position_embedding_initializer=None,
        position_embedding_type="learned",
        attention_initializer=None,
        ffn_initializer=None,
        pooler_initializer=None,
        norm_first=True,
        # vision related params
        image_size=[224, 224],
        num_channels=3,
        patch_size=[16, 16],
        use_conv_patchified_embedding=False,
        use_encoder_pooler_layer=False,
        prepend_cls_token=True,
    ):
        super(ViTModel, self).__init__()

        self.embedding_layer = ViTEmbeddingLayer(
            image_size=image_size,
            num_channels=num_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            initializer_range=initializer_range,
            embedding_dropout_rate=embedding_dropout_rate,
            projection_initializer=projection_initializer,
            position_embedding_type=position_embedding_type,
            position_embedding_initializer=position_embedding_initializer,
            use_conv_patchified_embedding=use_conv_patchified_embedding,
            prepend_cls_token=prepend_cls_token,
            use_post_embed_layer_norm=use_post_embed_layer_norm,
            use_embed_proj_bias=use_embed_proj_bias,
        )

        self.encoder = ViTEncoder(
            # Embedding
            hidden_size=hidden_size,
            # Encoder
            num_hidden_layers=num_hidden_layers,
            layer_norm_epsilon=layer_norm_epsilon,
            # Encoder Attn
            num_heads=num_heads,
            attention_module=attention_module,
            extra_attention_params=extra_attention_params,
            attention_type=attention_type,
            attention_softmax_fp32=attention_softmax_fp32,
            dropout_rate=dropout_rate,
            nonlinearity=nonlinearity,
            pooler_nonlinearity=pooler_nonlinearity,
            attention_dropout_rate=attention_dropout_rate,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            # Encoder ffn
            filter_size=filter_size,
            use_ffn_bias=use_ffn_bias,
            # Task-specific
            use_final_layer_norm=use_final_layer_norm,
            initializer_range=initializer_range,
            default_initializer=default_initializer,
            attention_initializer=attention_initializer,
            ffn_initializer=ffn_initializer,
            pooler_initializer=pooler_initializer,
            norm_first=norm_first,
            use_encoder_pooler_layer=use_encoder_pooler_layer,
        )

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.encoder.reset_parameters()

    def forward(self, input_image=None, input_image_embeddings=None):
        if input_image is not None and input_image_embeddings is not None:
            raise ValueError(
                f"Only one of `input_image` or `input_image_embeddings` should be passed to model.forward"
            )

        if input_image_embeddings is None:
            input_image_embeddings = self.embedding_layer(input_image)

        hidden_states, pooled_states = self.encoder(input_image_embeddings)

        return hidden_states, pooled_states

    def compute_input_embeddings(self, input_image):
        input_image_embeddings = self.embedding_layer(input_image)
        return input_image_embeddings

    def tie_weights(self):
        # weights not tied
        pass

    def extract_features(self, input_embeddings, extract_layer_idx):
        """
        Extract features from `extract_layer_idx` of encoder
        by passing input_tensor through encoder
        input_embeddings: Tensor with output from embeddings layer
        extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
                Applies encoder layers up to (and including) `extract_layer_idx`
                instead of all encoder layers.
                For ex: extract_layer_idx=3 would run fwd pass from encoder_block_0 to encoder_block_3
                and return outputs from encoder_block_3.
                If `extract_layer_idx` = None and `norm` != None, then
                the output returned would be encoder_block_{self.num_layers-1} -> norm -> output (return)
        """
        hidden_states, _ = self.encoder(
            input_embeddings, extract_layer_idx=extract_layer_idx
        )
        return hidden_states
