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

from cerebras.modelzoo.models.nlp.bert.bert_pretrain_models import (
    BertClassifierHead,
)
from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import ViTModel


class ViTClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
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
        dropout_rate=0.1,
        nonlinearity="gelu",
        pooler_nonlinearity=None,
        attention_dropout_rate=0.1,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        # Encoder ffn
        filter_size=3072,
        use_ffn_bias=True,
        # Task-specific
        initializer_range=0.02,
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
        use_bias_in_output=True,
    ):
        super(ViTClassificationModel, self).__init__()
        default_initializer = {
            "name": "truncated_normal",
            "std": initializer_range,
            "mean": 0.0,
            "a": initializer_range * -2.0,
            "b": initializer_range * 2.0,
        }

        self.vit_model = ViTModel(
            # Embedding
            embedding_dropout_rate=embedding_dropout_rate,
            hidden_size=hidden_size,
            use_post_embed_layer_norm=use_post_embed_layer_norm,
            use_embed_proj_bias=use_embed_proj_bias,
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
            initializer_range=initializer_range,
            default_initializer=default_initializer,
            projection_initializer=projection_initializer,
            position_embedding_initializer=position_embedding_initializer,
            position_embedding_type=position_embedding_type,
            attention_initializer=attention_initializer,
            ffn_initializer=ffn_initializer,
            pooler_initializer=pooler_initializer,
            norm_first=norm_first,
            # vision related params
            image_size=image_size,
            num_channels=num_channels,
            patch_size=patch_size,
            use_conv_patchified_embedding=use_conv_patchified_embedding,
            prepend_cls_token=prepend_cls_token,
            use_encoder_pooler_layer=use_encoder_pooler_layer,
        )

        self.classifier = BertClassifierHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            use_bias=use_bias_in_output,
            kernel_initializer=default_initializer,
        )

    def reset_parameters(self):
        self.vit_model.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, input_image):
        hidden_states, pooled_states = self.vit_model(input_image)
        logits = self.classifier(pooled_states)
        return logits

    def compute_input_embeddings(self, input_image):
        input_image_embeddings = self.vit_model.embedding_layer(input_image)
        return input_image_embeddings

    def get_input_embeddings(self):
        return self.vit_model.embedding_layer

    def get_output_embeddings(self):
        """
        Extract the final layer that produces logits
        """
        return self.classifier.classifier

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

        return self.vit_model.extract_features(
            input_embeddings, extract_layer_idx
        )
