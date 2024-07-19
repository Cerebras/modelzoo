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

from copy import deepcopy
from dataclasses import asdict

import torch
from torch.nn import CrossEntropyLoss

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.models.vision.vision_transformer.ViTClassificationModel import (
    ViTClassificationModel,
)
from cerebras.pytorch.metrics import AccuracyMetric


@registry.register_model(
    "vision_transformer", datasetprocessor=["ImageNet1KProcessor"]
)
class ViTClassificationWrapperModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        model_params = deepcopy(params.model)
        self.model = self.build_model(model_params)
        self.loss_fn = CrossEntropyLoss()

        self.compute_eval_metrics = model_params.compute_eval_metrics
        if self.compute_eval_metrics:
            self.accuracy_metric_cls = AccuracyMetric(name="eval/accuracy_cls")

    def build_model(self, model_params):
        self.num_classes = model_params.num_classes

        model = ViTClassificationModel(
            num_classes=self.num_classes,
            # Embedding
            embedding_dropout_rate=model_params.embedding_dropout_rate,
            hidden_size=model_params.hidden_size,
            use_post_embed_layer_norm=model_params.use_post_embed_layer_norm,
            use_embed_proj_bias=model_params.use_embed_proj_bias,
            # Encoder
            num_hidden_layers=model_params.num_hidden_layers,
            layer_norm_epsilon=float(model_params.layer_norm_epsilon),
            # Encoder Attn
            num_heads=model_params.num_heads,
            attention_type=model_params.attention_type,
            attention_softmax_fp32=model_params.attention_softmax_fp32,
            dropout_rate=model_params.dropout_rate,
            nonlinearity=model_params.nonlinearity,
            pooler_nonlinearity=model_params.pooler_nonlinearity,
            attention_dropout_rate=model_params.attention_dropout_rate,
            use_projection_bias_in_attention=model_params.use_projection_bias_in_attention,
            use_ffn_bias_in_attention=model_params.use_ffn_bias_in_attention,
            # Encoder ffn
            filter_size=model_params.filter_size,
            use_ffn_bias=model_params.use_ffn_bias,
            # Task-specific
            initializer_range=model_params.initializer_range,
            projection_initializer=(
                asdict(model_params.projection_initializer)
                if model_params.projection_initializer
                else None
            ),
            position_embedding_initializer=(
                asdict(model_params.position_embedding_initializer)
                if model_params.position_embedding_initializer
                else None
            ),
            position_embedding_type=model_params.position_embedding_type,
            attention_initializer=(
                asdict(model_params.attention_initializer)
                if model_params.attention_initializer
                else None
            ),
            ffn_initializer=(
                asdict(model_params.ffn_initializer)
                if model_params.ffn_initializer
                else None
            ),
            pooler_initializer=(
                asdict(model_params.pooler_initializer)
                if model_params.pooler_initializer
                else None
            ),
            norm_first=model_params.norm_first,
            use_final_layer_norm=model_params.use_final_layer_norm,
            # vision related params
            image_size=model_params.image_size,
            num_channels=model_params.num_channels,
            patch_size=model_params.patch_size,
            use_conv_patchified_embedding=model_params.use_conv_patchified_embedding,
            use_encoder_pooler_layer=model_params.use_encoder_pooler_layer,
            prepend_cls_token=model_params.prepend_cls_token,
            # classifier related
            use_bias_in_output=model_params.use_bias_in_output,
        )

        return model

    def _post_device_transfer(self):
        pass

    def forward(self, data):
        input_images, labels = data

        logits = self.model(input_images)

        total_loss = self.loss_fn(
            logits.view(-1, self.num_classes), labels.view(-1).long()
        )
        if not self.model.training and self.compute_eval_metrics:
            eval_labels = labels.clone()
            eval_preds = logits.argmax(-1).int()
            # eval/accuracy_cls
            self.accuracy_metric_cls(
                labels=eval_labels,
                predictions=eval_preds,
                dtype=logits.dtype,
            )

        return total_loss
