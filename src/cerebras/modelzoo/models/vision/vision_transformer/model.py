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
from torch.nn import CrossEntropyLoss

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.models.nlp.bert.utils import check_unused_model_params
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

        model_params = params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = CrossEntropyLoss()

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.accuracy_metric_cls = AccuracyMetric(name="eval/accuracy_cls")

    def build_model(self, model_params):
        self.num_classes = model_params.pop("num_classes")

        model = ViTClassificationModel(
            num_classes=self.num_classes,
            # Embedding
            embedding_dropout_rate=model_params.pop(
                "embedding_dropout_rate", 0.0
            ),
            hidden_size=model_params.pop("hidden_size"),
            use_post_embed_layer_norm=model_params.pop(
                "use_post_embed_layer_norm", False
            ),
            use_embed_proj_bias=model_params.pop("use_embed_proj_bias", True),
            # Encoder
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            layer_norm_epsilon=float(model_params.pop("layer_norm_epsilon")),
            # Encoder Attn
            num_heads=model_params.pop("num_heads"),
            attention_type=model_params.pop(
                "attention_type", "scaled_dot_product"
            ),
            attention_softmax_fp32=model_params.pop(
                "attention_softmax_fp32", True
            ),
            dropout_rate=model_params.pop("dropout_rate"),
            nonlinearity=model_params.pop("nonlinearity", "gelu"),
            pooler_nonlinearity=model_params.pop("pooler_nonlinearity", "tanh"),
            attention_dropout_rate=model_params.pop(
                "attention_dropout_rate", 0.0
            ),
            use_projection_bias_in_attention=model_params.pop(
                "use_projection_bias_in_attention", True
            ),
            use_ffn_bias_in_attention=model_params.pop(
                "use_ffn_bias_in_attention", True
            ),
            # Encoder ffn
            filter_size=model_params.pop("filter_size"),
            use_ffn_bias=model_params.pop("use_ffn_bias", True),
            # Task-specific
            initializer_range=model_params.pop("initializer_range", 0.02),
            projection_initializer=model_params.pop(
                "projection_initializer", None
            ),
            position_embedding_initializer=model_params.pop(
                "position_embedding_initializer", None
            ),
            position_embedding_type=model_params.pop(
                "position_embedding_type", "learned"
            ),
            attention_initializer=model_params.pop(
                "attention_initializer", None
            ),
            ffn_initializer=model_params.pop("ffn_initializer", None),
            pooler_initializer=model_params.pop("pooler_initializer", None),
            norm_first=model_params.pop("norm_first", True),
            # vision related params
            image_size=model_params.pop("image_size"),
            num_channels=model_params.pop("num_channels"),
            patch_size=model_params.pop("patch_size"),
            use_conv_patchified_embedding=model_params.pop(
                "use_conv_patchified_embedding", False
            ),
            use_encoder_pooler_layer=model_params.pop(
                "use_encoder_pooler_layer", False
            ),
            prepend_cls_token=model_params.pop("prepend_cls_token", True),
            # classifier related
            use_bias_in_output=model_params.pop("use_bias_in_output", True),
        )

        check_unused_model_params(model_params)

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
