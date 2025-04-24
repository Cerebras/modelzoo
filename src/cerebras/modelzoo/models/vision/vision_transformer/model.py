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

from typing import Literal, Optional

import torch
from torch.nn import CrossEntropyLoss

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.vision.vision_transformer.ViTClassificationModel import (
    ViTClassificationModel,
    ViTClassificationModelConfig,
)
from cerebras.pytorch.metrics import AccuracyMetric


class VisionTransformerModelConfig(ViTClassificationModelConfig):
    name: Literal["vision_transformer"]

    # Transformer
    num_channels: Optional[int] = 3
    "Number of input channels"

    compute_eval_metrics: bool = True


class ViTClassificationWrapperModel(torch.nn.Module):
    def __init__(self, config: VisionTransformerModelConfig):

        super().__init__()

        self.num_classes = config.num_classes

        self.model = self.build_model(config)
        self.loss_fn = CrossEntropyLoss()

        self.compute_eval_metrics = config.compute_eval_metrics
        if self.compute_eval_metrics:
            self.accuracy_metric_cls = AccuracyMetric(name="eval/accuracy_cls")

    def build_model(self, config):
        return ViTClassificationModel(config)

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
            metric_dtype = (
                torch.float32
                if cstorch.amp.is_cbfloat16_tensor(logits)
                else logits.dtype
            )
            self.accuracy_metric_cls(
                labels=eval_labels,
                predictions=eval_preds,
                dtype=metric_dtype,
            )

        return total_loss
