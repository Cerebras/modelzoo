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

from modelzoo.common.pytorch.metrics import (
    AccuracyMetric,
    DiceCoefficientMetric,
    MeanIOUMetric,
)
from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.vision.pytorch.unet.modeling_unet import UNet


class UNetModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        model_params = params["model"].copy()
        self.loss_type = model_params["loss"]
        self.model = self.build_model(model_params)

        self.compute_eval_metrics = model_params.pop("compute_eval_metrics")
        self.eval_ignore_classes = model_params.pop("eval_ignore_classes")
        self.eval_metrics = model_params.pop("eval_metrics")
        self.eval_metrics_objs = {}
        if self.compute_eval_metrics:
            if "Acc" in self.eval_metrics:
                self.eval_metrics_objs["Acc"] = AccuracyMetric(
                    name="eval/accuracy"
                )
            if "mIOU" in self.eval_metrics:
                self.eval_metrics_objs["mIOU"] = MeanIOUMetric(
                    name="eval/mean_iou",
                    num_classes=self.model.num_classes,
                    ignore_classes=self.eval_ignore_classes,
                )
            if "DSC" in self.eval_metrics:
                self.eval_metrics_objs["DSC"] = DiceCoefficientMetric(
                    name="eval/dice_coefficient",
                    num_classes=self.model.num_classes,
                    ignore_classes=self.eval_ignore_classes,
                )

    def build_model(self, model_params):
        model = UNet(model_params)
        self.loss_fn = model.loss_fn
        return model

    def forward(self, data):
        inputs, labels = data
        outputs = self.model(inputs)

        if "ssce" in self.loss_type:
            loss = self.loss_fn(outputs, labels.view(labels.shape).long())
        else:
            loss = self.loss_fn(outputs, labels)

        if not self.model.training and self.compute_eval_metrics:
            eval_labels = labels.clone()
            if self.model.num_output_channels > 1:
                predictions = outputs.argmax(dim=1).to(
                    half_dtype_instance.half_dtype
                )
            else:
                predictions = torch.where(
                    outputs
                    > torch.tensor(
                        0.5, dtype=outputs.dtype, device=outputs.device
                    ),
                    torch.tensor(
                        1,
                        dtype=half_dtype_instance.half_dtype,
                        device=outputs.device,
                    ),
                    torch.tensor(
                        0,
                        dtype=half_dtype_instance.half_dtype,
                        device=outputs.device,
                    ),
                )

            if self.model.loss_type == "multilabel_bce":
                # since labels are one-hot tensors
                # of shape (bsz, num_classes, H, W)
                eval_labels = torch.argmax(
                    eval_labels.to(half_dtype_instance.half_dtype), dim=1
                ).to(torch.int16)

            for _metric_name, metric_obj in self.eval_metrics_objs.items():
                metric_obj(labels=eval_labels, predictions=predictions)

        return loss
