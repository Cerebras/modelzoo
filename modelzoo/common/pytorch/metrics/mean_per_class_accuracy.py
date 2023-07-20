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

"""
Mean per class Accuracy metric for PyTorch.
Calculates the accuracy for each class, then takes the mean of that.

"""
from typing import Optional

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.metrics.cb_metric import CBMetric, DeviceOutputs
from modelzoo.common.pytorch.metrics.metric_utils import divide_no_nan


def compute_helper(total_per_class_correct_predictions, total_per_class_tokens):
    per_class_accuracy = divide_no_nan(
        total_per_class_correct_predictions, total_per_class_tokens,
    )
    return torch.mean(per_class_accuracy)


class _PipelineMeanPerClassAccuracyMetric(CBMetric):
    """
    Calculates the accuracy for each class, then takes the mean of that.

    Args:
        labels: A `Tensor` of ground truth labels of type `int32` or `int64`.
            The tensor will be flattened if its rank > 1.
        predictions: A `Tensor` of prediction results for semantic labels,
            of type `int32` or `int64`. The tensor will be
            flattened if its rank > 1.
        num_classes: The possible number of labels the prediction task can
            have. This value must be provided, since two variables with
            shape=[num_classes] will be allocated.
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
            `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
            be either `1`, or the same as the corresponding `labels` dimension).
            If `weights` is `None`, weights default to 1.
            Use weights of 0 to mask values.
        name: Optional `string` which indicates name of the metric.
            If None or empty string, it defaults to the name of the class.

    Raises:
        ValueError: If `predictions` and `labels` have mismatched shapes, or if
            `weights` is not `None` and its shape doesn't match `predictions`.
    """

    def __init__(self, num_classes, name: Optional[str] = None):
        self.num_classes = num_classes
        super().__init__(name=name)

    def init_state(self):
        self.reset_state()

    def update_on_host(self, labels, predictions, weights=None):
        if labels.shape != predictions.shape:
            raise ValueError(
                f"`labels` and `predictions` have mismatched shapes of "
                f"{labels.shape} and {predictions.shape} respectively."
            )
        if weights is not None:
            if weights.shape != labels.shape:
                raise ValueError(
                    f"`labels`={labels.shape} and ",
                    f"`weights`={weights.shape} have mismatched shapes",
                )
            weights = weights.detach().flatten()

        labels = labels.detach().to(torch.long)
        predictions = predictions.detach()

        if len(labels.shape) > 1:
            labels = labels.flatten()
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        correct_predictions = labels == predictions
        num_tokens = torch.ones_like(predictions)

        if weights is not None:
            correct_predictions = correct_predictions * weights
            num_tokens = num_tokens * weights

        self.total_per_class_correct_predictions.scatter_add_(
            dim=0, index=labels, src=correct_predictions.to(torch.int32)
        )
        self.total_per_class_tokens.scatter_add_(
            dim=0, index=labels, src=num_tokens.to(torch.int32)
        )

    def compute(self):
        """Returns the computed accuracy as a float."""
        return float(
            compute_helper(
                self.total_per_class_correct_predictions,
                self.total_per_class_tokens,
            )
        )

    def reset_state(self):
        self.total_per_class_correct_predictions = torch.zeros(
            self.num_classes, dtype=torch.int32
        )
        self.total_per_class_tokens = torch.zeros(
            self.num_classes, dtype=torch.int32
        )


class _WSMeanPerClassAccuracyMetric(CBMetric):
    def __init__(self, num_classes, name: Optional[str] = None):
        self.num_classes = num_classes
        super().__init__(name=name)

    def init_state(self):
        self.reset_state()

    def update_on_device(self, labels, predictions, weights=None):
        if labels.shape != predictions.shape:
            raise ValueError(
                f"`labels` and `predictions` have mismatched shapes of "
                f"{labels.shape} and {predictions.shape} respectively."
            )
        if weights is not None:
            if weights.shape != labels.shape:
                raise ValueError(
                    f"`labels`={labels.shape} and ",
                    f"`weights`={weights.shape} have mismatched shapes",
                )
            weights = weights.detach().flatten()

        labels = labels.detach().to(torch.long)
        predictions = predictions.detach()

        if len(labels.shape) > 1:
            labels = labels.flatten()
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        correct_predictions = labels == predictions
        num_tokens = torch.ones_like(predictions)

        if weights is not None:
            correct_predictions = correct_predictions * weights
            num_tokens = num_tokens * weights

        per_class_correct_predictions = torch.zeros(
            self.num_classes, dtype=torch.float32, device=predictions.device,
        )
        per_class_tokens = torch.zeros(
            self.num_classes, dtype=torch.float32, device=predictions.device,
        )

        per_class_correct_predictions.scatter_add_(
            dim=0, index=labels, src=correct_predictions.to(torch.float32)
        )
        per_class_tokens.scatter_add_(
            dim=0, index=labels, src=num_tokens.to(torch.float32)
        )

        self.total_per_class_correct_predictions.add_(
            per_class_correct_predictions
        )
        self.total_per_class_tokens.add_(per_class_tokens)
        mean_per_class_accuracy = compute_helper(
            self.total_per_class_correct_predictions,
            self.total_per_class_tokens,
        )
        # WS Stack limitation: Need to cast to fp16 before store output
        return DeviceOutputs(
            args=[mean_per_class_accuracy.to(predictions.dtype)]
        )

    def update_on_host(self, result):
        self.result = result

    def compute(self):
        """Returns the computed accuracy as a float."""
        return float(self.result)

    def reset_state(self):
        self.total_per_class_correct_predictions = torch.zeros(
            (self.num_classes,), dtype=torch.float32
        ).to(cm.device())
        self.total_per_class_tokens = torch.zeros(
            (self.num_classes,), dtype=torch.float32
        ).to(cm.device())

    def on_device_state_dict(self):
        return {
            "total_per_class_correct_predictions": self.total_per_class_correct_predictions,
            "total_per_class_tokens": self.total_per_class_tokens,
        }


# Create a factory for creating a metric depending on execution strategy.
MeanPerClassAccuracyMetric = CBMetric.create_metric_impl_factory(
    pipeline_metric_cls=_PipelineMeanPerClassAccuracyMetric,
    ws_metric_cls=_WSMeanPerClassAccuracyMetric,
)
