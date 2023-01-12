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
Dice coefficient metric for PyTorch.
"""
from typing import Optional

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch
from modelzoo.common.pytorch.metrics.cb_metric import CBMetric, DeviceOutputs
from modelzoo.common.pytorch.metrics.metric_utils import (
    compute_confusion_matrix,
    divide_no_nan,
)


def DiceCoefficientMetric(*args, **kwargs):
    """
    Dice Coefficient is a common evaluation metric for
    semantic image segmentation.

    Dice Coefficient is defined as follows:
    Dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative).

    The predictions are accumulated in a confusion matrix, weighted by `weights`,
    and dice coefficient is then calculated from it.

    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    Args:
        :param Tensor labels: A `Tensor` of ground truth labels of type `int32` or `int64`.
        :param Tensor predictions: A `Tensor` of prediction results for semantic labels,
            of type `int32` or `int64`.
        :param int num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of
            dimension = [num_classes, num_classes] will be allocated.
        :param Tensor weights: Optional `Tensor` whose rank is either 0, or the same rank
            as `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
            be either `1`, or the same as the corresponding `labels` dimension).
        :param name: Optional `string` which indicates name of the metric.
                If None or empty string, it defaults to the name of the class.

    Returns:
        dice_coefficient: A `Tensor` representing the dice coefficient.

    Raises:
        ValueError: If `predictions` and `labels` have mismatched shapes, or if
            `weights` is not `None` and its shape doesn't match `predictions`
    """

    ws_enabled = False
    if cm.use_cs():
        ws_enabled = cbtorch.env().weight_streaming_mode
    if ws_enabled:
        return WSDiceCoefficientMetric(*args, **kwargs)
    else:
        return PipelineDiceCoefficientMetric(*args, **kwargs)


def compute_helper(confusion_matrix):
    """Returns the dice-coefficient as a float."""
    sum_over_row = torch.sum(confusion_matrix, 0, dtype=torch.float)
    sum_over_col = torch.sum(confusion_matrix, 1, dtype=torch.float)

    # TODO: workaround for SW-76827
    # cm_diag = torch.diagonal(confusion_matrix).to(dtype=torch.float)
    wgth_id = torch.eye(
        confusion_matrix.shape[0], device=confusion_matrix.device
    )
    cm_diag = (wgth_id * confusion_matrix).sum(axis=-1, dtype=torch.float)
    denominator = sum_over_row + sum_over_col

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = torch.sum(torch.ne(denominator, 0).to(torch.float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )

    iou = divide_no_nan(2 * cm_diag, denominator)

    # If the number of valid entries is 0 (no classes) we return 0.
    dice_coefficient = torch.where(
        num_valid_entries > 0,
        torch.sum(iou) / num_valid_entries,
        torch.tensor(0, dtype=torch.float, device=iou.device),
    )
    return dice_coefficient


class PipelineDiceCoefficientMetric(CBMetric):
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
            weights = weights.detach()

        labels = labels.detach()
        predictions = predictions.detach()

        self.confusion_matrix += compute_confusion_matrix(
            labels=labels,
            predictions=predictions,
            weights=weights,
            num_classes=self.num_classes,
            on_device=False,
        )

    def compute(self):
        """Returns the dice-coefficient as a float."""
        return float(compute_helper(self.confusion_matrix))

    def reset_state(self):
        # rows -> groundtruth labels
        # cols -> predicted labels
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.float32
        )


class WSDiceCoefficientMetric(CBMetric):
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
            weights = weights.detach()

        labels = labels.detach()
        predictions = predictions.detach()

        confusion_matrix = compute_confusion_matrix(
            labels=labels,
            predictions=predictions,
            num_classes=self.num_classes,
            weights=weights,
            on_device=True,
        )
        self.confusion_matrix.add_(confusion_matrix)
        dice_coefficient = compute_helper(self.confusion_matrix)
        return DeviceOutputs(args=[dice_coefficient.to(torch.float16)])

    def update_on_host(self, result):
        self.result = result

    def compute(self):
        """Returns the dice-coefficient as a float."""
        return float(self.result)

    def reset_state(self):
        # rows -> groundtruth labels
        # cols -> predicted labels
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.float32
        ).to(cm.device())

    def on_device_state_dict(self):
        return {
            "confusion_matrix": self.confusion_matrix,
        }
