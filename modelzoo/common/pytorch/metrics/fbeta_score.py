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
F Beta Score metric for PyTorch.
Confusion matrix calculation in Pytorch referenced from: 
https://github.com/pytorch/ignite/blob/master/ignite/metrics/confusion_matrix.py

"""

from typing import List, Optional

import torch

from modelzoo.common.pytorch.metrics.cb_metric import CBMetric
from modelzoo.common.pytorch.metrics.metric_utils import divide_no_nan


class FBetaScoreMetric(CBMetric):
    """Calculates F Score from labels and predictions.
    
        fbeta = (1 + beta^2) * (precision*recall) / ((beta^2 * precision) + recall)

    Where beta is some positive real factor. 
    Args:
        num_classes: Number of classes. 
        beta: Beta coefficient in the F measure.
        average_type: Defines the reduction that is applied. Should be one 
            of the following:
            - 'micro' [default]: Calculate the metric globally, across all 
                samples and classes.
            - 'macro': Calculate the metric for each class separately, and 
                average the metrics across classes (with equal weights for 
                each class). This does not take label imbalance into account.
        ignore_labels: Integer specifying a target classes to ignore.
        name: Name of the metric 

    Raises:
        ValueError: If average_type is none of "micro", "macro".
    """

    def __init__(
        self,
        num_classes,
        beta: float = 1.0,
        average_type: str = "micro",
        ignore_labels: Optional[List] = None,
        name: Optional[str] = None,
    ):
        if num_classes <= 1:
            raise ValueError(
                f"'num_classes' should be at least 2, got {num_classes}"
            )
        self.num_classes = num_classes
        if beta <= 0:
            raise ValueError(f"'beta' should be a positive number, got {beta}")
        self.beta = beta
        self.ignore_labels = ignore_labels

        allowed_average = ["micro", "macro"]
        if average_type not in allowed_average:
            raise ValueError(
                f"The average_type has to be one of {allowed_average}, "
                f"got {average_type}."
            )
        self.average_type = average_type

        super().__init__(name=name)

    def init_state(self):
        self.reset_state()

    def update_on_host(self, labels, predictions):
        """
        Compute and aggregate confusion_matrix every iteration.
        """
        labels = labels.detach().flatten()
        predictions = predictions.detach().flatten()

        target_mask = torch.logical_and(
            (labels >= 0), (labels < self.num_classes)
        )
        labels = labels[target_mask]
        predictions = predictions[target_mask]

        indices = self.num_classes * labels + predictions
        confusion_matrix = torch.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += confusion_matrix.to(self.confusion_matrix)

    def compute(self):
        """
        Returns the FBeta Score as a float using the aggregate confusion matrix.
        """
        return _compute_fbeta(
            self.confusion_matrix,
            self.num_classes,
            self.beta,
            self.average_type,
            self.ignore_labels,
        )

    def reset_state(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int32
        )


def _compute_fbeta(
    total_confusion_matrix,
    num_classes,
    beta: Optional[float] = 1.0,
    average_type: Optional[str] = "micro",
    ignore_labels: Optional[List] = None,
) -> float:
    """
    Computes f_beta metric from: confusion matrix for labels excluding 
    "ignore_labels"

    Args:
        total_confusion_matrix: Confusion matrix of size [num_classes, num_classes]
        num_classes: Number of classes. 
        beta: Beta coefficient in the F measure.
        average_type: Defines the reduction that is applied. Should be one 
        of the following:
            - 'micro' [default]: Calculate the metric globally, across all 
                samples and classes.
            - 'macro': Calculate the metric for each class separately, and 
                average the metrics across classes (with equal weights for 
                each class). This does not take label imbalance into account.
        ignore_labels: Integer specifying a target classes to ignore.
    """
    true_pos = torch.diagonal(total_confusion_matrix).type(torch.float32)
    predicted_per_class = total_confusion_matrix.sum(dim=0).type(torch.float32)
    actual_per_class = total_confusion_matrix.sum(dim=1).type(torch.float32)

    if ignore_labels:
        all_labels = torch.unsqueeze(torch.arange(num_classes), 1)
        ignore_labels = torch.tensor(ignore_labels)
        mask = torch.not_equal(all_labels, ignore_labels).all(dim=1)
    else:
        mask = torch.tensor([True,] * num_classes)

    mask = mask.type(torch.float32)
    num_labels_to_consider = mask.sum()
    beta = torch.tensor(beta).type(torch.float32)

    if average_type == "micro":
        precision = divide_no_nan(
            (true_pos * mask).sum(), (predicted_per_class * mask).sum()
        )
        recall = divide_no_nan(
            (true_pos * mask).sum(), (actual_per_class * mask).sum()
        )
        fbeta = divide_no_nan(
            (1.0 + beta ** 2) * precision * recall,
            (beta ** 2) * precision + recall,
        )
    else:  # "macro"
        precision_per_class = divide_no_nan(true_pos, predicted_per_class)
        recall_per_class = divide_no_nan(true_pos, actual_per_class)
        fbeta_per_class = divide_no_nan(
            (1.0 + beta ** 2) * precision_per_class * recall_per_class,
            (beta ** 2) * precision_per_class + recall_per_class,
        )
        precision = (precision_per_class * mask).sum() / num_labels_to_consider
        recall = (recall_per_class * mask).sum() / num_labels_to_consider
        fbeta = (fbeta_per_class * mask).sum() / num_labels_to_consider

    return float(fbeta)
