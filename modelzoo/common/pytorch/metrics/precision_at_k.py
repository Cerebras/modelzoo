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
Precision@K metric for PyTorch.
"""
from typing import Optional

import torch

from modelzoo.common.pytorch.metrics.cb_metric import CBMetric


class PrecisionAtKMetric(CBMetric):
    """
    Precision@K takes the top K predictions and computes the true positive at K
    and false positive at K. For K = 1, it is the same as Precision.

    Precision@K is defined as follows:
    Precision@K = true_positive_at_k / (true_positive_at_k + false_positive_at_k).

    Internally, we keep track of true_positive_at_k and false_positive_at_k,
    weighted by `weights`.

    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    Args:
        :param Tensor labels: A `Tensor` of ground truth labels of type `int32`
            or `int64` and shape (batch_size, num_labels).
        :param Tensor predictions: A `Tensor` of predicted logit values for
            each class in the last dimention. It is of type `float` and shape
            (batch_size, num_classes).
        :param int k: The number of predictions for @k metric.
        :param Tensor weights: Optional `Tensor` whose rank is either 0, or n-1,
            where n is the rank of `labels`. If the latter, it must be
            broadcastable to `labels` (i.e., all dimensions must be either `1`,
            or the same as the corresponding `labels` dimension).
        :param name: Optional `string` which indicates name of the metric.
                If None or empty string, it defaults to the name of the class.

    Returns:
        precision_at_k: A float representing Precision@K.

    Raises:
        ValueError: If `weights` is not `None` and its shape doesn't match `predictions`
    """

    def __init__(self, k, name: Optional[str] = None):
        self.k = k
        super(PrecisionAtKMetric, self).__init__(name=name)

    def init_state(self):
        self.reset_state()

    def update_on_host(self, labels, predictions, weights=None):
        if weights is not None:
            if len(weights.shape) != 0 and weights.numel() != labels.shape[0]:
                raise ValueError(
                    f"`labels`={labels.shape} and `weights`={weights.shape} so"
                    f"`weights` must be a scalar or a vector of size "
                    f"{labels.shape[0]}"
                )
            weights = weights.detach()

        labels = labels.detach()
        predictions = predictions.detach()

        _, topk_pred_idx = torch.topk(predictions, self.k, dim=-1)

        # Computer the number of true positives per row
        lbl = labels.repeat_interleave(self.k, dim=1)
        pred_idx = topk_pred_idx.repeat(1, labels.shape[-1])
        intersection_per_row = torch.sum(lbl == pred_idx, dim=-1).float()

        if weights is not None:
            tp = intersection_per_row * weights
            fp = (self.k - intersection_per_row) * weights
        else:
            tp = intersection_per_row
            fp = self.k - intersection_per_row

        self.true_positive_at_k += torch.sum(tp).numpy()
        self.false_positive_at_k += torch.sum(fp).numpy()

    def compute(self):
        """Returns the Precision@K as a float."""
        return float(
            self.true_positive_at_k
            / (self.true_positive_at_k + self.false_positive_at_k)
        )

    def reset_state(self):
        self.true_positive_at_k = 0.0
        self.false_positive_at_k = 0.0
