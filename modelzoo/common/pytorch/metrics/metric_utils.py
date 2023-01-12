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
from torch import Tensor


def divide_no_nan(num: Tensor, denom: Tensor) -> Tensor:
    """
    Prevent zero division.
    Replicate the behavior of tf.math.divide_no_nan()
    """
    num = torch.where(denom == 0, torch.zeros_like(num), num)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return num / denom


def compute_confusion_matrix(
    labels: Tensor,
    predictions: Tensor,
    num_classes: int,
    weights: Tensor = None,
    on_device: bool = False,
) -> Tensor:
    """
    Computes the confusion matrix from predictions and labels.
    The matrix columns represent the prediction labels and the rows represent the
    real labels. The confusion matrix is always a 2-D array of shape `[n, n]`,
    where `n` is the number of valid labels for a given classification task.

    If `num_classes` is `None`, then `num_classes` will be set to one plus the
    maximum value in either predictions or labels. Class labels are expected to
    start at 0. For example, if `num_classes` is 3, then the possible labels
    would be `[0, 1, 2]`.

    If `weights` is not `None`, then each prediction contributes its
    corresponding weight to the total value of the confusion matrix cell.

    For example:
    ```
        confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
            [[0 0 0 0 0]
            [0 0 1 0 0]
            [0 0 1 0 0]
            [0 0 0 0 0]
            [0 0 0 0 1]]
    ```
    Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
    resulting in a 5x5 confusion matrix.

    Args:
        labels: `Tensor` of real labels for the classification task.
        predictions: `Tensor` of predictions for a given classification.
        weights: An optional `Tensor` whose shape matches `predictions`.
        num_classes: The possible number of labels the classification task can
                        have. If this value is not provided, it will be calculated
                        using both predictions and labels array.


    Returns:
        A `Tensor` with shape `[n, n]` representing the confusion
        matrix, where `n` is the number of possible labels in the classification
        task.
    Raises:
        ValueError: If `weights` is not `None` and its shape doesn't
            match `predictions`.
    """

    if len(labels.shape) > 1:
        labels = torch.flatten(labels)

    if weights is not None:
        if weights.shape != predictions.shape:
            raise ValueError(
                f"`predictions`={predictions.shape} and ",
                f"`weights`={weights.shape} have mismatched shapes",
            )
        if len(weights.shape) > 1:
            weights = torch.flatten(weights)

    if len(predictions.shape) > 1:
        predictions = torch.flatten(predictions)

    if not on_device:
        if torch.amin(labels) < 0:
            raise ValueError(
                f"Negative values in `labels` tensor is not allowed"
            )
        if torch.amin(predictions) < 0:
            raise ValueError(
                f"Negative values in `predictions` tensor is not allowed"
            )

        if num_classes is None:
            num_classes = (
                torch.amax(torch.maximum(labels, predictions)).item() + 1
            )

    confusion_matrix = torch.zeros(
        num_classes * num_classes,
        device=predictions.device,
        dtype=predictions.dtype,
    )
    index = labels * num_classes + predictions

    if weights is None:
        weights = torch.ones_like(
            predictions, device=predictions.device, dtype=predictions.dtype
        )

    if not on_device:
        index = index.to(torch.long)
    else:
        index = index.to(torch.int32)
    weights = weights.to(confusion_matrix.dtype)
    confusion_matrix.scatter_add_(dim=0, index=index, src=weights)
    confusion_matrix = torch.reshape(
        confusion_matrix, [num_classes, num_classes]
    )
    return confusion_matrix
