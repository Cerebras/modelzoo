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
Dice coefficient metric to be used with TF Estimator,
"""

import tensorflow as tf

from modelzoo.common.tf.metrics.utils import (
    aggregate_across_replicas,
    streaming_confusion_matrix,
)


def dice_coefficient_metric(
    labels,
    predictions,
    num_classes,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None,
):
    """Calculate per-step Dice Coefficient.
    Dice Coefficient is a common evaluation metric for
    semantic image segmentation.
    Dice Coefficient is defined as follows:
        Dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by `weights`,
    and dice ceoffiecient is then calculated from it.
    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the `dice_coefficient`.
    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    Returns:
        dice_coefficient: A `Tensor` representing the dice coefficient.
        update_op: An operation that increments the confusion matrix.
    Raises:
        ValueError: If `predictions` and `labels` have mismatched shapes, or if
        `weights` is not `None` and its shape doesn't match `predictions`, or if
        either `metrics_collections` or `updates_collections` are not a list or
        tuple.
        RuntimeError: If eager execution is enabled.

    :param Tensor labels: A `Tensor` of ground truth labels with shape [batch size] and
        of type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    :param Tensor predictions: A `Tensor` of prediction results for semantic labels,
        whose shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
    :param int num_classes: The possible number of labels the prediction task can have.
        This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
    :param Tensor weights: Optional `Tensor` whose rank is either 0, or the same rank
        as `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
    :param List metrics_collections: An optional list of collections that
        `dice_coefficient` should be added to.
    :param List updates_collections: An optional list of collections `update_op`
        should be added to.
    :param string name: An optional variable_scope name.
    """

    if tf.executing_eagerly():
        raise RuntimeError(
            'dice_coefficient metric is not supported when '
            'eager execution is enabled.'
        )

    with tf.compat.v1.variable_scope(
        name, 'dice_coefficient', (predictions, labels, weights)
    ):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        total_cm, update_op = streaming_confusion_matrix(
            labels, predictions, num_classes, weights
        )

        def compute_dice_coefficient(_, total_cm):
            """Compute the dice coefficient via the confusion matrix."""
            sum_over_row = tf.cast(tf.reduce_sum(total_cm, 0), tf.float32)
            sum_over_col = tf.cast(tf.reduce_sum(total_cm, 1), tf.float32)
            cm_diag = tf.cast(tf.linalg.diag_part(total_cm), tf.float32)
            denominator = sum_over_row + sum_over_col

            # The mean is only computed over classes that appear in the
            # label or prediction tensor. If the denominator is 0, we need to
            # ignore the class.
            num_valid_entries = tf.reduce_sum(
                tf.cast(tf.math.not_equal(denominator, 0), dtype=tf.float32)
            )

            # If the value of the denominator is 0, set it to 1 to avoid
            # zero division.
            denominator = tf.where(
                tf.math.greater(denominator, 0),
                denominator,
                tf.ones_like(denominator),
            )
            dice = tf.math.divide(2 * cm_diag, denominator)

            # If the number of valid entries is 0 (no classes) we return 0.
            result = tf.where(
                tf.math.greater(num_valid_entries, 0),
                tf.reduce_sum(dice, name='dice_coefficient')
                / num_valid_entries,
                0,
            )
            return result

        dice_coefficient_v = aggregate_across_replicas(
            metrics_collections, compute_dice_coefficient, total_cm
        )

        if updates_collections:
            tf.compat.v1.add_to_collection.add_to_collections(
                updates_collections, update_op
            )

        return dice_coefficient_v, update_op
