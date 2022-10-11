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
Matthews Correlation Coefficient metric to be used with TF Estimator.
"""

import tensorflow as tf

from modelzoo.common.tf.metrics.utils import (
    aggregate_across_replicas,
    streaming_confusion_matrix,
)


def mcc_metric(
    labels,
    predictions,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None,
):
    """
    Custom TF evaluation meric for calculating MCC when performing binary
    classification.
        
    Usage: Pass outputs to Estimator through ``eval_metric_ops``.
        
    The predictions are accumulated in a confusion matrix, weighted by `weights`,
    and MCC is then calculated from it.

    If `weights` is None, weights default to 1. Use weights of 0 to mask values.

    Returns:
        mcc: A `Tensor` representing the Matthews Correlation Coefficient.
        update_op: An operation that increments the confusion matrix.

    :param Tensor labels: A `Tensor` of binary labels with shape [batch size] and
        of type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    :param Tensor predictions: A `Tensor` of binary predictions, whose shape is 
        [batch size] and type `int32` or `int64`. The tensor will beflattened if 
        its rank > 1.
    :param Tensor weights: Optional `Tensor` whose rank is either 0, or the same rank
        as `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
    :param List metrics_collections: An optional list of collections that
        `mcc` should be added to.
    :param List updates_collections: An optional list of collections `update_op`
        should be added to.
    :param string name: An optional variable_scope name.
    """

    if tf.executing_eagerly():
        raise RuntimeError(
            "mcc metric is not supported when eager execution is enabled."
        )

    with tf.compat.v1.variable_scope(
        name, 'mcc_metric', (predictions, labels, weights)
    ):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        total_cm, update_op = streaming_confusion_matrix(
            labels, predictions, num_classes=2, weights=weights
        )

        def _compute_mcc(_, total_cm):
            """Compute the MCC via the confusion matrix."""
            # extract true/false negatives/positives from confusion matrix.
            tp = total_cm[1][1]
            fp = total_cm[0][1]
            tn = total_cm[0][0]
            fn = total_cm[1][0]

            numerator = tp * tn - fp * fn
            denominator = tf.math.sqrt(
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            )

            return tf.math.divide_no_nan(numerator, denominator)

        mcc = aggregate_across_replicas(
            metrics_collections, _compute_mcc, total_cm
        )

        if updates_collections:
            tf.compat.v1.add_to_collection.add_to_collections(
                updates_collections, update_op
            )

        return mcc, update_op
