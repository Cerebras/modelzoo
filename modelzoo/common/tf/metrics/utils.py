# This code is adapted from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/metrics_impl.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.

    If running in a `DistributionStrategy` context, the variable will be
    "sync on read". This means:

    *   The returned object will be a container with separate variables
            per replica of the model.

    *   When writing to the variable, e.g. using `assign_add` in a metric
            update, the update will be applied to the variable local to the
            replica.

    *   To get a metric's result value, we need to sum the variable values
            across the replicas before computing the final answer. Furthermore,
            the final answer should be computed once instead of in every
            replica. Both of these are accomplished by running the computation
            of the final result value inside
            `distribution_strategy_context.get_replica_context().merge_call(fn)`.
            Inside the `merge_call()`, ops are only added to the graph once
            and access to a sync on read variable in a computation returns
            the sum across all replicas.

    Returns:
        A (non-trainable) variable initialized to zero, or if inside a
        `DistributionStrategy` scope a sync on read variable container.

    param int shape: Shape of the created variable.
    param int dtype: Type of the created variable.
    param bool validate_shape: (Optional) Whether shape validation is enabled for
        the created variable.
    param string name: (Optional) String name of the created variable.
    """
    # Note that synchronization "ON_READ" implies trainable=False.
    return tf.compat.v1.Variable(
        initial_value=lambda: tf.zeros(shape, dtype),
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
        ],
        validate_shape=validate_shape,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.SUM,
        name=name,
    )


def streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.

    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.

    Returns:
        total_cm: A `Tensor` representing the confusion matrix.
        update_op: An operation that increments the confusion matrix.

    param Tensor labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    param Tensor predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
    param int num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
    param Tensor weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = metric_variable(
        [num_classes, num_classes], tf.float64, name='total_confusion_matrix'
    )

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = tf.cast(predictions, tf.int64)
    labels = tf.cast(labels, tf.int64)
    num_classes = tf.cast(num_classes, tf.int64)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = tf.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = tf.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = tf.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = tf.math.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=tf.float64
    )
    update_op = tf.compat.v1.assign_add(total_cm, current_cm)
    return total_cm, update_op


def aggregate_across_replicas(metrics_collections, metric_value_fn, *args):
    """Aggregate metric value across replicas."""

    def fn(distribution, *a):
        """Call `metric_value_fn` in the correct control flow context."""
        if hasattr(distribution.extended, '_outer_control_flow_context'):
            # If there was an outer context captured before this method was called,
            # then we enter that context to create the metric value op. If the
            # captured context is `None`, tf.control_dependencies(None) gives the
            # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
            # captured context.
            # This special handling is needed because sometimes the metric is created
            # inside a while_loop (and perhaps a TPU rewrite context). But we don't
            # want the value op to be evaluated every step or on the TPU. So we
            # create it outside so that it can be evaluated at the end on the host,
            # once the update ops have been evaluated.

            # pylint: disable=protected-access
            if distribution.extended._outer_control_flow_context is None:
                with tf.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution.extended._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution.extended._outer_control_flow_context.Exit()
                # pylint: enable=protected-access
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            tf.compat.v1.add_to_collections(metrics_collections, metric_value)
        return metric_value

    return tf.distribute.get_replica_context().merge_call(fn, args=args)
