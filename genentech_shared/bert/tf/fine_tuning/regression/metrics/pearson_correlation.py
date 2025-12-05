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
Pearson Correlation metric to be used with TF Estimator.
"""
import tensorflow as tf

from modelzoo.common.tf.metrics.utils import (
    aggregate_across_replicas,
    metric_variable,
)


def pearson_correlation_metric(
    labels,
    preds,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None,
):
    """Computes the Pearson correlation between `labels` and `preds`.

    If `weights` is not None, then it is used to compute weighted correlation.

    To facilitate the computation of correlation across multiple batches of data,
    the function creates an `update_op` operation, which updates underlying
    variables and returns the updated correlation.

    :param preds: a `Tensor` of arbitrary size.
    :param labels: a `Tensor` of the same size as `preds`.
    :param weights: Optional `Tensor` indicating the frequency with which an example is
        sampled. Rank must be 0, or the same rank as `labels`, and must be
        broadcastable to `labels` (i.e., all dimensions must be either `1`, or
        the same as the corresponding `labels` dimension).
    :param metrics_collections: An optional list of collections that the metric
        value variable should be added to.
    :param updates_collections: An optional list of collections that the metric update
        ops should be added to.
    :param name: An optional variable scope name.

    :return:
        - pearson_correlation: A `Tensor` representing the Pearson correlation.
        - update_op: An operation that updates the local variables appropriately.
    """
    preds = tf.cast(preds, tf.float64)
    labels = tf.cast(labels, tf.float64)
    if weights is not None:
        weights = tf.cast(weights, tf.float64)
    with tf.compat.v1.variable_scope(
        name, "pearson_correlation", (preds, labels, weights)
    ):
        preds.get_shape().assert_is_compatible_with(labels.get_shape())

        count = metric_variable([], tf.float64, name="count")
        sum_preds = metric_variable([], tf.float64, name="sum_preds")
        sum_labels = metric_variable([], tf.float64, name="sum_labels")
        sum_squared_preds = metric_variable(
            [], tf.float64, name="sum_squared_preds"
        )
        sum_squared_labels = metric_variable(
            [], tf.float64, name="sum_squared_labels"
        )
        sum_products = metric_variable([], tf.float64, name="sum_products")

        if weights is None:
            batch_count = tf.cast(tf.size(labels), tf.float64)
            weighted_preds = preds
            weighted_labels = labels
        else:
            batch_count = tf.reduce_sum(weights)
            weighted_preds = preds * weights
            weighted_labels = labels * weights

        with tf.control_dependencies(
            [batch_count, weighted_preds, weighted_labels]
        ):
            update_count = tf.compat.v1.assign_add(count, batch_count)
            update_sum_preds = tf.compat.v1.assign_add(
                sum_preds, tf.reduce_sum(weighted_preds)
            )
            update_sum_labels = tf.compat.v1.assign_add(
                sum_labels, tf.reduce_sum(weighted_labels)
            )
            update_sum_squared_preds = tf.compat.v1.assign_add(
                sum_squared_preds,
                tf.reduce_sum(weighted_preds * weighted_preds),
            )
            update_sum_squared_labels = tf.compat.v1.assign_add(
                sum_squared_labels,
                tf.reduce_sum(weighted_labels * weighted_labels),
            )
            update_sum_products = tf.compat.v1.assign_add(
                sum_products, tf.reduce_sum(weighted_preds * weighted_labels)
            )

        def _compute_pearson_correlation(
            _,
            count,
            sum_preds,
            sum_labels,
            sum_squared_preds,
            sum_squared_labels,
            sum_products,
        ):
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
            numerator = count * sum_products - sum_preds * sum_labels
            denominator = tf.sqrt(
                count * sum_squared_preds - sum_preds * sum_preds
            ) * tf.sqrt(count * sum_squared_labels - sum_labels * sum_labels)
            pearson_correlation = numerator / denominator
            return pearson_correlation

        pearson_correlation = aggregate_across_replicas(
            metrics_collections,
            _compute_pearson_correlation,
            count,
            sum_preds,
            sum_labels,
            sum_squared_preds,
            sum_squared_labels,
            sum_products,
        )

        update_op = _compute_pearson_correlation(
            None,
            update_count,
            update_sum_preds,
            update_sum_labels,
            update_sum_squared_preds,
            update_sum_squared_labels,
            update_sum_products,
        )

        if metrics_collections:
            tf.compat.v1.add_to_collections(
                metrics_collections, pearson_correlation
            )

        if updates_collections:
            tf.compat.v1.add_to_collections(updates_collections, update_op)

        return pearson_correlation, update_op
