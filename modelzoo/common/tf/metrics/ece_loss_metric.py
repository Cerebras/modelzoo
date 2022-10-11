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

import tensorflow as tf

from modelzoo.common.tf.metrics.utils import (
    aggregate_across_replicas,
    metric_variable,
)


def ece_loss_metric(
    total_loss_per_batch,
    num_tokens,
    metrics_collections=None,
    updates_collections=None,
    name=None,
):
    """
    Custom TF evaluation metric for calculating Exponential Cross Entropy
        loss metric (ECE) over the validation set.

        Usage: Pass to Estimator through ``eval_metric_ops``, TF will accumulate
        ``loss/token`` over the entire validation set and use that value to calculate
        ECE.
        Based off of the Tensorflow mean metric code:
        (TF 2.1.0): tensorflow/python/ops/metrics_impl.py#L315-L393.
    """
    with tf.compat.v1.variable_scope(
        name, "ece_loss_metric", (total_loss_per_batch, num_tokens)
    ):
        total_loss = metric_variable([], tf.float32, name="total_loss")
        total_tokens = metric_variable([], tf.float32, name="total_tokens")
        update_total_loss_op = tf.compat.v1.assign_add(
            total_loss,
            tf.reduce_sum(
                input_tensor=tf.cast(total_loss_per_batch, tf.float32)
            ),
        )
        with tf.control_dependencies([total_loss_per_batch]):
            update_total_tokens_op = tf.compat.v1.assign_add(
                total_tokens, tf.cast(num_tokens, tf.float32),
            )

        def _compute_ece(_, total_loss_per_step, total_target_tokens_per_step):
            total_target_tokens_per_step = tf.cast(
                total_target_tokens_per_step, total_loss_per_step.dtype
            )
            return total_loss_per_step / total_target_tokens_per_step

        total_loss_t = aggregate_across_replicas(
            metrics_collections, _compute_ece, total_loss, total_tokens
        )

        update_op = _compute_ece(
            None, update_total_loss_op, update_total_tokens_op
        )

        if updates_collections:
            tf.compat.v1.add_to_collections(updates_collections, update_op)

        return total_loss_t, update_op
