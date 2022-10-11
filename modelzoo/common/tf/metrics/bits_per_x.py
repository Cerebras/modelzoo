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

import math

import tensorflow as tf

from modelzoo.common.tf.metrics.utils import (
    aggregate_across_replicas,
    metric_variable,
)


def _get_bpb_constant(dataset="pile"):
    """Bits per byte has fixed constants (tokens per byte) that depend on the
    dataset used. The values are derived from Table 7 of the paper:
    `The Pile: An 800GB Dataset of Diverse Text for Language Modeling
    <https://arxiv.org/abs/2101.00027>`_.

    Args:
        dataset (str): The dataset to get constant for. Defaults to `pile`.

    Returns:
        Multiplicative factor for metric computation
    """

    if dataset not in ["pile", "openwebtext2"]:
        print(
            f"Currently supporting pile and openwebtext2, got {dataset}."
            + f" Reverting to a default value of 1.0. To estimate the correct"
            + f" bits per byte, add the relevant tokens_per_byte value for the"
            + f" dataset being passed, or revert to using provided values."
        )

    vals = {
        "pile": 0.29335,
        "openwebtext2": 0.2434,
    }
    # defaulting to 1.0 as we do not have any provided values for tokens_per_byte
    # this needs to be updated for custom datasets.
    return vals.get(dataset, 1.0)


def calculate_bits_per_x(
    total_loss_per_step,
    total_target_tokens_per_step,
    bits_per_type="per_byte",
    dataset="pile",
):
    """Calculates the bits per `type` per target token, where type is one of
        `byte`, `character`, or `word`.

    Args:
        total_target_tokens_per_step (float): The total loss summed over a
            train step or perhaps a longer period.
        total_target_tokens_per_step (int): The total number of target tokens
            seen in that period.
        bits_per_type (str): The type of bits_per metric to compute. This is one
            of `per_byte`, `per_character`, or `per_word`.
            Defaults to `per_byte`.
        dataset (str): The dataset to compute metric for. This is needed for
            the `bits_per_byte` metric. Defaults to `pile`.

    Returns:
        The bits_per metric computed based on the specified type and dataset
    """

    assert bits_per_type in [
        "per_byte",
        "per_character",
        "per_word",
    ], f"Expected one of per_byte, per_character or per_word, got {bits_per_type}"

    def _bp_compute(x):
        # convert base from e -> 2 as we need to compute bit level metrics
        x /= math.log(2)
        if bits_per_type == "per_byte":
            # get the tokens per byte of the corresponding dataset
            return x * _get_bpb_constant(dataset)
        return x

    total_target_tokens_per_step = tf.cast(
        total_target_tokens_per_step, total_loss_per_step.dtype
    )
    return _bp_compute(total_loss_per_step / total_target_tokens_per_step)


def bits_per_x_metric(
    total_loss_per_batch,
    num_tokens,
    bits_per_type="per_byte",
    dataset="pile",
    metrics_collections=None,
    updates_collections=None,
    name=None,
):
    """
    Custom TF evaluation meric for calculating bits per x over the validation
    set, where x is one of `byte`, `character`, or `word`.

    Pass to Estimator through ``eval_metric_ops``, TF will accumulate ``loss/token``
    over the entire validation set and use that value to calculate bits per byte,
    character, or word.
    """

    with tf.compat.v1.variable_scope(
        name, 'bpb_metric', (total_loss_per_batch, num_tokens)
    ):
        total_loss = metric_variable([], tf.float32, name='total_loss')
        total_tokens = metric_variable([], tf.float32, name='total_tokens')
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

        def _compute_bpb(_, total_loss, total_tokens):
            return calculate_bits_per_x(
                total_loss, total_tokens, bits_per_type, dataset
            )

        bpb_t = aggregate_across_replicas(
            metrics_collections, _compute_bpb, total_loss, total_tokens
        )

        update_op = _compute_bpb(
            None, update_total_loss_op, update_total_tokens_op
        )

        if updates_collections:
            tf.compat.v1.add_to_collections(updates_collections, update_op)
        return bpb_t, update_op
