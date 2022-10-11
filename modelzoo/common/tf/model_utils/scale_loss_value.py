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

_EPSILON = 1e-5


def scale_loss_value(
    loss, label_weights, scale_type, batch_size, output_type=tf.float32
):
    """
    Performs different types of scaling of the loss value.
    :param Tensor loss: The loss value to scale.
    :param Tensor label_weights:
        The mask of labels to use for modes `num_masked` or `num_cls`.
    :param str scale_type: Scale type one of `num_masked`, `num_cls`,
        "batch_size" or None.
    :param int batch_size: Required if scale type is batch_size.
    :param tf.dtype output_type: Type of the output. If None is specified
        no type casting is performed.
    :returns: The scaled loss value.
    """
    if scale_type is None:
        denominator = 1.0
    elif scale_type in {"num_masked", "num_cls"}:
        denominator = (
            tf.cast(tf.reduce_sum(input_tensor=label_weights), loss.dtype)
            + _EPSILON
        )
    else:
        if batch_size is None:
            raise ValueError(
                "batch_size must be provided when scale type is batch_size."
            )
        denominator = batch_size

    loss_scaled = loss / denominator

    if output_type:
        loss_scaled = tf.cast(loss_scaled, output_type)

    return loss_scaled
