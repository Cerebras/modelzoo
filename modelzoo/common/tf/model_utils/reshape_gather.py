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


def reshape_gather(inputs, masked_lm_positions, do_reshape=True):
    """
    Gather elements from inputs tensor based on mask provided.
    :param Tensor inputs: Input to get gathered elements from.
    :param Tensor masked_lm_positions: Indices to call tf.gather and get elements by.
    :param bool do_reshape: If set True, the size of the output is 3D tensor of
        shape [batch_size, length, hidden_size], otherwise,
        2D tensor is returned with shape of [batch_size * length, hidden_size].
    :returns: Tensor with gathered elements from inputs tensor.
    """
    [batch_size, length, hidden_size] = inputs.get_shape()
    # tf.gather can't handle different indices per batch, so flatten first

    # Gather masked positions into 2D tensor of shape
    # [batch_size * max_predictions_per_seq, hidden_size]
    flat_inputs = tf.reshape(inputs, [batch_size * length, hidden_size])
    # add the max_seq_len offsets
    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * length, [-1, 1]
    )
    flat_positions = tf.reshape(masked_lm_positions + flat_offsets, [-1])

    # do the gather on the flattened inputs
    masked_inputs = tf.gather(flat_inputs, flat_positions)

    if not do_reshape:
        # When compiling through the Cerebras stack, use a batched matrix
        # multiplication, to ensure the batch dimension can be kernel
        # matched to stream batch samples through the model.
        max_predictions_per_seq = masked_lm_positions.get_shape()[1]
        gather_shape = [batch_size, max_predictions_per_seq, hidden_size]
        # Reshape back to 3D tensor
        masked_inputs = tf.reshape(masked_inputs, gather_shape)

    return masked_inputs
