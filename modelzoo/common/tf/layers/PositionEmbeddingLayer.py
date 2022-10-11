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

from math import log

import tensorflow as tf

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class PositionEmbeddingLayer(BaseLayer):
    """Implementation of the position embedding layer.

    Adds positional information to the token embedding provided as input.
    Supports ``'fixed'`` and ``'learned'`` positional embeddings.

    Args:
        max_position_embeddings (int): Maximum sequence length to train using
            the model. If ``None``, set to the input sequence length.
        embedding_type (str): Options are ``'learned'`` or ``'fixed'``.

            - Learned: Trainable weights for embeddings.
            - Fixed: Fixed weights for embeddings.

        embeddings_initializer (callable): Embeddings initializer.
        embeddings_regularizer (callable): Embeddings regularizer.
        boundary_casting (bool): See the documentation for ``BaseLayer``.
        tf_summary: See the documentation for ``BaseLayer``.
        **kwargs: Additional keyword arguments for ``BaseLayer``.
    """

    def __init__(
        self,
        max_position_embeddings=None,
        embedding_type="fixed",
        embeddings_initializer='uniform',
        embeddings_regularizer=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(PositionEmbeddingLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.embedding_type = embedding_type
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.max_position_embeddings = max_position_embeddings

    def build(self, input_shape):
        seq_length = input_shape[1]
        if self.max_position_embeddings is None:
            self.max_position_embeddings = seq_length

        # ensure that input seq_length <= max_seq_length
        assert seq_length <= self.max_position_embeddings, (
            "Input sequence length is greater than maximum supported sequence length."
            + " Please check model definition"
        )

        embedding_size = input_shape[2]
        # create learned weights of dimensions [max_seq_length, embedding_size]
        if self.embedding_type == "learned":
            self.full_position_embedding_weights = self.add_weight(
                name="position_embedding_weights",
                shape=[self.max_position_embeddings, embedding_size],
                dtype=self.variable_dtype,
                experimental_autocast=False,
                initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer,
                trainable=True,
            )
        elif self.embedding_type == "fixed":
            self.fixed_position_embedding = tf.Variable(
                name="fixed_position_embedding",
                initial_value=lambda: self.setup_fixed_position_embedding(
                    seq_length, embedding_size
                ),
                dtype=self.variable_dtype,
                trainable=False,
            )

        self.built = True

    def call(self, inputs, position_ids=None):
        """Add position embeddings to the inputs.

        Args:
            inputs (Tensor): Input of the size
                ``[batch_size, seq_len, embedding_size]``.
            position_ids (Tensor): Position IDs of the inputs.A 1D tensor of
                size ``seq_len``. If ``None`` (default), assumes that
                corresponds to ``[0, 1, ..., seq_len-1]``.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        if self.embedding_type == "fixed":
            assert (
                position_ids is None
            ), "position_ids not supported for embedding_type='fixed'"
            output = inputs + tf.cast(
                self.fixed_position_embedding, self.compute_dtype
            )

        elif self.embedding_type == "learned":
            if position_ids is not None:
                assert (
                    position_ids.shape[0] == inputs.shape[1]
                ), "Number of position ids not equal to the number of input tokens"
                position_embedding_weights = tf.gather(
                    self.full_position_embedding_weights, position_ids,
                )
            else:
                position_embedding_weights = tf.slice(
                    self.full_position_embedding_weights,
                    [0, 0],
                    [tf.shape(inputs)[1], -1],
                )

            output = inputs + tf.cast(position_embedding_weights, inputs.dtype)
        elif self.embedding_type is None:
            output = inputs
        else:
            raise ValueError(
                f"Unsupported position embedding type {self.embedding_type}."
            )

        if self.tf_summary:
            output = summary_layer(output)
        return output

    def setup_fixed_position_embedding(
        self, length, channels, min_timescale=1.0, max_timescale=1.0e4
    ):
        """Adds several sinusoids of different frequencies to a Tensor.

        Each channel of the input Tensor is incremented by a sinusoid of a
        different frequency and phase.

        This allows the attention to learn to use absolute and relative
        positions. Timing signals should be added to some precursors of both
        the query and the memory inputs to the attention.

        The use of relative position is possible because ``sin(x+y)`` and
        ``cos(x+y)`` can be expressed in terms of ``y``, ``sin(x)`` and
        ``cos(x)``.

        In specific, this function uses a geometric sequence of timescales
        starting with  ``min_timescale`` and ending with ``max_timescale``.
        The number of different timescales is equal to ``channels / 2``. For
        each timescale, this function generates the two sinusoidal signals
        ``sin(timestep/timescale)`` and ``cos(timestep/timescale)``.  All
        these sinusoids are concatenated in the channels dimension.

        Args:
            min_timescale (float):
            max_timescale (float):

        Returns:
            Tensor: A tensor of the shape ``[length, channels]``. Based on
            `_get_timing_signal_1d
            <https://github.com/tensorflow/tensor2tensor/blob\
                /1843c72d1d5faf4c085bb198b5dde0908f4081d0/tensor2tensor/layers\
                /common_attention.py#L407>`_.

        """

        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = channels // 2
        log_timescale_increment = log(
            float(max_timescale) / float(min_timescale)
        ) / (tf.cast(num_timescales, tf.float32) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32)
            * -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0
        )
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.reshape(signal, [length, 2, num_timescales])
        signal = tf.transpose(a=signal, perm=[0, 2, 1])
        signal = tf.reshape(signal, [length, 2 * num_timescales])
        signal = tf.pad(
            tensor=signal, paddings=[[0, 0], [0, tf.math.mod(channels, 2)]]
        )

        return signal
