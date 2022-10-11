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

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import summary_layer


class EmbeddingLayer(BaseLayer):
    """Embedding layer. Built on top of the Keras Embedding layer.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer='uniform',
        bias_initializer='zeros',
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        input_length=None,
        use_bias=False,
        weight_name='embedding_weights',
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(EmbeddingLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.bias_initializer = bias_initializer
        self.weight_name = weight_name

    def build(self, input_shape):
        if self.use_bias:
            self._embedding_bias = tf.cast(
                self.add_weight(
                    name='embedding_bias',
                    shape=[self.output_dim],
                    dtype=self.variable_dtype,
                    experimental_autocast=False,
                    initializer=self.bias_initializer,
                    regularizer=self.embeddings_regularizer,
                    trainable=True,
                ),
                self.compute_dtype,
            )
        self._embedding_table = 1.0 * tf.cast(
            self.add_weight(
                name=self.weight_name,
                shape=[self.input_dim, self.output_dim],
                dtype=self.variable_dtype,
                experimental_autocast=False,
                initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer,
                trainable=True,
            ),
            self.compute_dtype,
        )
        self.built = True

    def call(self, inputs, pad_id=-1, scale=1):
        """Get token embeddings of inputs.

        Args:
            inputs (Tensor): A tensor with shape ``[batch_size, length]``.
            pad_id: Integer specifying which input ID corresponds instead to
                padding. It does not need to be a legal vocabulary entry.
                Any ```inputs``` elements equal to this value will not be
                looked up, but instead directly output zeros.
                On the Wafer Scale Engine, this indicates the presence of
                variable sequence length.
            scale: Scaling of the embedding (in MLPERF ``hidden_size**0.5`` is
                used).

        Returns:
            embeddings (Tensor): A tensor of embeddings with shape
            ``[batch_size, length, hidden_size]``. Padded positions are
            filled with zeros.

        """

        # Create boolean array of size [batch_size, length]
        # where True = padding, False = not padding
        padding = tf.equal(inputs, pad_id)

        inputs = tf.where(
            padding,
            # use 0 ID for embedding_lookup, but outputs are masked
            # anyway, so it doesn't matter which valid ID is used
            tf.zeros_like(inputs),
            inputs,
        )

        output = tf.nn.embedding_lookup(
            params=self._embedding_table, ids=inputs,
        )

        if self.use_bias:
            output += self._embedding_bias

        if scale != 1:
            output *= scale

        # Set all padding embedding values to 0
        output = tf.where(
            tf.expand_dims(padding, -1), tf.zeros_like(output), output
        )

        if self.tf_summary:
            output = summary_layer(output)
        return output

    def embedding_table(self):
        if not self.built:
            self.build([0])  # dummy shape, actual shape not needed
        return self._embedding_table
