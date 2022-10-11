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
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class SegmentEmbeddingLayer(BaseLayer):
    """Segment embedding layer. Adds segment information. For example,
    to which sentence the token belongs when an input sequence contains
    multiple sentences, such as two in the case of BERT model, to the token
    embedding provided as input.

    Args:
        num_segments (int): Number of encoded segments.
        embeddings_regularizer (callable): Embeddings regularizer.
    """

    def __init__(
        self,
        num_segments=2,
        embeddings_initializer='uniform',
        embeddings_regularizer=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(SegmentEmbeddingLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.num_segments = num_segments
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer

    def build(self, input_shape):
        self.segment_emb_weights = self.add_weight(
            name='segment_embedding_weights',
            shape=[self.num_segments, input_shape[2]],
            dtype=self.variable_dtype,
            experimental_autocast=False,
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, segment_ids):
        """Add segment embedding to inputs.

        Args:
            inputs: Tensor of input embeddings.
            segment_ids: Segment IDs.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        [batch_size, length, hidden_size] = inputs.get_shape()
        flat_segment_ids = tf.reshape(segment_ids, [-1])
        one_hot_ids = tf.one_hot(flat_segment_ids, depth=self.num_segments)
        segment_embeddings = tf.cast(
            tf.matmul(one_hot_ids, self.segment_emb_weights), inputs.dtype
        )
        segment_embeddings = tf.reshape(
            segment_embeddings, [batch_size, length, hidden_size]
        )
        output = inputs + segment_embeddings
        if self.tf_summary:
            output = summary_layer(output)
        return output
