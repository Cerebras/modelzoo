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


class PoolerLayer(BaseLayer):
    """The pooler layer.

    Currently supports the following pooler types:

        - ``"mean"``: Mean reduction.
        - ``"max"``: Max reduction.
        - ``"first"``: First slice in the axis dimension.
        - ``"last"``: Last slice in the axis dimension.
        - ``"sum"``: Takes the sum over the axis dimension. Defaults to the\
        entire Tensor.
        - ``None``: No pooling (output=input).
    """

    def __init__(
        self,
        pooler_type="mean",
        axis=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(PoolerLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.pooler_type = pooler_type
        self.axis = axis

    def call(self, inputs, padding_mask=None, **kwargs):
        """Apply pooler of a given type.

        Takes in a padding mask with 1s for tokens and 0s for padding.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        if self.pooler_type == "mean":
            if padding_mask is None:
                output = tf.cast(
                    tf.reduce_mean(
                        input_tensor=tf.cast(inputs, tf.float32),
                        axis=self.axis,
                        keepdims=False,
                    ),
                    self.compute_dtype,
                )
            else:
                total_tokens = tf.cast(tf.reduce_sum(padding_mask), tf.float32,)
                sum_over_tokens = tf.reduce_sum(
                    input_tensor=tf.cast(inputs * padding_mask, tf.float32),
                    axis=self.axis,
                )
                output = tf.cast(
                    sum_over_tokens / total_tokens, self.compute_dtype
                )
        elif self.pooler_type == "max":
            output = tf.cast(
                tf.reduce_max(
                    input_tensor=tf.cast(inputs, tf.float32), axis=self.axis,
                ),
                self.compute_dtype,
            )
        elif self.pooler_type == "first":
            output = self._get_slice_using_gather(inputs, 0, self.axis,)
        elif self.pooler_type == "last":
            output = self._get_slice_using_gather(
                inputs, inputs.get_shape()[1] - 1, self.axis,
            )
        elif self.pooler_type == "sum":
            output = tf.cast(
                tf.reduce_sum(
                    input_tensor=tf.cast(inputs, tf.float32), axis=self.axis,
                ),
                self.compute_dtype,
            )
        elif self.pooler_type is None:
            output = inputs
        else:
            raise ValueError(f"Pooler type {self.pooler_type} not supported.")

        if self.tf_summary:
            output = summary_layer(output)
        return output

    def _get_slice_using_gather(self, inputs, index, axis, keepdims=False):
        assert (
            axis is not None
        ), f"axis=None is not supported for pooler type {self.pooler_type}."
        output = tf.gather(inputs, indices=index, axis=axis)
        return output
