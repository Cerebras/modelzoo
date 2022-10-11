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


class PoolerLayerV2(BaseLayer):
    """The pooler layer. Usually used for pooling or summarizing the
    sequence data.

    This layer is added as a workaround to the existing pooler layer for
    additional masking support. The plan is to use this layer for kernel
    matching and integ bring up. After we have full support for this layer,
    we should deprecate the old ``PoolerLayer``.

    Args:
        pooler_type (str): Type of pooling. Currently supports the following
        pooler types:

                - ``"mean"``: Mean reduction.
                - ``"max"``: Max reduction.
                - ``"first"``: First slice in the axis dimension.
                - ``"last"``: Last slice in the axis dimension (Not yet\
                supported)
                - ``"sum"``: Takes the sum over the axis dimension. Defaults to\
                the entire Tensor.

        axis (int): The dimensions to reduce. If None (the default), reduces
            all dimensions.
        boundary_casting (bool): If ``True``, outputs the values in half
            precision and casts the input values up to full precision.
        tf_summary (bool): If ``True``, saves the activations with
            ``summary_layer``.
    """

    def __init__(
        self,
        pooler_type,
        axis=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(PoolerLayerV2, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.pooler_type = pooler_type
        self.axis = axis

    def call(self, inputs, padding_mask=None):
        """Apply pooling with optional masking.

        Args:
            inputs (Tensor): Input tensor.
            padding_mask (Tensor): The padding mask tensor. Assumed to be
                1-based, i.e., has ``1`` in the non-padded positions and ``0``
                elsewhere. If the input tensor is of the shape
                ``[d0, d1, ..., d_{k-1}, d_{axis}, d_{k+1}, ... d_n]``, then
                the ``padding_mask`` must have the shape
                ``[d0, d1, ..., d_{k-1}, axis]`` or
                ``[d0, d1, ..., d_{k-1}, axis, 1, ..., 1]``. If ``None``
                (the default), a padding mask of all 1's is used.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)
        _pooler_type = self.pooler_type.lower()

        if _pooler_type == "first":
            output = tf.gather(inputs, indices=0, axis=self.axis)
        else:
            input_shape = inputs.shape.as_list()
            if self.axis and self.axis < 0:
                self.axis += len(input_shape)

            padding_mask_shape = self._get_mask_shape(input_shape, self.axis)

            if padding_mask is not None:
                self._check_mask_shape(inputs, padding_mask, self.axis)
                padding_mask = tf.reshape(padding_mask, padding_mask_shape)
            else:
                padding_mask = tf.ones(padding_mask_shape, inputs.dtype)

            if _pooler_type == "last":
                output = self._last_pool(inputs, padding_mask, self.axis)
            elif _pooler_type == "mean":
                output = self._mean_pool(inputs, padding_mask, self.axis)
            elif _pooler_type == "sum":
                output = self._sum_pool(inputs, padding_mask, self.axis)
            elif _pooler_type == "max":
                output = self._max_pool(inputs, padding_mask, self.axis)
            else:
                raise ValueError(
                    f"Pooler type {self.pooler_type} not supported."
                )

        if self.tf_summary:
            output = summary_layer(output)
        return output

    @staticmethod
    def _get_mask_shape(input_shape, axis):
        if axis is None:
            # full reduce over all dimensions
            # padding mask has to be the same shape as input
            return input_shape
        else:
            # broadcastable mask shape
            return input_shape[0 : axis + 1] + (len(input_shape) - axis - 1) * [
                1
            ]

    def _check_mask_shape(self, inputs, mask, axis):
        mask_shape = mask.shape.as_list()
        input_shape = inputs.shape.as_list()
        expected_mask_shape = self._get_mask_shape(input_shape, axis)
        if axis is None:
            assert (
                mask_shape == expected_mask_shape
            ), f"Mask has a wrong shape. Must have shape {expected_mask_shape}"
        else:
            assert mask_shape == expected_mask_shape or (
                mask_shape == expected_mask_shape[0 : axis + 1]
            ), f"Mask has a wrong shape. Must have shape {expected_mask_shape} \
            or {expected_mask_shape[0 : axis + 1]}"

    def _last_pool(self, inputs, mask, axis):
        input_shape = inputs.shape.as_list()
        assert (
            axis is not None and axis >= 0
        ), "_last_pool assumes that axis > 0"

        merged_dim = 1
        for dim in range(axis + 1):
            merged_dim *= input_shape[dim]

        flat_input = tf.reshape(inputs, shape=[merged_dim, -1])
        flat_indices = tf.reshape(
            tf.cast(
                tf.reduce_sum(
                    input_tensor=tf.cast(mask, tf.float32), axis=axis
                ),
                tf.int32,
            )
            - 1,
            shape=[-1],
        ) + tf.range(0, merged_dim, input_shape[axis])

        input_shape.pop(axis)
        return tf.reshape(
            tf.gather(flat_input, flat_indices, axis=0), shape=input_shape,
        )

    def _mean_pool(self, inputs, mask, axis):
        sum_over_masked = tf.reduce_sum(
            input_tensor=tf.cast(inputs * mask, tf.float32), axis=axis,
        )
        num_masked = tf.reduce_sum(
            input_tensor=tf.cast(mask, tf.float32), axis=axis,
        )

        return tf.cast(sum_over_masked / num_masked, inputs.dtype,)

    def _sum_pool(self, inputs, mask, axis):
        return tf.reduce_sum(input_tensor=inputs * mask, axis=axis)

    def _max_pool(self, inputs, mask, axis):
        neg_inf = inputs.dtype.min
        return tf.reduce_max(
            input_tensor=inputs * mask + (1.0 - mask) * neg_inf, axis=axis,
        )
