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
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.utils import summary_layer
from modelzoo.common.tf.model_utils.reshape_gather import reshape_gather


class MultipleCLSLayer(BaseLayer):
    """
    Multiple CLS layer for BERT model https://arxiv.org/abs/1903.10318.
    """

    def __init__(
        self,
        output_size,
        nonlinearity=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        enable_gpu_optimizations=False,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(MultipleCLSLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.dense_layer = DenseLayer(
            output_size,
            activation=nonlinearity,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="multiple_cls_dense",
        )

        self.enable_gpu_optimizations = enable_gpu_optimizations

    def call(self, inputs, cls_tokens_positions):
        # Use the same gather technique and apply same gpu
        # optimizations as in MLMLayer.
        masked_inputs = reshape_gather(
            inputs, cls_tokens_positions, self.enable_gpu_optimizations
        )
        output = self.dense_layer(masked_inputs)

        if self.enable_gpu_optimizations:
            # On GPUs, leave `masked_inputs` flattened for more efficient use
            # of memory. In particular, by using a standard matrix
            # multiplication (rather than BatchMatMul), the gradient
            # calculation will use a MatMul and sum gradients across the batch
            # dimension inside the MatMul kernel.
            # Here, reshape the output to recover the batch dimension after
            # using a flattened matrix multiplication above.
            max_cls_tokens = cls_tokens_positions.get_shape()[1]
            output = tf.reshape(
                output, [inputs.get_shape()[0], max_cls_tokens, -1]
            )

        if self.tf_summary:
            output = summary_layer(output)

        return output
