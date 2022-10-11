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
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.layers.utils import summary_layer
from modelzoo.common.tf.model_utils.reshape_gather import reshape_gather


class MLMLayer(BaseLayer):
    """
    MLM layer for BERT model https://arxiv.org/pdf/1810.04805.pdf.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        embedding_size=None,
        nonlinearity="gelu",
        use_ffn_bias=False,
        use_output_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        weight_regularizer=None,
        layer_norm_epsilon=1e-8,
        boundary_casting=False,
        tf_summary=False,
        enable_gpu_optimizations=False,
        **kwargs,
    ):
        super(MLMLayer, self).__init__(boundary_casting, tf_summary, **kwargs)

        use_projected_encoder_output = (
            embedding_size is not None and embedding_size != hidden_size
        )

        dense_layer_size = hidden_size
        if use_projected_encoder_output:
            # Embedding dimension is different from encoder hidden dimension,
            # so project back to embedding dimension
            dense_layer_size = embedding_size
            self.encoder_output_projection = DenseLayer(
                embedding_size,
                activation=None,
                use_bias=False,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.dtype_policy,
                name="output_embed_projection",
            )

        self.dense_layer = DenseLayer(
            dense_layer_size,
            nonlinearity,
            use_bias=use_ffn_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=weight_regularizer,
            bias_regularizer=weight_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.dtype_policy,
            name="mlm_ffn",
        )

        self.layer_norm_layer = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            beta_regularizer=weight_regularizer,
            gamma_regularizer=weight_regularizer,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.dtype_policy,
            name="mlm_layer_norm",
        )

        self.use_output_bias = use_output_bias
        self.use_projected_encoder_output = use_projected_encoder_output
        self.output_size = output_size
        self.weight_regularizer = weight_regularizer
        self.enable_gpu_optimizations = enable_gpu_optimizations

    def build(self, input_shape):
        if self.use_output_bias:
            self.bias = self.add_weight(
                name='mlm_bias',
                shape=[self.output_size],
                dtype=self.variable_dtype,
                experimental_autocast=False,
                initializer=tf.compat.v1.zeros_initializer,
                regularizer=self.weight_regularizer,
                trainable=True,
            )
        self.built = True

    def call(self, inputs, masked_lm_positions, embedding_table):
        [batch_size, length, hidden_size] = inputs.get_shape()
        max_predictions_per_seq = masked_lm_positions.get_shape()[1]
        assert (
            length >= max_predictions_per_seq
        ), "Max number of predictions larger than max sequence length."
        masked_inputs = reshape_gather(
            inputs, masked_lm_positions, self.enable_gpu_optimizations
        )

        if self.use_projected_encoder_output:
            # Embedding dimension is different from encoder hidden dimension,
            # so project back to embedding dimension
            masked_inputs = self.encoder_output_projection(masked_inputs)

        # FFN + embedding
        masked_inputs = self.dense_layer(masked_inputs)

        masked_inputs = self.layer_norm_layer(masked_inputs)

        assert (
            embedding_table.shape[0] == self.output_size
        ), "Vocabulary size and last embedding dimension mismatch."

        output = tf.matmul(
            masked_inputs,
            tf.cast(embedding_table, masked_inputs.dtype),
            transpose_b=True,
        )

        # Add bias to token logits
        if self.use_output_bias:
            output += tf.cast(self.bias, output.dtype)

        if self.tf_summary:
            output = summary_layer(output)

        if self.enable_gpu_optimizations:
            # On GPUs, leave `masked_inputs` flattened for more efficient use
            # of memory. In particular, by using a standard matrix
            # multiplication (rather than BatchMatMul), the gradient
            # calculation will use a MatMul and sum gradients across the batch
            # dimension inside the MatMul kernel.
            # Here, reshape the output to recover the batch dimension after
            # using a flattened matrix multiplication above.
            output_shape = [batch_size, max_predictions_per_seq, -1]
            output = tf.reshape(output, output_shape)

        return output
