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

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.PoolerLayerV2 import PoolerLayerV2


class CLSLayer(BaseLayer):
    """
    CLS layer for BERT model https://arxiv.org/pdf/1810.04805.pdf
    Used for NSP and other sequence classification tasks, such as
    sentiment analysis.

    :param int hidden_size: Size of hidden dimension.
    :param int output_size: Size of output logits, so that the layer
                            output has shape ``[batch_size, output_size]``
    :param str pooler_type: Type of poolers currently supported:
                            {"first", "last", "mean", "max,", "sum"}
    :param str nonlinearity: Nonlinearity type (None, by default)
    :param bool use_bias: Use bias in the dense layer following pooler
    :param float dropout_rate: dropout rate for the hidden activations
    :param kernel_initializer: Kernel initializer. Defaults to "glorot_uniform".
    :param bias_initializer: Bias initializer. Defaults to "zeros".
    :param callable weight_regularizer: Weights regularizer.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        pooler_type="first",
        nonlinearity=None,
        use_bias=False,
        dropout_rate=0.0,
        dropout_seed=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        weight_regularizer=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        """
        Initialize the CLS layer object instance.
        """
        super(CLSLayer, self).__init__(boundary_casting, tf_summary, **kwargs)

        self.pooler_layer = PoolerLayerV2(
            pooler_type=pooler_type,
            axis=1,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="cls_pooler",
        )

        self.dense_layer = DenseLayer(
            hidden_size,
            nonlinearity,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer=weight_regularizer,
            bias_regularizer=weight_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="cls_dense",
        )

        self.dropout_layer = DropoutLayer(
            dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="cls_dropout",
        )

        self.output_dense_layer = DenseLayer(
            output_size,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=weight_regularizer,
            bias_regularizer=weight_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="cls_output_dense",
        )

    def call(self, inputs, training=True, padding_mask=None):
        pooled_input = self.pooler_layer(inputs, padding_mask=padding_mask)
        hidden_activations = self.dense_layer(pooled_input)
        hidden_activations = self.dropout_layer(
            hidden_activations, training=training
        )
        return self.output_dense_layer(hidden_activations)
