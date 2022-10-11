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


class FeedForwardNetworkV2(BaseLayer):
    """
    Implement a feed forward network as used in the T5 model.
    """

    def __init__(
        self,
        d_ff,
        d_model,
        activation="relu",
        dropout_rate=0.0,
        use_bias=False,
        input_layer_initializer="glorot_uniform",
        output_layer_initializer="glorot_uniform",
        dropout_seed=None,
        **kwargs,
    ):
        """
        Setup the FFN components

        :param int d_ff: The hidden dimension of the feed forward network, i.e.
            the output dimension of the first layer.
        :param int d_model: The output dimension of the feed forward network.
        :param string activation: The name of the activation to apply after
            the first dense layer.
        :param float dropout_rate: Dropout rate applied after the first dense
            layer.
        :param bool use_bias: Whether or not to use bias in the dense layers
            of the feed forward network.
        :param initializer input_layer_initializer: A string or initializer to
            use to initialize the weights of the first dense layer.
        :param initializer output_layer_initializer: A string or initializer to
            use to initialize the weights of the second dense layer.
        :param int dropout_seed: The seed to make the dropout layer
            deterministic.
        :param **kwargs: Keyword arguments to be passed into BaseLayer.
        """
        super(FeedForwardNetworkV2, self).__init__(**kwargs)

        self.dense_1 = DenseLayer(
            units=d_ff,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=input_layer_initializer,
            dtype=self.dtype_policy,
        )
        self.dense_2 = DenseLayer(
            units=d_model,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=output_layer_initializer,
            dtype=self.dtype_policy,
        )

        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout_layer = DropoutLayer(
                rate=self.dropout_rate,
                seed=dropout_seed,
                dtype=self.dtype_policy,
            )

    def call(self, inputs, training=True, **kwargs):
        x = self.dense_1(inputs)
        if self.dropout_rate:
            x = self.dropout_layer(x, training=training)
        return self.dense_2(x)
