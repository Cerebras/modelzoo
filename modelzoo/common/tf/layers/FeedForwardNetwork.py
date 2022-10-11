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

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer


class FeedForwardNetwork(BaseLayer):
    """A feed forward network that consists of a stack of fully connected\
    layers.

    Args:
        layers_units (int): List of units for each layer.
        layers_activation (str): List of activation types (str) for each layer.
        layers_dropout_rates (float): List of dropout rates (float) for each
            layer.
        use_bias (bool): If ``True``, use bias throughout all layers.
        kernel_initializer (string): Kernel initializer. Defaults to
            ``"glorot_uniform"``.
        bias_initializer: Bias initializer. Defaults to ``"zeros"``.
        output_layer_initializer: If not None, initialize the last projection
            layer with this initializer. Defaults to None.
        kernel_regularizer (callable): Kernel regularizer.
        bias_initializer (callable): Bias regularizer.
        dropout_seed (int): Seed with which to initialize the dropout layer.
            Defaults to ``None``.

    """

    def __init__(
        self,
        layers_units,
        layers_activation=None,
        layers_dropout_rates=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        output_layer_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        dropout_seed=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        """Initialize the FFN object instance.
        """

        super(FeedForwardNetwork, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.num_dense_layers = len(layers_units)
        assert (
            self.num_dense_layers > 0
        ), "Number of dense layers should be at least 1."

        if layers_activation is not None:
            assert len(layers_activation) == self.num_dense_layers, (
                "len(layers_activation) should equal the number"
                " of dense layers."
            )
        if layers_dropout_rates is not None:
            assert len(layers_dropout_rates) == self.num_dense_layers, (
                "len(layers_dropout) should equal the number" "of dense layers."
            )

        self.layers = []
        for dense_layer in range(self.num_dense_layers):
            dense_initializer = kernel_initializer
            if (
                dense_layer == self.num_dense_layers - 1
                and output_layer_initializer is not None
            ):
                dense_initializer = output_layer_initializer
            self.layers.append(
                DenseLayer(
                    layers_units[dense_layer],
                    None,
                    use_bias,
                    dense_initializer,
                    bias_initializer,
                    kernel_regularizer,
                    bias_regularizer,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.dtype_policy,
                )
            )

            # Activation
            self.layers.append(
                ActivationLayer(
                    activation=layers_activation[dense_layer]
                    if layers_activation is not None
                    else None,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.dtype_policy,
                )
            )

            # Dropout
            self.layers.append(
                DropoutLayer(
                    rate=layers_dropout_rates[dense_layer]
                    if layers_dropout_rates is not None
                    else 0.0,
                    seed=dropout_seed,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.dtype_policy,
                )
            )

    def call(self, inputs, training=True, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return inputs
