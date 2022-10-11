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

from tensorflow.keras.layers import Dense

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class DenseLayer(BaseLayer):
    """Wrapper around the Keras densely-connected layer. Provides support for
    ``"gelu"`` activation.

    Args:
        units (int): Number of units in the layer output.
        activation (Optional[Union[str, Callable]]): If not ``None``, an
            activation function to be applied after the dense layer. The
            activation function can either be a callable string name of a
            Tensorflow built-in activation, or ``"gelu"``.
        use_bias (bool): Whether to use bias.
        kernel_initializer (str): Kernel intializer. Defaults to
            ``"glorot_uniform"``.
        kernel_initializer (str): Bias intializer. Defaults to ``"zeros"``.
        kernel_regularizer (Optional[Callable]): Kernel regularizer. Defaults
            to ``None``.
        bias_regularizer (Optional[Callable]): Bias regularizer. Defaults to
            ``None``.
        activity_regularizer (Optional[Callable]): Activity (output activation)
            regularizer. Defaults to ``None``.
        kernel_constraint (Optional[Callable]): Kernel constraint. Defaults
            to ``None``.
        bias_constraint (Optional[Callable]): Bias constraint. Defaults to
            ``None``.
        boundary_casting (bool): If ``True``, outputs the values in half
            precision and casts the input values up to full precision.
        tf_summary (bool): If ``True``, saves the activations with
            ``summary_layer``.


    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(DenseLayer, self).__init__(boundary_casting, tf_summary, **kwargs)

        if activation == "gelu":
            self.activation_layer = ActivationLayer(
                activation,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.dtype_policy,
            )
        else:
            self.activation_layer = None

        self.layer = Dense(
            units,
            activation if self.activation_layer is None else None,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            name=self.name,
            dtype=self.dtype_policy,
        )

    def call(self, inputs, **kwargs):
        """Apply the densely-connected layer.

        Args:
            inputs (Tensor): An N-D tensor with shape:
                ``(batch_size, ..., input_dim)``.

        Returns:
            Tensor: An N-D tensor with shape:
            ``(batch_size, ..., units)``.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)
        output = self.layer(inputs)
        if self.activation_layer is not None:
            output = self.activation_layer(output)
        if self.tf_summary:
            output = summary_layer(output)
        return output
