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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class ActivationLayer(BaseLayer):
    """Wrapper around the Keras activation layer.

    Also supports ``activation="GeLU"``, ``activation="relu6"`` and
    ``activation="hard_swish"`` which are currently missing in
    ``keras.layers.ActivationLayer`` v2.2.

    Args:
        activation (Union[str, Callable]): The function to be applied. This can
            either be callable string name of a Tensorflow built-in activation,
            or one of ``"gelu"``, ``"lrelu"`` (``lrelu`` denotes LeakyReLU),
            ``"relu6"`` or ``"hard_swish"``.
        boundary_casting (bool): If ``True``, outputs the values in half
            precision and casts the input values up to full precision.
        tf_summary (bool): If ``True``, saves the activations with the
            ``summary_layer``.
    """

    def __init__(
        self, activation, boundary_casting=False, tf_summary=False, **kwargs,
    ):
        super(ActivationLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.layer = None
        if isinstance(activation, str):
            if activation.lower() == "gelu":
                self.layer = self.gelu
            elif activation.lower() == "lrelu":
                raise ValueError("We only support LeakyReLU as a callable.")
            elif activation.lower() == "relu6":
                self.layer = self.relu6
            elif activation.lower() == "hard_swish":
                self.layer = self.hard_swish

        if self.layer is None:
            self.layer = Activation(
                activation, name=self.name, dtype=self.dtype_policy
            )

    def call(self, inputs, **kwargs):
        """
        Apply the activation layer.

        Args:
            inputs: Arbitrary tensor.
        Returns:
            Tensor: A tensor of the same shape as the input.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        if self.tf_summary:
            inputs = summary_layer(inputs)

        output = self.layer(inputs)
        if self.tf_summary:
            output = summary_layer(output)
        return output

    @staticmethod
    def gelu(x):
        cdf = 0.5 * (
            1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
        )
        return x * cdf

    @staticmethod
    def relu6(x):
        return tf.nn.relu6(x)

    @staticmethod
    def hard_swish(x):
        return x * tf.nn.relu6(x + 3) / 6
