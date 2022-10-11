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

from tensorflow.keras.layers import LayerNormalization

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class LayerNormalizationLayer(BaseLayer):
    """Wrapper around the Keras layer normalization.
    Reference: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_.
    """

    def __init__(
        self,
        axis=-1,
        epsilon=1e-8,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        trainable=True,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(LayerNormalizationLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.layer = LayerNormalization(
            axis=axis,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            trainable=trainable,
            name=self.name,
            dtype=self.dtype_policy,
        )

    def call(self, inputs, **kwargs):
        """Apply the layer normalization.

        Args:
            inputs (Tensor): Arbitrary tensor.

        Returns:
            Tensor: A normalized tensor of the same shape as input.

            **NOTE**: While ``**kwargs`` are passed, the training arg is
            never used.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)
        output = self.layer(inputs)
        if self.tf_summary:
            output = summary_layer(output)
        return output
