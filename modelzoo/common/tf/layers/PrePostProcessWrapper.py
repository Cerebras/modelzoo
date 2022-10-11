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

from modelzoo.common.tf.layers.AddLayer import AddLayer


class PrePostProcessWrapper:
    """Helper class that allows for a flexible specification of pre- and
    post-process operations wrapped around a given layer.
    """

    def __init__(
        self, layer, pre_process_config=[], post_process_config=[], **kwargs
    ):
        self.layer = layer
        self.pre_process_layers = self._setup_layer_list(pre_process_config)
        self.post_process_layers = self._setup_layer_list(post_process_config)

    def __call__(self, inputs, training=True, **kwargs):
        outputs = inputs
        other_outputs = None
        for layer in range(len(self.pre_process_layers)):
            outputs = self._apply_layer(
                self.pre_process_layers[layer], outputs, inputs, training
            )

        outputs = self.layer(outputs, training=training, **kwargs)

        if isinstance(outputs, (tuple, list)):
            # Layer returns multiple outputs.
            # In this case only the 0th entry
            # is propagated through post-process
            # layers, while the remaining outputs
            # are returned as are.
            other_outputs = outputs[1:]
            outputs = outputs[0]

        for layer in range(len(self.post_process_layers)):
            outputs = self._apply_layer(
                self.post_process_layers[layer], outputs, inputs, training
            )

        if other_outputs:
            outputs = (outputs,) + other_outputs

        return outputs

    def _setup_layer_list(self, layer_config):
        layer_list = []
        for l in range(len(layer_config)):
            layer = layer_config[l][0]
            layer_params = layer_config[l][1]
            layer_list.append(layer(**layer_params))
        return layer_list

    def _apply_layer(
        self, layer, inputs, residual_connection=None, training=True
    ):
        if isinstance(layer, AddLayer):
            assert (
                residual_connection is not None
            ), f"Residual connection not provided \
                for PrePostProcessWrapper layer {layer.name}."
            assert (
                inputs.get_shape().as_list()
                == residual_connection.get_shape().as_list()
                or None
                in inputs.get_shape().as_list()  # in gpu runs, can have partially defined shapes
            ), f"Residual connection and inputs shape mismatch \
                    in PrePostProcessWrapper layer {layer.name}."
            return layer([inputs, residual_connection], training=training)

        return layer(inputs, training=training)
