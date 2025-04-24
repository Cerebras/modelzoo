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

import contextlib

import torch.nn

from cerebras.modelzoo.tools.gns.per_example_grad_norm_layers import (
    PEGradNormShimEmbedding,
    PEGradNormShimLayerNorm,
    PEGradNormShimLinear,
)


class ShimLayerState:
    def __init__(
        self,
        library_pointer,
        submodule_name: str,
        shim_class,
    ):
        self.enabled = False
        self.library_pointer = library_pointer
        self.library_submodule_name = submodule_name
        self.base_class = library_pointer.__getattribute__(submodule_name)
        self.shim_class = shim_class
        # Verify that the shim_class is a super class of the base_class.
        # This ensures that the forward, backward functionality of the
        # shim layer can just access the super class versions of the
        # forward and backward functionality.
        assert self.base_class in shim_class.__bases__

    def __repr__(self):
        return (
            f'<PerExampleGradNormShim.ShimLayerState,\n'
            f'\tsubmodule_name = {self.library_submodule_name},\n'
            f'\tlibrary_pointer = {self.library_pointer},\n'
            f'\tenabled = {self.enabled}>'
        )

    def enable(self):
        assert (
            not self.enabled
        ), f'Enabling a layer type twice is not supported: Would permit complicated nesting'
        self.enabled = True
        # First, check it is the expected type with __getattribute__.
        assert (
            self.library_pointer.__getattribute__(self.library_submodule_name)
            == self.base_class
        )
        # Must call __setattr__ here to overwrite the pointer within the
        # library.
        self.library_pointer.__setattr__(
            self.library_submodule_name, self.shim_class
        )

    def disable(self):
        assert (
            self.enabled
        ), f'Disabling a layer type that is not enabled is unsupported: {self.library_pointer} -> {self.library_submodule_name}'
        self.enabled = False
        # First, check it is the expected type with __getattribute__.
        assert (
            self.library_pointer.__getattribute__(self.library_submodule_name)
            == self.shim_class
        )
        # Must call __setattr__ here to overwrite the pointer within the
        # library.
        self.library_pointer.__setattr__(
            self.library_submodule_name, self.base_class
        )


class PerExampleGradNormShim(contextlib.AbstractContextManager):

    def __init__(self, enable_per_example_grad_norms: list):
        self.enable_per_example_grad_norms = enable_per_example_grad_norms

        SHIM_LAYER_EMBED = 'torch.nn.Embedding'
        SHIM_LAYER_LAYERNORM = 'torch.nn.LayerNorm'
        SHIM_LAYER_LINEAR = 'torch.nn.Linear'

        self.SHIM_STATE = {
            SHIM_LAYER_EMBED: ShimLayerState(
                torch.nn, 'Embedding', PEGradNormShimEmbedding
            ),
            SHIM_LAYER_LAYERNORM: ShimLayerState(
                torch.nn, 'LayerNorm', PEGradNormShimLayerNorm
            ),
            SHIM_LAYER_LINEAR: ShimLayerState(
                torch.nn, 'Linear', PEGradNormShimLinear
            ),
        }

    def __enter__(self):
        # Enable all layers in desired set
        for layer_name in self.enable_per_example_grad_norms:
            if layer_name not in self.SHIM_STATE.keys():
                raise NotImplementedError(
                    f'Per example grad norm logging is '
                    f'not supported for layer type: {layer_name}'
                )
            layer_state = self.SHIM_STATE[layer_name]
            layer_state.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Disable all layers that were enabled on entrance
        for layer_name in self.enable_per_example_grad_norms:
            if layer_name not in self.SHIM_STATE.keys():
                raise NotImplementedError(
                    f'Per example grad norm logging is '
                    f'not supported for layer type: {layer_name}'
                )
            layer_state = self.SHIM_STATE[layer_name]
            layer_state.disable()

    def get_state(self):
        return self.SHIM_STATE

    def get_state_str(self):
        state_str = '{\n'
        for layer_name in self.SHIM_STATE.keys():
            layer_state = self.SHIM_STATE[layer_name]
            state_str += f'\t{layer_name}: {layer_state.enabled}\n'
        state_str += "}"
        return state_str
