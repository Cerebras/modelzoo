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

"""Compute the norm of the parameters and gradients of the model."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.tools.gns.per_example_grad_norm_utils import (
    PerExampleGradNormShim,
)
from cerebras.modelzoo.trainer.callbacks import Callback, ModelCallback
from cerebras.modelzoo.trainer.callbacks.summaries import summarize_scalar

if TYPE_CHECKING:
    from ..trainer import Trainer


def gather_marked_buffers(root_module: nn.Module, marker: str) -> List[str]:
    """
    Gather buffer names that share the same marker across all modules that have marked_buffers.

    Args:
        root_module: The root module to search through
        marker: The marker to search for

    Returns:
        List of fully qualified buffer names that share the marker
    """
    buffer_names = []

    # Go through all modules in the hierarchy
    for module_path, module in root_module.named_modules():
        # Check if this module has marked_buffers
        if hasattr(module, 'marked_buffers'):
            # For each buffer in this module's marked_buffers
            for buffer_name, buffer_marker in module.marked_buffers.items():
                if buffer_marker == marker:
                    # Construct full path to buffer
                    full_path = (
                        f"{module_path}.{buffer_name}"
                        if module_path
                        else buffer_name
                    )
                    buffer_names.append(full_path)

    return buffer_names


def get_buffers_by_marker(root_module: nn.Module, marker: str):
    """
    Get all buffer tensors that share the same marker across all modules.

    Args:
        root_module: The root module to search through
        marker: The marker to search for

    Returns:
        List of (buffer name, buffer tensor) tuples that share the given marker
    """
    buffers = []
    buffer_names = gather_marked_buffers(root_module, marker)

    # Get the actual buffer tensors using get_buffer
    for buffer_name in buffer_names:
        buffer = root_module.get_buffer(buffer_name)
        buffers.append((buffer_name, buffer))

    return buffers


def get_upegsqnorm_buffers(root_module: nn.Module):
    """
    Get all unnormalized per-example gradient squared norm buffers.

    Args:
        root_module: The root module to search through
    Returns:
        List of (buffer name, buffer tensor) tuples that are unnormalized per-example gradient squared norms
    """
    return get_buffers_by_marker(root_module, "is_pegsqnorm")


class ComputeNormGNS(Callback):
    """
    Callback class that computes the model wise and per layer norm of the parameters.
    """

    def __init__(self, per_example_grad_norms: Optional[List[str]]):
        self.per_example_grad_norms = per_example_grad_norms

    def apply_shim(self, model_fn):
        def inner():
            with PerExampleGradNormShim(self.per_example_grad_norms):
                return model_fn()

        return inner

    def pre_setup(self, trainer):
        self.model_cb = trainer.get_callback(ModelCallback)
        # "model_cb.model" is the function that when executed constructs the actual model (e.g., Gpt2Model).
        # That call is made in `Trainer.setup()`. So by applying the shim layer in this callback's pre_setup()
        # method, we can create a new callable that applies the shim.
        if type(self.model_cb.model) == torch.nn.Module:
            raise ValueError(
                "Cannot apply shim layer to an already constructed model."
            )
        self.model_cb.model = self.apply_shim(self.model_cb.model)

    def compute_param_norm(  # pylint: disable=no-self-use
        self, trainer: Trainer, model: torch.nn.Module
    ):
        """Compute the model wise norm of the parameters."""
        param_norm = torch.tensor(0.0).to(model.device)
        for _, param in model.named_parameters():
            if param.requires_grad:
                # simply add if we want to include all params
                param_norm += torch.pow(torch.norm(param), 2.0)

        trainer.log_metrics(model_wise_params_norm=torch.sqrt(param_norm))

    def on_before_backward(self, trainer, model, outputs):
        with cstorch.name_scope("params_norm"):
            self.compute_param_norm(trainer, model)

    def compute_grad_norm(  # pylint: disable=no-self-use
        self, trainer: Trainer, model: torch.nn.Module, use_prod=False
    ):
        """Compute the model wise and per layer norm of the gradients."""
        params_grad_norm = torch.tensor(0.0).to(model.device)
        for name, buffer in get_upegsqnorm_buffers(model):
            if "bias" not in name:
                if use_prod:
                    pegsqnorm = torch.prod(buffer)
                else:
                    pegsqnorm = torch.exp(torch.sum(torch.log(buffer)))
                name = name.replace("_upegsqnorm", "")
                summarize_scalar("pegsqnorm/" + name, pegsqnorm)
                buffer.zero_()  # clear the buffer
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grad_norm_sq = torch.sum(torch.pow(param.grad, 2.0))
                summarize_scalar("gsqnorm/" + name, param_grad_norm_sq)
                params_grad_norm += param_grad_norm_sq
        params_grad_norm = torch.sqrt(params_grad_norm)

        trainer.log_metrics(model_wise_grad_norm=params_grad_norm)

        per_layer_grad_norm = {}
        layer_pattern = re.compile(r".*(layers\.)(\d+)(\.).*")
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # get a match if module name contains `layers.i.0` where i is layer num
            match = layer_pattern.match(name)
            if match:
                layer_id = match.group(2)
                if layer_id not in per_layer_grad_norm:
                    per_layer_grad_norm[layer_id] = torch.tensor(0.0).to(
                        model.device
                    )
                per_layer_grad_norm[layer_id] += torch.pow(
                    torch.norm(param.grad), 2.0
                )

        trainer.log_metrics(
            **{
                f"per_layer_grad_norm/layer_{layer_id}": torch.sqrt(
                    per_layer_grad_norm[layer_id]
                )
                for layer_id in per_layer_grad_norm
            }
        )

    def on_before_optimizer_step(self, trainer, model, optimizer):
        with cstorch.name_scope("grad_norm"):
            self.compute_grad_norm(trainer, model)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class ComputeNorm(Callback):
    """
    Callback class that computes the model wise and per layer norm of the parameters.
    """

    def compute_param_norm(  # pylint: disable=no-self-use
        self, trainer: Trainer, model: torch.nn.Module
    ):
        """Compute the model wise norm of the parameters."""
        param_norm = torch.tensor(0.0).to(model.device)
        for _, param in model.named_parameters():
            if param.requires_grad:
                # simply add if we want to include all params
                param_norm += torch.pow(torch.norm(param), 2.0)

        trainer.log_metrics(model_wise_params_norm=torch.sqrt(param_norm))

    def on_before_backward(self, trainer, model, outputs):
        with cstorch.name_scope("params_norm"):
            self.compute_param_norm(trainer, model)

    def compute_grad_norm(  # pylint: disable=no-self-use
        self, trainer: Trainer, model: torch.nn.Module
    ):
        """Compute the model wise and per layer norm of the gradients."""
        params_grad_norm = torch.tensor(0.0).to(model.device)
        for _, param in model.named_parameters():
            if param.grad is not None:
                params_grad_norm += torch.pow(torch.norm(param.grad), 2.0)
        params_grad_norm = torch.sqrt(params_grad_norm)

        trainer.log_metrics(model_wise_grad_norm=params_grad_norm)

        per_layer_grad_norm = {}
        layer_pattern = re.compile(r".*(layers\.)(\d+)(\.).*")
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # get a match if module name contains `layers.i.0` where i is layer num
            match = layer_pattern.match(name)
            if match:
                layer_id = match.group(2)
                if layer_id not in per_layer_grad_norm:
                    per_layer_grad_norm[layer_id] = torch.tensor(0.0).to(
                        model.device
                    )
                per_layer_grad_norm[layer_id] += torch.pow(
                    torch.norm(param.grad), 2.0
                )

        trainer.log_metrics(
            **{
                f"per_layer_grad_norm/layer_{layer_id}": torch.sqrt(
                    per_layer_grad_norm[layer_id]
                )
                for layer_id in per_layer_grad_norm
            }
        )

    def on_before_optimizer_step(self, trainer, model, optimizer):
        with cstorch.name_scope("grad_norm"):
            self.compute_grad_norm(trainer, model)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
