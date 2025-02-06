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
from typing import TYPE_CHECKING

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback

if TYPE_CHECKING:
    from ..trainer import Trainer


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
