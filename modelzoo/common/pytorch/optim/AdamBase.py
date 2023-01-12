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

import math
import sys
from typing import Iterable, Tuple

import torch
from torch import nn

from modelzoo.common.pytorch.optim.CSOptimizer import CSOptimizer


class AdamBase(CSOptimizer):
    """
    AdamW optimizer implemented to conform to execution within the constraints
    of the Cerebras WSE, including pre-initilizing optimizer state and
    performing a gradual reduction of bias correction using exponential decay
    of `beta1_power` and `beta2_power` rather than recomputing `beta1^step` each
    step.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        l2_regularization_rate: float = 0.0,
        correct_bias: bool = True,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            l2_regularization_rate=l2_regularization_rate,
            correct_bias=correct_bias,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                # State initialization

                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, device="cpu"
                    ).to(p.device)

                if group["correct_bias"]:  # No bias correction for Bert
                    beta1, beta2 = group["betas"]

                    # beta1 ^ step, initialized for used on step 1
                    state["beta1_power"] = torch.tensor(beta1).to(p.device)
                    state["beta2_power"] = torch.tensor(beta2).to(p.device)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # This is equivalent to Algorithm 2 i.e Adam with L2 regularization
                # (https://arxiv.org/pdf/1711.05101.pdf)
                if group["l2_regularization_rate"] > 0.0:
                    grad = grad.add(
                        p.data, alpha=group["l2_regularization_rate"]
                    )

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(
                        max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq
                    )
                    state["max_exp_avg_sq"] = max_exp_avg_sq
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                update = exp_avg / denom

                if group["correct_bias"]:  # No bias correction for Bert.
                    one = torch.tensor(
                        1.0, dtype=torch.float32, device=p.device
                    )
                    bias_correction1 = one - state["beta1_power"]
                    bias_correction2 = one - state["beta2_power"]
                    step_size = torch.sqrt(bias_correction2) / bias_correction1
                    update *= step_size
                    # Update `beta1^step` for the next step.
                    state["beta1_power"] *= beta1
                    state["beta2_power"] *= beta2

                # Applying weight decay here is equivalent to Algorithm 2
                # (https://arxiv.org/pdf/1711.05101.pdf)
                # Decoupled Weight Decay regularization i.e AdamW
                if group["weight_decay"] > 0.0:
                    update.add_(p.data, alpha=group["weight_decay"])

                # Scale the update by the learning rate.
                update *= group["lr"]

                # Finally, update the weight data.
                p.data.sub_(update)

        return loss

    def convert_state_dict_for_checkpoint(self, state_dict):
        """
        Converts the state_dict for compatibility with AdamW from
        huggingface_common, which is the optimizer used by PyTorchBaseModel
        when not run on WSE and is otherwise API compatible.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        Returns the modified state_dict.
        """
        # huggingface AdamW and PyTorch Adam stores a `step`
        # Cerebras (this) AdamW/Adam stores `beta1^step` as `beta1_power`.

        for param, state in state_dict["state"].items():
            if "beta1_power" not in state:
                # huggingface AdamW increments `step` always, but doesn't use
                # it if isn't performing bias correction. We don't have the
                # step available, so save a dummy value.
                state["step"] = 0
                continue
            for param_groups in state_dict["param_groups"]:
                if param in param_groups["params"]:
                    beta1, beta2 = param_groups["betas"]
                    # beta1_power = beta1**step
                    # so step = log(beta1_power, beta1)
                    if state["beta1_power"]:
                        # With default beta1=0.9, this should be finite (and
                        # have the most resolution) until ~6700 steps.
                        beta1_power = state["beta1_power"]
                        state["step"] = int(math.log(beta1_power, beta1))
                    else:
                        # if beta1_power is 0, it likely underflowed. Check
                        # beta2_power. Otherwise, use DOUBLE_MIN
                        # With default beta2=0.999, this should be finite
                        # until ~700k step.
                        beta2_power = state["beta2_power"] or sys.float_info.min
                        state["step"] = int(math.log(beta2_power, beta2))
                    break

        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        This overrides torch.optim.Optimizer to add checkpoint compatibility
        with the AdamW from huggingface_common, which is otherwise API
        compatible.
        """

        # huggingface AdamW and PyTorch Adam stores a `step`
        # Cerebras (this) AdamW/Adam stores `beta1^step` as `beta1_power`.
        for param, state in state_dict["state"].items():
            if "step" in state and "beta1_power" not in state:
                step = state.pop("step")

                # go find betas for this parameter
                correct_bias = False
                beta1 = None
                beta2 = None
                for param_group in state_dict["param_groups"]:
                    if param in param_group["params"]:
                        correct_bias = param_group["correct_bias"]
                        beta1, beta2 = param_group["betas"]
                        break
                if correct_bias:
                    state["beta1_power"] = torch.tensor(
                        beta1 ** step, dtype=torch.float32
                    )
                    state["beta2_power"] = torch.tensor(
                        beta2 ** step, dtype=torch.float32
                    )

        super().load_state_dict(state_dict)


class AdamW(AdamBase):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        amsgrad: bool = False,
    ):
        super(AdamW, self).__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            l2_regularization_rate=0.0,
            correct_bias=correct_bias,
            amsgrad=amsgrad,
        )

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        Adds checkpoint compatibility with the AdamW from HuggingFace
        """
        for group in state_dict["param_groups"]:
            group["l2_regularization_rate"] = 0.0

        super().load_state_dict(state_dict)


class Adam(AdamBase):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        # This init uses `weight_decay` to be in sync with PyTorch API
        super(Adam, self).__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            l2_regularization_rate=weight_decay,
            correct_bias=True,
            amsgrad=amsgrad,
        )
        for group in self.param_groups:
            group["l2_regularization_rate"] = group.pop("weight_decay", 0.0)
            group["weight_decay"] = 0.0

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        Adds checkpoint compatibility with the Adam from PyTorch
        """
        for group in state_dict["param_groups"]:
            group["l2_regularization_rate"] = group.pop("weight_decay", 0.0)
            group["weight_decay"] = 0.0
            group["correct_bias"] = True
        super().load_state_dict(state_dict)
