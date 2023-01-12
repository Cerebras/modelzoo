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

# coding=utf-8
#
# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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


import torch
from torch.optim import Optimizer

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.utils import to_tensor


class Adafactor(Optimizer):
    """
    Adafactor optimizer implemented to conform to execution within the
    constraints of the Cerebras WSE.
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError(
                "Cannot combine manual `lr` and `relative_step=True` options"
            )
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` is not supported yet")
        if clip_threshold != 1.0:
            raise ValueError(
                f"Only `clip_threshold=1.0` is supported now. "
                f"It was set to {clip_threshold}."
            )
        if beta1 is not None:
            raise ValueError(
                f"Only `beta1=None` is supported now. It was set to {beta1}."
            )
        if relative_step:
            raise ValueError("`relative_step=True` is not supported yet")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

        if not cm.use_cs():
            self.preinitialize()

    def add_global_step(self, global_step):
        """
        Stores a `global_step` tensor which will be used in computation and
        shared by all params.
        """
        self.global_step = global_step

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                grad_shape = p.shape

                factored, use_first_moment = self._get_options(
                    group, grad_shape
                )
                if use_first_moment:
                    state["exp_avg"] = torch.zeros_like(p, device="cpu").to(
                        p.device
                    )
                if factored:
                    state["exp_avg_sq_row"] = torch.zeros(
                        grad_shape[:-1], device="cpu"
                    ).to(p.device)
                    state["exp_avg_sq_col"] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:], device="cpu"
                    ).to(p.device)
                else:
                    state["exp_avg_sq"] = torch.zeros_like(p, device="cpu").to(
                        p.device
                    )
                if not hasattr(self, "global_step"):
                    state["step"] = torch.tensor(0).to(p.device)

    @staticmethod
    def _get_lr(param_group, rms):
        rel_step_sz = param_group["lr"]
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = torch.maximum(rms, to_tensor(param_group["eps"][1]))
        return param_scale * rel_step_sz

    @staticmethod
    def _rms(tensor):
        return tensor.square().mean().sqrt()

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates
            the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients."
                    )

                state = self.state[p]

                factored = "exp_avg_sq_row" in state
                use_first_moment = "exp_avg" in state

                p_data = p.data

                if hasattr(self, "global_step"):
                    global_step_fp32 = self.global_step.add(1).float()
                else:
                    state["step"] += 1
                    global_step_fp32 = state["step"].float()

                lr = self._get_lr(group, self._rms(p_data))

                beta2t = 1.0 - torch.pow(
                    global_step_fp32, to_tensor(group["decay_rate"]).item()
                )
                update = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1).mul(1.0 - beta2t)
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2).mul(1.0 - beta2t)
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col
                    )
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update.mul(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    torch.maximum(
                        self._rms(update) / group["clip_threshold"],
                        torch.tensor(1.0, dtype=torch.float32, device=p.device),
                    )
                )
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(
                        update.mul(1 - group["beta1"])
                    )
                    update = exp_avg

                if group["weight_decay"] > 0.0:
                    p_data.sub_(p_data.mul(group["weight_decay"] * lr))

                p_data.sub_(update)

        return loss
