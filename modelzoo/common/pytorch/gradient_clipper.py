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

import torch

from modelzoo.common.pytorch import amp


class GradientClipper:
    def __init__(
        self, max_gradient_norm: float = 0.0, max_gradient_value: float = 0.0
    ):
        self.max_gradient = 0.0
        self.max_gradient_fn = None

        self.set_max_gradients(max_gradient_norm, max_gradient_value)

    def set_max_gradients(
        self, max_gradient_norm: float = 0.0, max_gradient_value: float = 0.0
    ):
        if max_gradient_norm is None or max_gradient_norm < 0.0:
            raise ValueError(
                f"max_gradient_norm cannot be none or negative. Got "
                f"{max_gradient_norm}"
            )
        if max_gradient_value is None or max_gradient_value < 0.0:
            raise ValueError(
                f"max_gradient_value cannot be none or negative. Got "
                f"{max_gradient_value}"
            )
        if max_gradient_norm > 0.0 and max_gradient_value > 0.0:
            raise ValueError(
                f"Gradients can be clipped by norm(={max_gradient_norm}) or by "
                f"value(={max_gradient_value}), but not both. "
                f"Do not set both `max_gradient_norm` and `max_gradient_value`."
            )
        elif max_gradient_norm > 0.0:
            self.max_gradient = max_gradient_norm
            self.max_gradient_fn = torch.nn.utils.clip_grad_norm_
        elif max_gradient_value > 0.0:
            self.max_gradient = max_gradient_value
            self.max_gradient_fn = torch.nn.utils.clip_grad_value_

        self.check_amp()

    def check_amp(self):
        """Disable GGC here if GGC + DLS is enabled by the GradScaler"""
        if (
            self.max_gradient > 0
            and amp.get_init_params().get("max_gradient_norm", None) is not None
        ):
            assert self.max_gradient_fn == torch.nn.utils.clip_grad_norm_
            self.max_gradient_fn = None

    def clip(self, params: dict):
        if self.max_gradient_fn is not None:
            self.max_gradient_fn(params, self.max_gradient)

    def __call__(self, *args, **kwargs):
        return self.clip(*args, **kwargs)
