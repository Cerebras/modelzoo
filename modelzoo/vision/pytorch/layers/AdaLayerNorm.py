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

import torch.nn as nn


class AdaLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(AdaLayerNorm, self).__init__()

        self.layernorm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=False,
            **factory_kwargs,
        )
        self.scale_linear = nn.Sequential(
            nn.SiLU(), nn.Linear(normalized_shape, normalized_shape, bias=True)
        )
        self.shift_linear = nn.Sequential(
            nn.SiLU(), nn.Linear(normalized_shape, normalized_shape, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            param.data.zero_()

    def forward(self, input, context):
        shift = self.shift_linear(context)
        scale = self.scale_linear(context)
        output = (1 + scale.unsqueeze(1)) * self.layernorm(
            input
        ) + shift.unsqueeze(1)
        return output
