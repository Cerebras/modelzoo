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


class GroupInstanceNorm(nn.Module):
    """
    Uses torch.nn.GroupNorm to emulate InstanceNorm by setting number of groups
    equal to the number of channels.

    Args:
        num_channels (int): number of channels. `C` from an expected input of size (N, C, H, W).

    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
        super(GroupInstanceNorm, self).__init__()

        self.instancenorm = nn.GroupNorm(
            num_channels,
            num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        out = self.instancenorm(input)
        return out
