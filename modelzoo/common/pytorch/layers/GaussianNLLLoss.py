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

import torch
import torch.nn as nn

from modelzoo.common.pytorch.layers.utils import apply_loss_reduction


# Adapted from https://github.com/pytorch/pytorch/blob/b136f3f310aa01a8b3c1e63dc0bfda8fd2234b06/torch/nn/functional.py#L2765
class GaussianNLLLoss(nn.Module):
    def __init__(self, full=False, eps=1e-6, reduction='mean'):
        super(GaussianNLLLoss, self).__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target, var):
        # Check var size
        # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
        # Otherwise:
        if var.size() != input.size():

            # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. input.size = (10, 2, 3), var.size = (10, 2)
            # -> unsqueeze var so that var.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if input.size()[:-1] == var.size():
                var = torch.unsqueeze(var, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
            # This is also a homoscedastic case.
            # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
            elif (
                input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1
            ):  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of var is incorrect.
            else:
                raise ValueError(f"`var` is of incorrect size: {var.size()}")

        # Entries of var must be non-negative
        # if torch.any(var < 0):
        #    raise ValueError("var has negative entry/entries")

        # Clamp for stability
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=self.eps)

        # Calculate the loss
        loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)
        if self.full:
            loss += 0.5 * math.log(2 * math.pi)

        return apply_loss_reduction(loss, self.reduction)
