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


def apply_mask(input_tensor, masks):
    """
    input_tensor: torch.Tensor of shape (bsz, num_patches, H)
    masks: torch.Tensor of shape (bsz, num_masks, num_masked_patches)
    Returns a tensor of shape (bsz, num_masks, num_masked_patches, H)
    """

    bsz, n_masks, n_masked_patches = masks.shape
    masks = masks.reshape(bsz, n_masks * n_masked_patches, -1)
    masks = masks.repeat((1, 1, input_tensor.shape[2]))

    input_tensor = torch.gather(input_tensor, dim=1, index=masks)
    input_tensor = input_tensor.reshape(bsz, n_masks, n_masked_patches, -1)
    return input_tensor
