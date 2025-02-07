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

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

LOSS_SCOPE = "loss"


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )


def reset_norm(module):
    "Resets a norm layer by zeroing out the bias and replacing weight with 1s."
    if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
        module.bias.data.zero_()
    if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
        module.weight.data.fill_(1.0)


def patchify_helper(input_image, patch_size):
    batch_size, num_channels, height, width = input_image.shape
    num_patches = [
        (height // patch_size[0]),
        (width // patch_size[1]),
    ]

    assert (
        height % patch_size[0] == 0 and width % patch_size[1] == 0
    ), f"image size {height, width} is not divisible by patch_size {patch_size}"

    sequence_length = num_patches[0] * num_patches[1]
    patchified_image = input_image.reshape(
        batch_size,
        num_channels,
        num_patches[0],
        patch_size[0],
        num_patches[1],
        patch_size[1],
    )
    patchified_image = patchified_image.permute(
        0, 2, 4, 3, 5, 1
    )  # output shape = [bs,
    # num_patches_vertical, num_patches_horizontal,
    # patch_size_vertical, patch_size_horizontal,
    # num_channels]
    patchified_image = patchified_image.reshape(batch_size, sequence_length, -1)

    return patchified_image


def unpatchify_helper(inputs, image_size, patch_size):
    # reverses the patchify process: (B x S x D) -> (B x C x H x W)
    # image_size: (C, H, W)
    # patch_size: (patch_height, patch_width)
    bsz = inputs.shape[0]
    num_channels, image_height, image_width = image_size
    patch_height, patch_width = patch_size
    num_patches = (image_height // patch_height, image_width // patch_width)

    outputs = inputs.reshape(
        bsz,  # bsz
        image_height // patch_height,
        image_width // patch_width,
        patch_height,
        patch_width,
        num_channels,
    )
    outputs = torch.einsum('nhwpqc->nchpwq', outputs)
    outputs = outputs.reshape(
        bsz,
        num_channels,
        image_height,
        image_width,
    )
    return outputs


def get_2d_fixed_position_embeddings(
    num_patches, hidden_size, add_cls_token=False
):
    position_ids_width = (
        torch.arange(0, num_patches[0]).expand(num_patches[1], -1).permute(1, 0)
    )
    position_ids_height = torch.arange(0, num_patches[1]).expand(
        (num_patches[0], -1)
    )

    # divide between width/height
    freq_embedding = hidden_size // 2
    # divide between sin/cos
    assert freq_embedding % 2 == 0, "freq_embedding must be even"

    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, freq_embedding // 2) / (freq_embedding / 2.0))
    )

    out_width = torch.einsum(
        "m,d->md", position_ids_width.reshape(-1), inv_freq
    )
    out_height = torch.einsum(
        "m,d->md", position_ids_height.reshape(-1), inv_freq
    )

    pe_width = torch.cat(
        [
            torch.sin(out_width),
            torch.cos(out_width),
        ],
        dim=1,
    )
    pe_height = torch.cat(
        [
            torch.sin(out_height),
            torch.cos(out_height),
        ],
        dim=1,
    )
    pe_seq = torch.cat(
        [
            pe_height,
            pe_width,
        ],
        dim=1,
    )

    if add_cls_token:
        pe_cls = torch.zeros(1, hidden_size)
        pe_seq = torch.cat(
            [pe_seq, pe_cls], dim=0
        )  # [seq_length+1, hidden_size]

    return pe_seq


class ModuleWrapperClass(nn.Module):
    def __init__(self, fcn, name=None, kwargs=None):
        self.fcn = fcn
        self.name = name
        self.kwargs = kwargs
        super(ModuleWrapperClass, self).__init__()

    def extra_repr(self) -> str:
        repr_str = 'fcn={}'.format(
            self.name if self.name is not None else self.fcn.__name__
        )
        if self.kwargs is not None:
            for k, val in self.kwargs.items():
                repr_str += f", {k}={val}"

        return repr_str

    def forward(self, input):
        return self.fcn(input)
