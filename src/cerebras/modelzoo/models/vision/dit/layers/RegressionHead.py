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

from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.utils import unpatchify_helper


class RegressionHead(nn.Module):
    def __init__(
        self,
        image_size=[224, 224],
        out_channels=4,
        patch_size=[16, 16],
        hidden_size=384,
        use_conv_transpose_unpatchify=False,
        kernel_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
    ):
        super(RegressionHead, self).__init__()
        self.image_size = image_size
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_conv_transpose_unpatchify = use_conv_transpose_unpatchify
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        if use_conv_transpose_unpatchify:
            # combine linear + unpatchify into a single conv transpose 2d
            self.conv_transpose = nn.ConvTranspose2d(
                hidden_size,
                out_channels,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        else:
            self.linear = nn.Linear(
                hidden_size, patch_size[0] * patch_size[1] * out_channels
            )

        # Initialize weights and bias
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        bias_initializer = create_initializer(self.bias_initializer)
        kernel_initializer = create_initializer(self.kernel_initializer)

        if self.use_conv_transpose_unpatchify:
            bias_initializer(self.conv_transpose.bias.data)
            kernel_initializer(self.conv_transpose.weight.data)
        else:
            bias_initializer(self.linear.bias.data)
            kernel_initializer(self.linear.weight.data)

    def forward(self, inputs):
        # inputs: (B, (H * W) / (P * P), D)
        if self.use_conv_transpose_unpatchify:
            # reverse the conv patchified embedding
            batch_size = inputs.shape[0]
            outputs = inputs.transpose(1, 2)  # (B, D, (H * W) / (P * P))
            outputs = outputs.reshape(
                batch_size,
                self.hidden_size,
                self.image_size[0] // self.patch_size[0],
                self.image_size[1] // self.patch_size[1],
            )  # (B, D, H / P, W / P)
            return self.conv_transpose(outputs)  # (B, C, H, W)
        else:
            outputs = self.linear(inputs)  # (B, (H * W) / (P * P), P * P * C)
            return unpatchify_helper(
                outputs, (self.out_channels, *self.image_size), self.patch_size
            )  # (B, C, H, W)
