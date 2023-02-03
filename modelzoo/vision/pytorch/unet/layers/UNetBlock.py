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

import warnings

import torch.nn as nn

from modelzoo.vision.pytorch.layers.ConvNormActBlock import ConvNormActBlock


class UNetBlock(nn.Module):
    """

    Args:
        in_channels (int): The input channel dimension before passing through the block.
        out_channels (int): The output channel dimension after passing through the block.
        encoder (bool): Flag if the block is part of the encoder section of the model.
            Returns the block-output if `True`, and a tuple of block-output, block-output
            before downsampling if `False`.
        convs_per_block ([str]): List of conv specifications for each conv in the block.
        bias (bool): Flag to use bias vectors.
        norm_layer (nn.Module): Desired normalization layer.
        norm_kwargs (dict): A dictionary of the arguments to pass to the
            constructor of the normalization layer.
        act (str): Activation to use.
        skip_connect (bool): Flag for if the model concatenates encoder outputs to decoder inputs.
        residual_blocks (bool): Flag for using residual connections.
        downscale_method (str): Downscaling method at the end of the block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        encoder,
        convs_per_block,
        bias,
        norm_layer,
        norm_kwargs,
        act="relu",
        skip_connect=True,
        residual_blocks=False,
        downscale_method="max_pool",
        use_conv3d=False,
        downscale_first_conv=False,
        downscale=True,
    ):
        super(UNetBlock, self).__init__()

        self.encoder = encoder
        self.downscale_method = downscale_method
        self.residual_blocks = residual_blocks
        self.skip_connect = skip_connect
        if downscale_first_conv:
            self.downscale_conv_idx = 0
        else:
            self.downscale_conv_idx = len(convs_per_block) - 1
            if (self.downscale_conv_idx == 0) and downscale:
                warnings.warn(
                    "Downscaling first conv in block because `convs_per_block` = 1"
                )

        layers = []
        for conv_idx, conv_type in enumerate(convs_per_block):
            stride = (
                2
                if (
                    downscale
                    and downscale_method == "strided_conv"
                    and conv_idx == self.downscale_conv_idx
                )
                else 1
            )

            conv_in_chs = conv_out_chs if conv_idx else in_channels
            conv_out_chs = out_channels

            if conv_type == "3x3_conv":
                groups = 1
                kernel_size = 3
            elif conv_type == "3x3_dw_conv":
                groups = conv_in_chs
                kernel_size = 3
                conv_out_chs = conv_in_chs
            elif conv_type == "1x1_conv":
                groups = 1
                kernel_size = 1
            else:
                raise ValueError(f"Unsupported convolution type: {conv_type}")

            layers.append(
                ConvNormActBlock(
                    in_channels=conv_in_chs,
                    out_channels=conv_out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    groups=groups,
                    padding="same",
                    bias=bias,
                    act=act,
                    use_conv3d=use_conv3d,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = inputs
        skip_connection = None
        for layer_id, layer in enumerate(self.layers):
            outputs = layer(outputs)
            if self.residual_blocks:
                if layer_id == 0:
                    block_input = outputs
                elif layer_id == len(self.layers) - 1:
                    outputs = outputs + block_input
            if (
                self.downscale_method == "strided_conv"
                and layer_id == len(self.layers) - 2
                and self.skip_connect
            ):
                skip_connection = outputs

        if self.encoder:
            if self.downscale_method == "strided_conv":
                return outputs, skip_connection
            elif self.downscale_method == "max_pool":
                return outputs, outputs

        return outputs
