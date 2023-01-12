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

from itertools import zip_longest

import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d, ConvTranspose3d, Dropout

from modelzoo.vision.pytorch.unet.layers.UNetBlock import UNetBlock


class Decoder(nn.Module):
    """
    PyTorch port of the Decoder TF reference.
    https://github.com/Cerebras/monolith/blob/master/src/models/vision/tf/unet/layers/Decoder.py

    Args:
        in_channels (int): The input channel dimension before passing through the decoder.
        decoder_filters ([int]): List of filter sizes for each block in the decoder
        encoder_filters ([int]): List of filter sizes for each block in the encoder.
            Used to calculate correct input channel dimension when concatenating
            encoder outputs and transpose outputs.
        convs_per_block ([str]): List of conv specifications for each conv in the block.
        bias (bool): Flag to use bias vectors.
        norm_layer (nn.Module): Desired normalization layer.
        norm_kwargs (dict): A dictionary of the arguments to pass to the
            constructor of the normalization layer.
        act (str): Activation to use.
        skip_connect (bool): Flag for if the model concatenates encoder outputs to decoder inputs.
        residual_blocks (bool): Flag for using residual connections at the end of each block.
        downscale_method (str):Downscaling method at the end of each block.
        dropout_rate (float): The probability that each element is dropped.
    """

    def __init__(
        self,
        in_channels,
        decoder_filters,
        encoder_filters,
        convs_per_block,
        bias,
        norm_layer,
        norm_kwargs,
        act="relu",
        skip_connect=True,
        residual_blocks=False,
        downscale_method="max_pool",
        dropout_rate=0.0,
        use_conv3d=False,
    ):
        super(Decoder, self).__init__()
        self.skip_connect = skip_connect

        dropout_layers = []
        unet_blocks = []
        transpose_layers = []

        if use_conv3d:
            transpose_conv_op = ConvTranspose3d
        else:
            transpose_conv_op = ConvTranspose2d

        for block_idx in range(len(decoder_filters)):
            # transpose in/out ch calculations
            transpose_in_chs = (
                decoder_filters[block_idx - 1] if block_idx else in_channels
            )
            # Special case when last conv in a unet block is depthwise because the output
            # channel dim is not given by decoder_filters[i] but by the input channel dim
            # to the depthwise block. If we concatenate the encoder output with the decoder
            # input before passing through the depthwise block, then then output channel dim
            # is given by the sum the channel dims.
            if (
                block_idx
                and self.skip_connect
                and "dw_conv" in convs_per_block[-1]
            ):
                # FIXME: -2 because last filter size is the bottleneck filter
                transpose_in_chs += encoder_filters[-block_idx - 2]
            transpose_out_chs = decoder_filters[block_idx]
            transpose_layers.append(
                transpose_conv_op(
                    in_channels=transpose_in_chs,
                    out_channels=transpose_out_chs,
                    kernel_size=2,
                    stride=2,
                    bias=bias,
                )
            )

            if skip_connect and dropout_rate:
                dropout_layers.append(Dropout(p=dropout_rate))

            blk_in_chs = transpose_out_chs
            if self.skip_connect:
                # concatenation with skip connections
                blk_in_chs += encoder_filters[-block_idx - 2]
            unet_blocks.append(
                UNetBlock(
                    in_channels=blk_in_chs,
                    out_channels=decoder_filters[block_idx],
                    encoder=False,
                    convs_per_block=convs_per_block,
                    downscale_method=downscale_method,
                    skip_connect=skip_connect,
                    residual_blocks=residual_blocks,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    bias=bias,
                    act=act,
                    use_conv3d=use_conv3d,
                    downscale=False,
                )
            )

        # Updating decoder_filters[-1] because the final conv uses `decoder_filters[-1]`
        # to determine input channel dim
        if block_idx and self.skip_connect and "dw_conv" in convs_per_block[-1]:
            decoder_filters[-1] += encoder_filters[0]

        self.unet_blocks = nn.ModuleList(unet_blocks)
        self.transpose_layers = nn.ModuleList(transpose_layers)
        self.dropout_layers = nn.ModuleList(dropout_layers)

    def forward(self, inputs, skip_connections):
        outputs = inputs
        for (
            transpose_layer,
            unet_block,
            dropout_layer,
            skip_connection,
        ) in zip_longest(
            self.transpose_layers,
            self.unet_blocks,
            self.dropout_layers,
            reversed(skip_connections),
        ):
            outputs = transpose_layer(outputs)
            if self.skip_connect:
                # channels first in torch
                outputs = torch.cat([outputs, skip_connection], dim=1)
            outputs = unet_block(outputs)
            if dropout_layer:
                outputs = dropout_layer(outputs)

        return outputs
