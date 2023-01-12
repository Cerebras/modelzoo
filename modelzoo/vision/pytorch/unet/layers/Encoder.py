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

import torch.nn as nn
from torch.nn import Dropout, MaxPool2d, MaxPool3d

from modelzoo.vision.pytorch.unet.layers.UNetBlock import UNetBlock


class Encoder(nn.Module):
    """
    PyTorch port of the Encoder TF reference.
    https://github.com/Cerebras/monolith/blob/master/src/models/vision/tf/unet/layers/Encoder.py

    Args:
        in_channels (int): The input channel dimension before passing through the encoder.
        encoder_filters ([int]): List of filter sizes for each block in the encoder.
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
        use_conv3d (bool): 3D convolutions will be used when set to True
        downscale_first_conv (bool): If True, the first convolution operation in each UNetBlock will
            be downscaled. If False, the last convolution in each UNetBlock will be downscaled
        downscale_encoder_blocks (bool or [bool]): bool or list of bools that determine whether each
            block in the Encoder includes downsampling. Length of the list must correspond to the
            number of UNetBlocks in the Encoder. If a single bool is provided, all blocks will use 
            this value.
    """

    def __init__(
        self,
        in_channels,
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
        downscale_first_conv=False,
        downscale_encoder_blocks=True,
    ):
        super(Encoder, self).__init__()
        self.skip_connect = skip_connect
        dropout_layers = []
        pooling_layers = []
        unet_blocks = []
        if isinstance(downscale_encoder_blocks, bool):
            downscale_encoder_blocks = [downscale_encoder_blocks] * len(
                encoder_filters
            )
        for block_idx in range(len(encoder_filters) - 1):
            unet_blocks.append(
                UNetBlock(
                    in_channels=encoder_filters[block_idx - 1]
                    if block_idx
                    else in_channels,
                    out_channels=encoder_filters[block_idx],
                    encoder=True,
                    convs_per_block=convs_per_block,
                    skip_connect=skip_connect,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    downscale_method=downscale_method,
                    bias=bias,
                    residual_blocks=residual_blocks,
                    act=act,
                    use_conv3d=use_conv3d,
                    downscale_first_conv=downscale_first_conv,
                    downscale=downscale_encoder_blocks[block_idx],
                )
            )

            if downscale_method == "max_pool":
                if use_conv3d:
                    max_pool = MaxPool3d(kernel_size=2, stride=2)
                else:
                    max_pool = MaxPool2d(kernel_size=2, stride=2)
                pooling_layers.append(max_pool)

            if dropout_rate:
                dropout_layers.append(Dropout(p=dropout_rate))

        self.unet_blocks = nn.ModuleList(unet_blocks)
        self.pooling_layers = nn.ModuleList(pooling_layers)
        self.dropout_layers = nn.ModuleList(dropout_layers)

    def forward(self, inputs):
        skip_connections = []
        outputs = inputs
        for unet_block, pooling_layer, dropout_layer in zip_longest(
            self.unet_blocks, self.pooling_layers, self.dropout_layers
        ):
            outputs, skip_connection = unet_block(outputs)
            if self.skip_connect:
                skip_connections.append(skip_connection)
            if pooling_layer:
                outputs = pooling_layer(outputs)
            if dropout_layer:
                outputs = dropout_layer(outputs)
        return outputs, skip_connections
