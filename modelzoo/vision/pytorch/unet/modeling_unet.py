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
import torch.nn as nn

from modelzoo.common.pytorch.model_utils.create_initializer import (
    create_initializer,
)
from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.vision.pytorch.layers.ConvNormActBlock import ConvNormActBlock
from modelzoo.vision.pytorch.unet.layers.Decoder import Decoder
from modelzoo.vision.pytorch.unet.layers.Encoder import Encoder
from modelzoo.vision.pytorch.unet.layers.UNetBlock import UNetBlock


class UNet(nn.Module):
    """
    UNet Models
    """

    def bce_loss(self, outputs, labels):
        neg_outputs = -1 * outputs
        zero_const = torch.tensor(
            0.0, dtype=outputs.dtype, device=outputs.device
        )
        max_val = torch.where(neg_outputs > zero_const, neg_outputs, zero_const)
        loss = (
            (1 - labels)
            .mul(outputs)
            .add(max_val)
            .add((-max_val).exp().add((neg_outputs - max_val).exp()).log())
        )
        mean = torch.mean(loss)
        # The return needs to be a dtype of FP16 for WS
        return mean.to(half_dtype_instance.half_dtype)

    def __init__(self, model_params):
        super(UNet, self).__init__()
        self.num_classes = model_params["num_classes"]
        self.skip_connect = model_params["skip_connect"]
        self.downscale_method = model_params["downscale_method"]
        self.residual_blocks = model_params["residual_blocks"]
        self.use_conv3d = model_params["use_conv3d"]
        self.downscale_first_conv = model_params["downscale_first_conv"]
        self.downscale_encoder_blocks = model_params["downscale_encoder_blocks"]
        self.downscale_bottleneck = model_params["downscale_bottleneck"]

        self.loss_type = model_params.get("loss", "bce")
        assert (
            self.num_classes == 2
        ), "BCE loss may only be used when there are two classes!"
        self.num_output_channels = 1
        if "bce" in self.loss_type:
            self.loss_fn = self.bce_loss

        if self.residual_blocks:
            assert self.downscale_method == "max_pool"

        self.norm_layer = model_params.get("norm_layer", None)
        self.norm_kwargs = model_params.get("norm_kwargs", None)
        self.dropout_rate = model_params.get("dropout_rate", 0.0)

        self.enable_bias = model_params.get(
            "enable_bias", self.norm_layer == None
        )
        self.act = model_params["nonlinearity"].lower()

        self.encoder_filters = model_params["encoder_filters"]
        self.decoder_filters = model_params["decoder_filters"]

        self.input_image_channels = model_params.get("input_channels")
        self.initial_conv_filters = model_params.get("initial_conv_filters")
        self.convs_per_block = model_params.get(
            "convs_per_block", ["3x3_conv", "3x3_conv"]
        )

        assert (
            len(self.encoder_filters) == len(self.decoder_filters) + 1
        ), "Number of encoder filters should be equal to number of decoder filters + 1 (bottleneck)"

        # initializers
        self.initializer = create_initializer(model_params["initializer"])
        self.bias_initializer = create_initializer(
            model_params["bias_initializer"]
        )
        self.norm_weight_initializer = create_initializer(
            model_params.get("norm_weight_initializer", {"name": "ones"})
        )

        self.initial_conv = None
        if self.initial_conv_filters:
            self.initial_conv = ConvNormActBlock(
                in_channels=self.input_image_channels,
                out_channels=self.initial_conv_filters,
                kernel_size=3,
                padding="same",
                bias=self.enable_bias,
                act="relu",
                norm_layer=None,
                use_conv3d=self.use_conv3d,
            )

        self.encoder = Encoder(
            in_channels=self.initial_conv_filters
            if self.initial_conv_filters
            else self.input_image_channels,
            encoder_filters=self.encoder_filters,
            convs_per_block=self.convs_per_block,
            bias=self.enable_bias,
            norm_layer=self.norm_layer,
            norm_kwargs=self.norm_kwargs,
            act=self.act,
            skip_connect=self.skip_connect,
            residual_blocks=self.residual_blocks,
            downscale_method=self.downscale_method,
            dropout_rate=self.dropout_rate,
            use_conv3d=self.use_conv3d,
            downscale_first_conv=self.downscale_first_conv,
            downscale_encoder_blocks=self.downscale_encoder_blocks,
        )

        self.bottleneck = UNetBlock(
            in_channels=self.encoder_filters[-2],
            out_channels=self.encoder_filters[-1],
            encoder=False,
            convs_per_block=self.convs_per_block,
            skip_connect=self.skip_connect,
            residual_blocks=self.residual_blocks,
            norm_layer=self.norm_layer,
            norm_kwargs=self.norm_kwargs,
            downscale_method=self.downscale_method,
            bias=self.enable_bias,
            use_conv3d=self.use_conv3d,
            downscale_first_conv=self.downscale_first_conv,
            downscale=self.downscale_bottleneck,
        )

        self.decoder = Decoder(
            in_channels=self.encoder_filters[-1],
            decoder_filters=self.decoder_filters,
            encoder_filters=self.encoder_filters,
            convs_per_block=self.convs_per_block,
            bias=self.enable_bias,
            norm_layer=self.norm_layer,
            norm_kwargs=self.norm_kwargs,
            act=self.act,
            skip_connect=self.skip_connect,
            residual_blocks=self.residual_blocks,
            downscale_method=self.downscale_method,
            dropout_rate=self.dropout_rate,
            use_conv3d=self.use_conv3d,
        )

        self.final_conv = ConvNormActBlock(
            in_channels=self.decoder_filters[-1],
            out_channels=self.num_output_channels,
            kernel_size=1,
            bias=True,
            padding="same",
            act="linear",
            norm_layer=None,
            use_conv3d=self.use_conv3d,
        )

        # initialize weights
        self.reset_parameters()

    def forward(self, inputs):
        outputs = inputs
        if self.initial_conv:
            outputs = self.initial_conv(outputs)
        outputs, skip_connections = self.encoder(outputs)
        outputs = self.bottleneck(outputs)
        outputs = self.decoder(outputs, skip_connections)
        outputs = self.final_conv(outputs)
        return outputs

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                self.initializer(m.weight)
                if m.bias is not None:
                    self.bias_initializer(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                self.norm_weight_initializer(m.weight)
                self.bias_initializer(m.bias)
