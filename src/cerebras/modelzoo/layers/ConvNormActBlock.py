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

import torch.nn as nn

from cerebras.modelzoo.layers.activations import get_activation
from cerebras.modelzoo.layers.norms import get_norm
from cerebras.modelzoo.layers.utils import ModuleWrapperClass


class ConvNormActBlock(nn.Sequential):
    """
    Customizable Convolution -> Normalization -> Activation Block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="valid",
        dilation=1,
        groups=1,
        bias=False,
        padding_mode='zeros',
        norm_layer="batchnorm2d",
        norm_kwargs=None,
        act="relu",
        device=None,
        dtype=None,
        use_conv3d=False,
        affine=True,
    ):
        """
        :param (int) in_channels : Number of channels in the input image
        :param (int) out_channels: Number of channels produced by convolution
        :param (int or tuple) kernel_size : Size of the convolving kernel
        :param (int or tuple, optional) stride : Stride of the convolution.
            Default: 1
        :param (str) padding : controls the amount of padding applied
            to the input. Can be either `valid` or `same`.
        :param (int or tuple, optional) dilation: Spacing between
            kernel elements. Default: 1
        :param (int, optional) groups : Number of blocked connections from
            input channels to output channels. Default: 1
        :param (bool, optional) bias: If `True`, adds a learnable bias to the
            output. Default: `False`
        :param (str, optional) padding_mode: `'zeros'`, `'reflect'`,
            `'replicate'` or `'circular'`. Default: `'zeros'`
        :param (str, optional) norm_layer: Type of normalization to be used.
            Supported norm layers can be found in
            `modelzoo.layers.norms.py`.
            Default: `batchnorm2d`
        :param (dict, optional) norm_kwargs: args to be passed to norm layers
            during initialization.
            For
            `norm_type` = `group`,
                `norm_kwargs` must include `num_groups` key value pair.
            `norm_type` = `layer`,
                `norm_kwargs` must include `normalized_shape` key value pair.
        :param (str, optional) act: Activation to be used.
            Supported activation layers can be
            found in `../../../common/pytorch/model_utils/activations.py`.
            Default: `relu`
        :param (str, optional) device: Device to place conv layer.
        :param (torch.dtype, optional) dtype: Datatype to be used for
            `weight` and `bias` of convolution layer.
        """
        super(ConvNormActBlock, self).__init__()
        if padding == "same":
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

        conv_layer = nn.Conv3d if use_conv3d else nn.Conv2d

        self.conv = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self._layers = [self.conv]

        if norm_kwargs is None:
            norm_kwargs = {}
        else:
            norm_kwargs = copy.deepcopy(norm_kwargs)

        if norm_layer is not None:
            norm_kwargs.update(_get_norm_kwargs(norm_layer, out_channels))
            norm_layer = get_norm(norm_layer)
            self.norm_layer = norm_layer(**norm_kwargs)
            self._layers.append(self.norm_layer)

        if act is not None:
            self.act = get_activation(act)
            self._layers.append(ModuleWrapperClass(self.act, act))

        super(ConvNormActBlock, self).__init__(*self._layers)


class ConvNormActLayers:
    """
    Customizable Convolution -> Normalization -> Activation Block.
    Returns list of layers in the above order when `get_layers` method is called.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="valid",
        dilation=1,
        groups=1,
        bias=False,
        padding_mode='zeros',
        norm_layer="batchnorm2d",
        norm_kwargs=None,
        act="relu",
        device=None,
        dtype=None,
        use_conv3d=False,
        affine=True,
    ):
        """
        :param (int) in_channels : Number of channels in the input image
        :param (int) out_channels: Number of channels produced by convolution
        :param (int or tuple) kernel_size : Size of the convolving kernel
        :param (int or tuple, optional) stride : Stride of the convolution.
            Default: 1
        :param (str) padding : controls the amount of padding applied
            to the input. Can be either `valid` or `same`.
        :param (int or tuple, optional) dilation: Spacing between
            kernel elements. Default: 1
        :param (int, optional) groups : Number of blocked connections from
            input channels to output channels. Default: 1
        :param (bool, optional) bias: If `True`, adds a learnable bias to the
            output. Default: `False`
        :param (str, optional) padding_mode: `'zeros'`, `'reflect'`,
            `'replicate'` or `'circular'`. Default: `'zeros'`
        :param (str, optional) norm_layer: Type of normalization to be used.
            Supported norm layers can be found in
            `modelzoo.layers.norms.py`.
            Default: `batchnorm2d`
        :param (dict, optional) norm_kwargs: args to be passed to norm layers
            during initialization.
            For
            `norm_type` = `group`,
                `norm_kwargs` must include `num_groups` key value pair.
            `norm_type` = `layer`,
                `norm_kwargs` must include `normalized_shape` key value pair.
        :param (str, optional) act: Activation to be used.
            Supported activation layers can be
            found in `../../../common/pytorch/model_utils/activations.py`.
            Default: `relu`
        :param (str, optional) device: Device to place conv layer.
        :param (torch.dtype, optional) dtype: Datatype to be used for
            `weight` and `bias` of convolution layer.
        """

        if padding == "same":
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

        conv_layer = nn.Conv3d if use_conv3d else nn.Conv2d

        conv = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self._layers = [conv]

        if norm_kwargs is None:
            norm_kwargs = {}
        else:
            norm_kwargs = copy.deepcopy(norm_kwargs)

        if norm_layer is not None:
            norm_kwargs.update(_get_norm_kwargs(norm_layer, out_channels))
            norm_layer = get_norm(norm_layer)
            norm = norm_layer(**norm_kwargs)
            self._layers.append(norm)

        if act is not None:
            act_layer = get_activation(act)
            self._layers.append(ModuleWrapperClass(act_layer, act))

    def get_layers(self):
        return self._layers


def _get_norm_kwargs(norm_type, value):
    """
    Function to update the positional args of norm layers.
    :param str norm_type: Normalization layer to be used.
    :param int value: Update positional args of norm layers
        `num_channels` for `group` and `batchchannel` norm_type
        `num_features` for `batch` and `instance` norm_type
        with this value
    """

    kwargs = {}

    if "group" in norm_type or "batchchannel" in norm_type:
        # `norm_kwargs` must include "num_groups"
        kwargs = {"num_channels": value}
    elif "batch" in norm_type:
        kwargs = {"num_features": value}
    elif "instance" in norm_type:
        # by default, affine = False for InstanceNorm, setting to True for
        # consistency with batchnorm
        kwargs = {"num_features": value, "affine": True}
    elif "layer" in norm_type:
        # `norm_kwargs`` must include "normalized_shape" kwargs
        pass

    return kwargs
