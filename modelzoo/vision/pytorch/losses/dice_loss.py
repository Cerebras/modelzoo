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

from modelzoo.common.pytorch.run_utils import half_dtype_instance


class Dice:
    def __init__(
        self,
        num_classes: int,
        to_onehot_y: bool = True,
        to_onehot_x: bool = False,
        use_softmax: bool = True,
        use_argmax: bool = False,
        include_background: bool = False,
        input_shape=None,
    ):
        self.num_classes = num_classes
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.to_onehot_x = to_onehot_x
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        self.smooth_nr = 0.0
        self.smooth_dr = 1e-6
        self.include_background = include_background
        self.input_shape = None
        self.bg_mask = None
        if not self.include_background:
            if input_shape:
                self.input_shape = input_shape
            else:
                raise ValueError(
                    "must supply input shape when include_background=False"
                )

    def _create_background_mask(self, device, dtype, ish, chanx):
        from modelzoo.common.pytorch import cb_model as cm

        z_shape = ish[0:chanx] + [1] + ish[chanx + 1 :]  # [N,1,D,H,W]
        o_shape = (
            ish[0:chanx] + [ish[chanx] - 1] + ish[chanx + 1 :]
        )  # [N,C-1,D,H,W]
        zeros = torch.zeros(z_shape, device=device, dtype=dtype)
        ones = torch.ones(o_shape, device=device, dtype=dtype)
        weights = torch.cat(
            (zeros, ones), chanx
        )  # [N,C,D,H,W] w/ first ch 0'ed
        if cm.use_cs():
            bg_mask = cm.make_constant(weights)
        else:
            bg_mask = weights.to(device)
        return bg_mask

    def __call__(self, prediction, target):
        target = torch.unsqueeze(target, 1)
        channel_axis = 1
        reduce_axis = list(range(2, len(prediction.shape)))
        num_pred_ch = prediction.shape[channel_axis]

        if self.use_softmax:
            prediction = torch.softmax(prediction, dim=channel_axis)
        elif self.use_argmax:
            prediction = torch.argmax(prediction, dim=channel_axis)

        if self.to_onehot_y:
            target = to_one_hot(target, channel_axis, self.num_classes)
        if self.to_onehot_x:
            prediction = to_one_hot(prediction, channel_axis, self.num_classes)

        if not self.include_background:
            if self.bg_mask is None:
                self.bg_mask = self._create_background_mask(
                    target.device, torch.float16, self.input_shape, channel_axis
                )
            assert (
                num_pred_ch > 1
            ), f"To exclude background the prediction needs more than one channel. Got {num_pred_ch}."
            target = target * self.bg_mask
            prediction = prediction * self.bg_mask

        assert (
            target.shape == prediction.shape
        ), f"Target and prediction shape do not match. Target: ({target.shape}), prediction: ({prediction.shape})."

        intersection = torch.sum(target * prediction, dim=reduce_axis)
        target_sum = torch.sum(target, dim=reduce_axis)
        prediction_sum = torch.sum(prediction, dim=reduce_axis)

        res = (2.0 * intersection + self.smooth_nr) / (
            target_sum + prediction_sum + self.smooth_dr
        )
        return res


def to_one_hot(array, channel_axis, num_classes):
    if len(array.shape) >= 5:
        array = torch.squeeze(array, dim=channel_axis)
    init = torch.zeros(
        array.shape + (num_classes,),
        device=array.device,
        dtype=half_dtype_instance.half_dtype,
    )
    array = init.scatter_(-1, array.long().unsqueeze(-1), 1.0).float()
    array = array.permute(0, 4, 1, 2, 3).float()
    return array


class DiceCELoss(nn.Module):
    def __init__(
        self, num_classes, input_shape, include_background, wc=0.5, wd=0.5,
    ):
        super(DiceCELoss, self).__init__()
        self.dice = Dice(
            num_classes=num_classes,
            include_background=include_background,
            input_shape=input_shape,
        )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.wc = wc
        self.wd = wd
        if not include_background:
            self.mean_correction = torch.tensor(
                num_classes / (num_classes - 1), dtype=torch.float32,
            )
        else:
            self.mean_correction = torch.tensor(1.0, dtype=torch.float32,)
        self.one_const = torch.tensor(1.0, dtype=torch.float32,)

    def forward(self, outputs, labels):
        ce = self.cross_entropy(outputs, labels)
        dc = self.mean_correction * torch.mean(self.dice(outputs, labels))
        loss = self.wc * ce + self.wd * (self.one_const - dc)
        return loss


class DiceScore:
    def __init__(
        self,
        to_onehot_y: bool = True,
        to_onehot_x: bool = True,
        use_argmax: bool = False,  # argmax already done in model
        use_softmax: bool = False,
        include_background: bool = False,
    ):
        self.dice = Dice(
            to_onehot_y=to_onehot_y,
            to_onehot_x=to_onehot_x,
            use_softmax=use_softmax,
            use_argmax=use_argmax,
            include_background=include_background,
        )

    def __call__(self, labels=None, predictions=None, weights=None):
        return torch.mean(self.dice(predictions, labels), dim=0)
