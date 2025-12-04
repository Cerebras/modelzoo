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

#    Adapted from: https://github.com/mlcommons/training/tree/8e7ad54541aeda54a8e5152732b9fb293a22b10c
#    and: https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet/training/data_augmentation (f2282ed)
#
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import scipy.ndimage
import torch
from torchvision import transforms as torch_transforms

from cerebras.modelzoo.data.vision.segmentation.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.custom_transforms import (
    MaskTransform,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.default_data_augmentation import (
    default_3D_augmentation_params,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.noise_transforms import (
    GaussianNoiseTransform,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.utility_transforms import (
    NumpyToTensor,
    OneHotTransform,
    OneHotTransformKits,
    RemoveLabelTransform,
    RenameTransform,
)


class Compose:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


def get_moreDA_augmentation(
    is_training,
    patch_size,
    params=default_3D_augmentation_params,
    border_val_seg=-1,
    order_seg=1,
    order_data=3,
    convert_to_onehot=False,
    num_classes=3,
    image_dtype=torch.float32,
    labels_dtype=torch.int32,
):
    assert (
        params.get('mirror') is None
    ), "old version of params, use new keyword do_mirror"
    transforms = []
    if not is_training:
        transforms.append(RemoveLabelTransform(-1, 0))

    if is_training:
        patch_size_spatial = patch_size
        ignore_axes = None

        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=params.get("do_elastic"),
                alpha=params.get("elastic_deform_alpha"),
                sigma=params.get("elastic_deform_sigma"),
                do_rotation=params.get("do_rotation"),
                angle_x=params.get("rotation_x"),
                angle_y=params.get("rotation_y"),
                angle_z=params.get("rotation_z"),
                p_rot_per_axis=params.get("rotation_p_per_axis"),
                do_scale=params.get("do_scaling"),
                scale=params.get("scale_range"),
                border_mode_data=params.get("border_mode_data"),
                border_cval_data=0,
                order_data=order_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_seg,
                random_crop=params.get("random_crop"),
                p_el_per_sample=params.get("p_eldef"),
                p_scale_per_sample=params.get("p_scale"),
                p_rot_per_sample=params.get("p_rot"),
                independent_scale_for_each_axis=params.get(
                    "independent_scale_factor_for_each_axis"
                ),
            )
        )
        transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        transforms.append(
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25), p_per_sample=0.15
            )
        )
        transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
                ignore_axes=ignore_axes,
            )
        )
        transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                True,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=0.1,
            )
        )  # inverted gamma

        if params.get("do_gamma"):
            transforms.append(
                GammaTransform(
                    params.get("gamma_range"),
                    False,
                    True,
                    retain_stats=params.get("gamma_retain_stats"),
                    p_per_sample=params["p_gamma"],
                )
            )

        if params.get("do_mirror") or params.get("mirror"):
            transforms.append(MirrorTransform(params.get("mirror_axes")))

        if params.get("use_mask_for_norm") is not None:
            use_mask_for_norm = params.get("use_mask_for_norm")
            transforms.append(
                MaskTransform(
                    use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0
                )
            )

        transforms.append(RemoveLabelTransform(-1, 0))
    transforms.append(RenameTransform('seg', 'target', True))
    transforms.append(NumpyToTensor(cast_to=[image_dtype, labels_dtype]))
    if convert_to_onehot:
        transforms.append(OneHotTransform(num_classes))

    transforms = Compose(transforms)
    return transforms


def get_mlperf_transforms(
    patch_size,
    input_shape,
    convert_to_onehot=False,
    num_classes=3,
    image_dtype=torch.float32,
    labels_dtype=torch.int32,
    is_training=True,
):
    transforms = []
    if is_training:
        transforms.append(RandFlip())
        transforms.append(CSToTensor())
        if list(input_shape) != list(patch_size):
            transforms.append(
                CSReshape(
                    input_shape,
                    interpolation_type="nearest",
                )
            )
        transforms.append(Cast(types=(image_dtype, labels_dtype)))
        transforms.append(RandomBrightnessAugmentation(factor=0.3, prob=0.1))
        transforms.append(GaussianNoise(mean=0.0, std=0.1, prob=0.1))
        if convert_to_onehot:
            transforms.append(OneHotTransformKits(num_classes))
        transforms = torch_transforms.Compose(transforms)
    else:
        transforms.append(CSToTensor())
        transforms.append(Cast(types=(image_dtype, labels_dtype)))
        if convert_to_onehot:
            transforms.append(OneHotTransformKits(num_classes))
        transforms = torch_transforms.Compose(transforms)
    return transforms


class CSToTensor:
    def __call__(self, data_dict):
        data_dict['image'] = torch.tensor(data_dict['image'])
        data_dict['label'] = torch.tensor(data_dict['label'])
        return data_dict


class CSReshape:
    def __init__(self, new_size, interpolation_type):
        self.new_size = new_size
        self.transform = torch.nn.Upsample(
            size=list(self.new_size),
            mode='nearest',
        )

    def __call__(self, data_dict):
        data_dict['image'] = torch.squeeze(
            self.transform(torch.unsqueeze(data_dict['image'], 0)),
            0,
        )
        data_dict['label'] = torch.squeeze(
            self.transform(torch.unsqueeze(data_dict['label'], 0)),
            0,
        )
        return data_dict


class RandBalancedCrop:
    def __init__(self, patch_size, oversampling):
        self.patch_size = patch_size
        self.oversampling = oversampling

    def __call__(self, data):
        image, label = data["image"], data["label"]
        if False in [
            s1 >= s2 for s1, s2 in zip(image.shape[1:], self.patch_size)
        ]:
            raise ValueError(f'patch size greater than image size')
        if np.random.random() < self.oversampling:
            image, label, cords = self.rand_foreg_cropd(image, label)
        else:
            image, label, cords = self._rand_crop(image, label)
        data.update({"image": image, "label": label})
        return data

    @staticmethod
    def randrange(max_range):
        if max_range == 0:
            return 0
        else:
            return int(
                np.round(np.random.random_sample(1)[0] * (max_range - 1))
            )

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (
                foreg_slice[idx].stop - foreg_slice[idx].start
            )
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(
            scipy.ndimage.measurements.label(label == cl)[0]
        )
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [
            np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices
        ]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        # foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        foreg_slice = foreg_slices[np.random.randint(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]


class RandFlip:
    def __init__(self):
        self.axis = [1, 2, 3]
        self.prob = 1 / len(self.axis)
        # self.prob=1

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        for axis in self.axis:
            if np.random.random() < self.prob:
                data = self.flip(data, axis)
        return data


class Cast:
    def __init__(self, types):
        self.types = types

    def __call__(self, data):
        data["image"] = data["image"].to(self.types[0])
        data["label"] = data["label"].to(self.types[1])
        return data


class RandomBrightnessAugmentation:
    def __init__(self, factor, prob):
        self.prob = prob
        self.factor = factor

    def __call__(self, data):
        image = data["image"]
        if np.random.random() < self.prob:
            factor = torch.tensor(
                np.random.uniform(
                    low=1.0 - self.factor, high=1.0 + self.factor, size=1
                )
            ).to(image.dtype)
            image = (image * (1 + factor)).to(image.dtype)
            data.update({"image": image})
        return data


class GaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, data):
        image = data["image"]
        if np.random.random() < self.prob:
            scale = np.random.uniform(low=0.0, high=self.std)
            noise = torch.tensor(
                np.random.normal(loc=self.mean, scale=scale, size=image.shape)
            ).to(image.dtype)
            data.update({"image": image + noise})
        return data
