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

import random

import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from modelzoo.vision.pytorch.input.transforms import LambdaWithParam


def dtype_transform(x, mp_type, *args, **kwargs):
    return x.to(mp_type)


def preprocess_for_train(image_size, params):
    transform_list = []
    if params["use_i1k_aug"]:
        if image_size > 224:
            size = image_size
            resize_transform = transforms.Resize([size, size])
            transform_list.append(resize_transform)

        rand_resize_crop_transform = transforms.RandomResizedCrop(
            image_size,
            scale=params["rand_scale"],
            ratio=params["rand_ratio"],
            interpolation=InterpolationMode(params["rand_interpolation"]),
        )
        transform_list.append(rand_resize_crop_transform)

        h_flip_prob = params["h_flip_prob"]
        v_flip_prob = params["v_flip_prob"]
        if h_flip_prob > 0.0:
            h_flip_transform = transforms.RandomHorizontalFlip(h_flip_prob)
            transform_list.append(h_flip_transform)
        if v_flip_prob > 0.0:
            v_flip_transform = transforms.RandomVerticalFlip(v_flip_prob)
            transform_list.append(v_flip_transform)

        # apply automatic data augmentation
        if params["augname"] is not None:
            interpolation = InterpolationMode(params["aug_interpolation"])
            if params["augname"] == "randaug":
                num_ops = params["ra_num_layer"]
                magnitude = params["ra_magnitude"]
                magnitude_std = params["ra_magnitude_std"]
                # If magnitude_std is > 0, we introduce some randomness in the fixed
                # policy and sample magnitude from a normal distribution with mean
                # `magnitude` and std-dev of `magnitude_std`.
                if magnitude_std > 0:
                    # If magnitude_std is inf, we sample from a uniform distribution
                    if magnitude_std == float("inf"):
                        magnitude = random.uniform(0, magnitude)
                    else:
                        magnitude = random.gauss(magnitude, magnitude_std)
                # default upper_bound is 10
                upper_bound = params["ra_magnitude_max"] or 10
                magnitude = max(0.0, min(magnitude, upper_bound))
                transform_list.append(
                    autoaugment.RandAugment(
                        num_ops=num_ops,
                        magnitude=magnitude,
                        interpolation=interpolation,
                    )
                )
            elif params["augname"] == "trivialaug":
                transform_list.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif params["augname"] == "augmix":
                transform_list.append(
                    autoaugment.AugMix(interpolation=interpolation)
                )
            else:
                policy = params["aa_policy"]
                transform_list.append(
                    autoaugment.AutoAugment(
                        policy=autoaugment.AutoAugmentPolicy(policy),
                        interpolation=interpolation,
                    )
                )

        if params["color_jitter"] is not None:
            color_jitter = params["color_jitter"]
            if isinstance(params["color_jitter"], (list, tuple)):
                # color jitter should be: [brightness, contrast, saturation, (hue)]
                assert len(color_jitter) in (3, 4)
            else:
                # treat scalar as strength
                color_jitter = (float(color_jitter),) * 3

            color_jitter_transform = transforms.ColorJitter(*color_jitter)
            if params["color_jitter_prob"] > 0:
                color_jitter_transform = transforms.RandomApply(
                    [color_jitter_transform], p=params["color_jitter_prob"]
                )
            transform_list.append(color_jitter_transform)

        if params["rand_grayscale_prob"] > 0:
            transform_list.append(
                transforms.RandomGrayscale(p=params["rand_grayscale_prob"])
            )

        if params["rand_erase_prob"] > 0:
            transform_list.append(
                transforms.RandomErasing(p=params["rand_erase_prob"])
            )

    else:  # use cifar aug
        transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params["norm_mean"], std=params["norm_std"]
            ),
        ]
    )

    mixed_precision = params.get("mixed_precision")
    if mixed_precision:
        mp_type = torch.float16
    else:
        mp_type = torch.float32

    transform_list.append(LambdaWithParam(dtype_transform, mp_type))

    transform = transforms.Compose(transform_list)
    return transform


def preprocess_for_eval(image_size, params):
    if params["use_i1k_aug"]:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    256,
                    interpolation=InterpolationMode(
                        params["rand_interpolation"]
                    ),
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=params["norm_mean"], std=params["norm_std"]
                ),
            ]
        )

    else:  # use cifar aug
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=params["norm_mean"], std=params["norm_std"]
                ),
            ]
        )
    return transform


def get_preprocess_transform(image_size, params, use_training_transforms=True):
    if not use_training_transforms or params["noaugment"]:
        return preprocess_for_eval(image_size, params)
    else:
        return preprocess_for_train(image_size, params)
