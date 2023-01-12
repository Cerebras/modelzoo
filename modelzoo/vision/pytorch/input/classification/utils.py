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

import logging

# ImageNet Defaults
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def create_preprocessing_params_with_defaults(params):
    """Preprocessing params for augmentations"""
    pp_params = dict()
    pp_params["noaugment"] = params.get("noaugment", False)
    pp_params["mixed_precision"] = params["mixed_precision"]

    ## crop, pad, and resize
    pp_params["rand_interpolation"] = params.get(
        "rand_interpolation", "bilinear"
    )
    pp_params["rand_scale"] = params.get("rand_scale", (0.08, 1.0))
    pp_params["rand_ratio"] = params.get("rand_ratio", (3.0 / 4.0, 4.0 / 3.0))
    pp_params["crop_padding"] = params.get("crop_padding", 32)

    ## flipping
    pp_params["h_flip_prob"] = params.get("h_flip_prob", 0.5)
    pp_params["v_flip_prob"] = params.get("v_flip_prob", 0.0)

    ## automatic data augmentmentation
    pp_params["augname"] = params.get("augname", None)
    pp_params["aug_interpolation"] = params.get("aug_interpolation", "bilinear")
    pp_params["ra_num_layer"] = params.get("ra_num_layer", 2)
    pp_params["ra_magnitude"] = params.get("ra_magnitude", 10)
    pp_params["ra_magnitude_std"] = params.get("ra_magnitude_std", 0)
    pp_params["ra_magnitude_max"] = params.get("ra_magnitude_max", None)
    pp_params["aa_policy"] = params.get("aa_policy", "imagenet")

    ## color jitter
    pp_params["color_jitter"] = params.get("color_jitter", None)
    pp_params["color_jitter_prob"] = params.get("color_jitter_prob", 0)

    ## random grayscale
    pp_params["rand_grayscale_prob"] = params.get("rand_grayscale_prob", 0)

    ## random erasing
    pp_params["rand_erase_prob"] = params.get("rand_erase_prob", 0)

    ## normalization
    pp_params["norm_mean"] = params.get("norm_mean", IMAGENET_DEFAULT_MEAN)
    pp_params["norm_std"] = params.get("norm_std", IMAGENET_DEFAULT_STD)

    ## switch between CIFAR10/100 and ImageNet-style augmentation
    pp_params["use_i1k_aug"] = params.get("use_i1k_aug", True)

    processor = params.get("data_processor")
    if pp_params["use_i1k_aug"]:
        if processor != "ImageNet1KProcessor":
            logging.warn(
                "Using ImageNet-1k data augmentation (use_i1k_aug=True), but the "
                f"data processor is {processor}!"
            )
    else:
        if processor not in ("CIFAR10Processor", "CIFAR100Processor"):
            logging.warn(
                "Using CIFAR10/100 data augmentation (use_i1k_aug=False), but the "
                f"data processor is {processor}!"
            )

    return pp_params
