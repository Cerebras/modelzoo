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
import random
from inspect import getfullargspec, signature

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

import torch  # noqa


__all__ = [
    "create_transform",
]


SUPPORTED_TRANSFORMS = [
    # Transforms on PIL Image and torch.*Tensor
    "center_crop",
    "color_jitter_with_prob",
    "random_crop",
    "random_grayscale",
    "random_horizontal_flip",
    "random_resized_crop",
    "random_vertical_flip",
    "resize",
    "random_solarize",
    # Transforms on torch.*Tensor only
    "normalize",
    "random_erase",
    # Conversion transforms
    "to_dtype",
    "to_tensor",
    # Automatic augmentation transforms
    "autoaug",
    "randaug",
    "trivialaug",
    # "augmix", # available from torchvision 0.13
    # transforms on PIL image only
    "resize_center_crop_pil_image",
    "expand_to_square",
    "random_apply",
    "random_gray_scale",
    "random_gaussian_blur_random_radius",
]


def get_or_use_default(transform_spec, key, default_val):
    name = transform_spec.get("name")
    val = transform_spec.get(key)
    if val is None:
        logging.debug(
            f"Transform {name}'s {key} parameter is not specified. "
            f"Using default value {default_val}."
        )
        val = default_val
    return val


def create_transform(transform_spec):
    """
    Create the specified transform. For each transform, the parameter list (name
    and default value) follows those in torchvision 0.12
    (https://pytorch.org/vision/0.12/transforms.html)

    Args:
        name (str): name of the transform
        args (dict): a dictionary of parameters used to initialize the transform.
            Default is None.
    """
    name = transform_spec["name"].lower()

    if "interpolation" in transform_spec.keys():
        transform_spec["interpolation"] = InterpolationMode(
            transform_spec["interpolation"]
        )

    # Transforms on PIL Image and torch.*Tensor
    if name == "center_crop":
        return transforms.CenterCrop(size=transform_spec.get("size"))
    elif name == "color_jitter_with_prob":
        transform = transforms.ColorJitter(
            brightness=get_or_use_default(transform_spec, "brightness", 0),
            contrast=get_or_use_default(transform_spec, "contrast", 0),
            saturation=get_or_use_default(transform_spec, "saturation", 0),
            hue=get_or_use_default(transform_spec, "hue", 0),
        )
        color_jitter_prob = get_or_use_default(transform_spec, "p", 0)
        if color_jitter_prob > 0:
            transform = transforms.RandomApply([transform], p=color_jitter_prob)
        return transform
    elif name == "random_crop":
        pad_if_needed = get_or_use_default(
            transform_spec, "pad_if_needed", False
        )
        if pad_if_needed:
            logging.info(
                f"For RandomCrop, pad_if_needed is set to {pad_if_needed}, which "
                f"is different from torchvision's default (False)."
            )
        fill = get_or_use_default(transform_spec, "fill", 0)
        if fill != 0:
            logging.info(
                f"For RandomCrop, fill is set to {fill}, which is different "
                f"from torchvision's default (0)."
            )
        padding_mode = get_or_use_default(
            transform_spec, "padding_mode", "constant"
        )
        if padding_mode != "constant":
            logging.info(
                f"For RandomCrop, padding_mode is set to {padding_mode}, which "
                f"is different from torchvision's default (`constant`)."
            )
        return transforms.RandomCrop(
            size=transform_spec.get("size"),
            padding=get_or_use_default(transform_spec, "padding", None),
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
        )
    elif name == "random_grayscale":
        return transforms.RandomGrayscale(
            p=get_or_use_default(transform_spec, "p", 0.1),
        )
    elif name == "random_horizontal_flip":
        return transforms.RandomHorizontalFlip(
            p=get_or_use_default(transform_spec, "p", 0.5),
        )
    elif name == "random_resized_crop":
        return transforms.RandomResizedCrop(
            size=transform_spec.get("size"),
            scale=get_or_use_default(transform_spec, "scale", (0.08, 1.0)),
            ratio=get_or_use_default(
                transform_spec, "ratio", (0.75, 4.0 / 3.0)
            ),
            interpolation=get_or_use_default(
                transform_spec, "interpolation", InterpolationMode.BILINEAR
            ),
        )
    elif name == "random_vertical_flip":
        return transforms.RandomVerticalFlip(
            p=get_or_use_default(transform_spec, "p", 0.5),
        )
    elif name == "resize":
        return transforms.Resize(
            size=transform_spec.get("size"),
            interpolation=get_or_use_default(
                transform_spec, "interpolation", InterpolationMode.BILINEAR
            ),
            max_size=get_or_use_default(transform_spec, "max_size", None),
            antialias=get_or_use_default(transform_spec, "antialias", None),
        )

    # Transforms on torch.*Tensor only
    elif name == "normalize":
        inplace = get_or_use_default(transform_spec, "inplace", False)
        if inplace:
            logging.info(
                f"For Normalize, inplace is set to {inplace}, which "
                f"is different from torchvision's default (False)."
            )
        return transforms.Normalize(
            mean=transform_spec.get("mean"),
            std=transform_spec.get("std"),
            inplace=inplace,
        )
    elif name == "random_erase":
        return transforms.RandomErasing(
            p=get_or_use_default(transform_spec, "p", 0.5),
            scale=get_or_use_default(transform_spec, "scale", (0.02, 0.33)),
            ratio=get_or_use_default(transform_spec, "ratio", (0.3, 3.3)),
            value=get_or_use_default(transform_spec, "value", 0),
            inplace=get_or_use_default(transform_spec, "inplace", False),
        )

    # Conversion transforms
    elif name == "to_dtype":
        return LambdaWithParam(dtype_transform, transform_spec.get("mp_type"))
    elif name == "to_tensor":
        return transforms.ToTensor()

    elif name == "resize_center_crop_pil_image":
        # Used to DiT model image transform
        return LambdaWithParam(
            resize_center_crop_pil_image, transform_spec.get("size")
        )

    elif name == "expand_to_square":
        return LambdaWithParam(
            expand_to_square, transform_spec.get("background_color")
        )

    elif name == "random_apply":
        random_transforms = transform_spec.get("transforms")
        transform_list = []
        for t in random_transforms:
            transform_list.append(create_transform(t))

        transform_list = transforms.Compose(transform_list)
        return transforms.RandomApply(
            random_transforms, transform_spec.get("p", 0.5)
        )

    elif name == "random_gray_scale":
        return transforms.RandomGrayscale(transform_spec.get("p"))

    elif name == "random_gaussian_blur_random_radius":
        return RandomGaussianBlurRandomRadius(
            p=transform_spec.get("p", 0.5),
            radius_min=transform_spec.get("radius_min", 1.0),
            radius_max=transform_spec.get("radius_max", 2.0),
        )

    elif name == "random_solarize":
        return transforms.RandomSolarize(
            threshold=transform_spec.get("threshold"),
            p=transform_spec.get("p", 0.5),
        )

    # Automatic augmentation transforms
    elif name == "autoaug":
        policy = get_or_use_default(transform_spec, "policy", "imagenet")
        interpolation = get_or_use_default(
            transform_spec, "interpolation", InterpolationMode.NEAREST
        )
        if interpolation != InterpolationMode.NEAREST:
            logging.info(
                f"For AutoAugment, interpolation is set to {interpolation}, which "
                f"is different from torchvision's default (InterpolationMode.NEAREST)."
            )
        return autoaugment.AutoAugment(
            policy=autoaugment.AutoAugmentPolicy(policy),
            interpolation=interpolation,
            fill=get_or_use_default(transform_spec, "fill", None),
        )
    elif name == "randaug":
        magnitude = get_or_use_default(transform_spec, "magnitude", 9)

        # magnitude_std and magnitude_max are extra parameters to keep
        # consistent with timm's implementation: https://timm.fast.ai/RandAugment
        magnitude_std = get_or_use_default(transform_spec, "magnitude_std", 0)
        # If magnitude_std is > 0, we introduce some randomness in the fixed
        # policy and sample magnitude from a normal distribution with mean
        # `magnitude` and std-dev of `magnitude_std`.
        if magnitude_std > 0:
            # If magnitude_std is inf, we sample from a uniform distribution
            logging.info(
                f"RandAugment's magnitude_std={magnitude_std}. We will "
                f"introduce some randomness in the usually fixed policy and "
                f"sample from a distribution according to timm."
            )
            if magnitude_std == float("inf"):
                magnitude = random.uniform(0, magnitude)
            else:
                magnitude = random.gauss(magnitude, magnitude_std)

        upper_bound = get_or_use_default(transform_spec, "magnitude_max", 10)
        if magnitude > upper_bound:
            logging.info(
                f"Capping magnitude for RandAugment from {magnitude} to "
                f"magnitude_max={upper_bound} following timm."
            )
        magnitude = max(0.0, min(magnitude, upper_bound))

        return autoaugment.RandAugment(
            num_ops=get_or_use_default(transform_spec, "num_ops", 2),
            magnitude=magnitude,
            num_magnitude_bins=get_or_use_default(
                transform_spec, "num_magnitude_bins", 31
            ),
            interpolation=get_or_use_default(
                transform_spec, "interpolation", InterpolationMode.NEAREST
            ),
            fill=get_or_use_default(transform_spec, "fill", None),
        )
    elif name == "trivialaug":
        return autoaugment.TrivialAugmentWide(
            num_magnitude_bins=get_or_use_default(
                transform_spec, "num_magnitude_bins", 31
            ),
            interpolation=get_or_use_default(
                transform_spec, "interpolation", InterpolationMode.NEAREST
            ),
            fill=get_or_use_default(transform_spec, "fill", None),
        )
    # Only available starting torchvision 0.13
    # elif name == "augmix":
    #     return autoaugment.AugMix(
    #         severity=get_or_use_default(transform_spec, "severity", 3),
    #         mixture_width=get_or_use_default(transform_spec, "mixture_width", 3),
    #         chain_depth=get_or_use_default(transform_spec, "chain_depth", -1),
    #         alpha=get_or_use_default(transform_spec, "alpha", 1.0),
    #         all_ops=get_or_use_default(transform_spec, "all_ops", True),
    #         interpolation=get_or_use_default(transform_spec, "interpolation", InterpolationMode.BILINEAR),
    #         fill=get_or_use_default(transform_spec, "fill", None),
    #     )
    else:
        raise ValueError(f"Unsupported or invalid transform name: {name}.")


def dtype_transform(x, mp_type, *args, **kwargs):
    if isinstance(mp_type, str):
        mp_type = eval(mp_type)
    return x.to(mp_type)


def resize_center_crop_pil_image(
    pil_image, image_height, image_width, *args, **kwargs
):
    """
    Using same cropping mechanism as source DiT repo
    https://github.com/facebookresearch/DiT/blob/main/train.py#L85

    Based on Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while (
        pil_image.size[0] >= 2 * image_width
        and pil_image.size[1] >= 2 * image_height
    ):
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scales = [
        image_width / min(*pil_image.size),
        image_height / min(*pil_image.size),
    ]
    pil_image = pil_image.resize(
        tuple(round(x * s) for x, s in zip(pil_image.size, scales)),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)  # (W, H) -> (H, W)
    arr_height = arr.shape[0]
    arr_width = arr.shape[1]
    crop_y = (arr_height - image_height) // 2
    crop_x = (arr_width - image_width) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_height, crop_x : crop_x + image_width]
    )


def expand_to_square(pil_img, background_color, *args, **kwargs):
    """Pad with background color with image centered

    Args:
        pil_image: Input PIL image
        background_color: Tuple of integers representing the color
    """
    background_color = tuple(background_color)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class RandomGaussianBlurRandomRadius(object):
    """
    Apply Gaussian Blur to the PIL image with a probability
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, pil_img):

        do_it = random.random() <= self.prob
        if not do_it:
            return pil_img

        return pil_img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.prob}, radius_min={self.radius_min}, radius_max={self.radius_max})"


class LambdaWithParam(object):
    def __init__(self, lambd, *args, **kwargs):
        assert callable(lambd), (
            repr(type(lambd).__name__) + " object is not callable"
        )
        self.lambd = lambd
        self.args = args
        self.kwargs = kwargs
        ll_sig = getfullargspec(lambd)
        if not ll_sig.varargs or not ll_sig.varkw:
            raise TypeError(
                "User-defined lambda transform function must have signature: "
                "function(img, positional args, *args, **kwargs). Instead, "
                f"got function{str(signature(lambd))}."
            )

    def __call__(self, img):
        return self.lambd(img, *self.args, **self.kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(args={0}, kwargs={1})'.format(
            self.args, self.kwargs
        )
