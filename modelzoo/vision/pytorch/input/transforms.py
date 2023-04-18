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

from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

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
    return x.to(mp_type)


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
