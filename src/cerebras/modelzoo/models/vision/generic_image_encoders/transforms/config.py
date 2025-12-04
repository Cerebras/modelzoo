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

"""
Config classes of transforms configs for generic image encoders.
"""

from typing import List, Literal, Optional, Tuple, Union

from annotated_types import Ge, Le
from pydantic import field_validator, model_validator
from typing_extensions import Annotated

from cerebras.modelzoo.models.vision.generic_image_encoders.base.BaseSSLImageTransform import (
    BaseSSLImageTransformConfig,
)


class PassThroughTransformConfig(BaseSSLImageTransformConfig):
    name: Literal["PassThroughTransform"]

    output_keys: List[str] = ...

    @property
    def __transform_cls__(self):
        from cerebras.modelzoo.models.vision.generic_image_encoders.transforms import (
            PassThroughTransform,
        )

        return PassThroughTransform


class Dinov2SyntheticTransformConfig(BaseSSLImageTransformConfig):
    name: Literal["Dinov2SyntheticTransform"]

    image_size: List[int] = ...
    patch_size: List[int] = ...
    mask_probability: float = ...
    mask_ratio_tuple: Tuple = ...
    min_num_patches: int = 4

    @property
    def output_keys(self):
        return ["local_view", "global_view", "labels", "collated_masks"]

    @property
    def __transform_cls__(self):
        from cerebras.modelzoo.models.vision.generic_image_encoders.transforms import (
            Dinov2SyntheticTransform,
        )

        return Dinov2SyntheticTransform


class MultiBlockMaskedContextImageTransformConfig(BaseSSLImageTransformConfig):
    name: Literal["MultiBlockMaskedContextImageTransform"]
    "Name of the data transform. Must be set to `MultiBlockMaskedContextImageTransform`."

    image_size: Union[int, Tuple[int, int]] = ...  # (H, W)
    "Size of the input image. If a single integer is provided, the image is assumed to be square."

    transform: Optional[List[dict]] = None
    "Optional list of sub-transforms to apply to the image."

    patch_size: Tuple[int, int] = [16, 16]
    "Patch size for the image."

    # Encoder
    num_encoder_masks: int = 1
    "How many encoder masks to use."
    encoder_mask_scale: Tuple[float, float] = [0.2, 0.8]
    encoder_aspect_ratio: Tuple[float, float] = [1.0, 1.0]
    # Predictor
    num_predictor_masks: int = 2
    predictor_mask_scale: Tuple[float, float] = [0.2, 0.8]
    predictor_aspect_ratio: Tuple[float, float] = [0.3, 3.0]
    min_mask_patches: int = 4
    allow_overlap: bool = False

    @property
    def output_keys(cls):
        return [
            "image",
            "labels",
            "encoder_mask_idx",
            "predictor_mask_idx",
            "num_valid_mask_encoder",
            "num_valid_mask_predictor",
            "loss_mask",
        ]

    @field_validator("image_size", "patch_size", mode="after")
    @classmethod
    def validate_sizes(cls, size):
        if isinstance(size, int):
            return (size, size)
        return size

    @property
    def __transform_cls__(self):
        from cerebras.modelzoo.models.vision.generic_image_encoders.transforms import (
            MultiBlockMaskedContextImageTransform,
        )

        return MultiBlockMaskedContextImageTransform


class MaskedPatchTransformConfig(BaseSSLImageTransformConfig):
    name: Literal["MaskedPatchTransform"]
    "Name of the data transform. Must be set to `MaskedPatchTransform`."

    image_size: Union[int, List[int]] = ...
    "The size of the input image. When provided as a single int, the image is assumed to be square."

    patch_size: Union[int, List[int]] = ...
    "The size of each patch to be masked. When provided as a single int, the patch is assumed to be square."

    mask_probability: Annotated[float, Ge(0), Le(1)] = ...
    "Probability of applying masking to image. When the value is 0, the image is left unmasked."

    mask_ratio_tuple: Tuple[float, float] = ...
    """
    When `mask_probability` is nonzero, this field specifies the proportion of patches to mask.
    The two fields represent the lower and upper bounds of a uniform distribution that is used to sample the masks.
    """

    min_num_patches: int = 4
    """
    Minimum number of patches to be masked in images that have masked patches.
    Default is 4 (from DinoV2 repo).
    """

    min_aspect: float = 0.3
    "Minimum aspect ratio for the patches."

    max_aspect: Optional[float] = None
    """
    Maximum aspect ratio for the patches. When `None`, this is calculated as
    `1 / min_aspect`.
    """

    composed_transform: bool = False
    "Whether this transform is a sub-class of another transform."

    transform_list: Optional[List[dict]] = None
    """
    List of transforms to be applied if composed_transform is False. Default is
    None. If composed_transform is True the transforms are assumed to be
    performed in the other transform.
    """

    @property
    def output_keys(self):
        keys = ["collated_masks"]
        if not self.composed_transform:
            keys.append("image")
        return keys

    @field_validator("image_size", "patch_size", mode="after")
    @classmethod
    def validate_sizes(cls, size):
        if isinstance(size, int):
            return (size, size)
        return size

    @model_validator(mode="after")
    def validate_transform_list(self):
        if self.composed_transform and self.transform_list is not None:
            raise ValueError(
                "Cannot specify transform_list if MaskedPatchTransform is a composed_transform"
            )
        return self

    def post_init(self, context):
        if self.max_aspect is None:
            self.max_aspect = 1 / self.min_aspect

    @property
    def __transform_cls__(self):
        from cerebras.modelzoo.models.vision.generic_image_encoders.transforms import (
            MaskedPatchTransform,
        )

        return MaskedPatchTransform


class ImageRandomMultiCropTransformConfig(BaseSSLImageTransformConfig):
    name: Literal["ImageRandomMultiCropTransform"]
    "Name of the data transform. Must be set to `ImageRandomMultiCropTransform`."

    global_num_crops: int = ...
    "Number of global crops of image."

    global_image_size: int = ...
    "Image size of global crops."

    global_crops_scale: Tuple[float, float] = ...
    """
    The scale (between 0 and 1) of the global crops with respect to the original image size.
    The two values represent the scales for the width and height of the crop.
    """

    local_num_crops: int = ...
    "Number of local crops of image."

    local_image_size: int = ...
    """
    Image size of local crops.
    Local crops are always square, so this value is defined by a single integer.
    """

    local_crops_scale: Tuple[float, float] = ...
    """
    The scale (between 0 and 1) of the local crops with respect to the original image size.
    The two values represent the scales for the width and height of the crop.
    """

    interpolation_type: Literal[
        "bicubic",
        "bilinear",
        "nearest",
        "nearest-exact",
        "box",
        "hamming",
        "lanczos",
    ] = "bicubic"  # default from DinoV2 repo
    """
    Method of interpolation for RandomResizedCropTransform to generate global and local views.

    Default used in DinoV2 is bicubic.
    """

    multicrop_transform_list: Optional[List[dict]] = None
    "Optional set of additional transforms to apply on top of local and global crops."

    @property
    def output_keys(self):
        return ["local_view", "global_view", "labels"]

    @property
    def __transform_cls__(self):
        from cerebras.modelzoo.models.vision.generic_image_encoders.transforms import (
            ImageRandomMultiCropTransform,
        )

        return ImageRandomMultiCropTransform


class Dinov2TransformConfig(BaseSSLImageTransformConfig):
    name: Literal["Dinov2Transform"]
    "Name of the data transform. Must be set to `Dinov2Transform`."

    multi_crop_transform: ImageRandomMultiCropTransformConfig = ...
    "Configuration for the image transformation. See `ImageRandomMultiCropTransformConfig` for more details."

    masked_patch_transform: MaskedPatchTransformConfig = ...
    "Configuration for the masked patch transformation. See `MaskedPatchTransformConfig` for more details."

    @property
    def output_keys(self):
        return (
            self.multi_crop_transform.output_keys
            + self.masked_patch_transform.output_keys
        )

    def post_init(self, context):
        if not self.masked_patch_transform.composed_transform:
            self.masked_patch_transform = self.masked_patch_transform.copy(
                update=dict(composed_transform=True)
            )

    @property
    def __transform_cls__(self):
        from cerebras.modelzoo.models.vision.generic_image_encoders.transforms import (
            Dinov2Transform,
        )

        return Dinov2Transform
