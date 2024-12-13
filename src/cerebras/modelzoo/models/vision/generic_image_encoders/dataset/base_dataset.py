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

from abc import ABC, abstractmethod
from typing import ClassVar, Union

from pydantic import Field
from typing_extensions import Annotated

from cerebras.modelzoo.config import NamedConfig, create_config_class
from cerebras.modelzoo.models.vision.generic_image_encoders.base.BaseSSLImageTransform import (
    BaseSSLImageTransformConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.Dinov2SyntheticTransform import (
    Dinov2SyntheticTransformConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.Dinov2Transform import (
    Dinov2TransformConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.ImageRandomMultiCropTransform import (
    ImageRandomMultiCropTransformConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.MaskedPatchTransform import (
    MaskedPatchTransformConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.MultiBlockMaskedContextImageTransform import (
    MultiBlockMaskedContextImageTransformConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.PassThroughTransform import (
    PassThroughTransformConfig,
)

SSLTransform = Annotated[
    Union[
        Dinov2SyntheticTransformConfig,
        Dinov2TransformConfig,
        ImageRandomMultiCropTransformConfig,
        MaskedPatchTransformConfig,
        MultiBlockMaskedContextImageTransformConfig,
        PassThroughTransformConfig,
    ],
    Field(discriminator=BaseSSLImageTransformConfig.discriminator),
]


class DatasetConfig(NamedConfig, ABC):
    discriminator: ClassVar[str] = "name"

    @property
    @abstractmethod
    def __dataset_cls__(self):
        pass

    def __call__(self, **kwargs):
        return create_config_class(self.__dataset_cls__).model_validate(self)()
