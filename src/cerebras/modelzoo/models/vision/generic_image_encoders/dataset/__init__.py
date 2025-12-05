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

from pathlib import Path
from typing import Literal, Optional, Union

from torchvision.datasets import ImageNet as TorchVisionImageNet

from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.base_dataset import (
    DatasetConfig,
    SSLTransform,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.Dinov1SyntheticDataset import (
    Dinov1SyntheticDataset,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.IJEPASyntheticDataset import (
    IJEPASyntheticDataset,
)


class ImageNetConfig(DatasetConfig):
    name: Literal["Imagenet"]

    root: Union[str, Path]
    split: str = "train"

    transform: Optional[SSLTransform] = None

    @property
    def __dataset_cls__(self):
        return ImageNet


class ImageNet(TorchVisionImageNet):
    def __init__(self, config: ImageNetConfig):
        if isinstance(config, dict):
            config = ImageNetConfig(**config)

        super().__init__(
            root=config.root,
            split=config.split,
            transform=config.transform(),
        )
