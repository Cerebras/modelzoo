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

from typing import Literal, Optional, Tuple, Union

import torch
from pydantic import field_validator
from torch.utils import data

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.base_dataset import (
    DatasetConfig,
    SSLTransform,
)


class Dinov1SyntheticDatasetConfig(DatasetConfig):
    name: Literal["Dinov1SyntheticDataset"]

    transform: Optional[SSLTransform] = None

    channels: int = ...
    global_image_size: Union[int, Tuple[int, int]] = ...  # (H, W)
    num_global_crops: int = ...
    local_image_size: Union[int, Tuple[int, int]] = ...
    num_local_crops: int = ...
    num_samples: Optional[int] = 100
    mixed_precision: Optional[bool] = True
    num_labels: Optional[int] = 100
    seed: Optional[int] = 1223

    @field_validator("global_image_size", "local_image_size", mode="after")
    @classmethod
    def validate_image_size(cls, image_size):
        if isinstance(image_size, int):
            return (image_size, image_size)
        return image_size

    @property
    def __dataset_cls__(self):
        return Dinov1SyntheticDataset


class Dinov1SyntheticDataset(data.Dataset):
    def __init__(self, config: Dinov1SyntheticDatasetConfig):
        if isinstance(config, dict):
            config = Dinov1SyntheticDatasetConfig(**config)

        super().__init__()

        self.channels = config.channels
        self.seed = config.seed
        self.num_samples = config.num_samples
        self.num_local_crops = config.num_local_crops
        self.num_labels = config.num_labels
        self.num_global_crops = config.num_global_crops
        self.transform = config.transform

        self.global_image_size = config.global_image_size
        self.local_image_size = config.local_image_size

        self.mp_type = cstorch.amp.get_floating_point_dtype()

        self.data = self._create_data()

    def _create_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        # Image data
        global_view = torch.rand(
            (
                self.num_samples,
                self.channels,
                self.global_image_size[0],
                self.global_image_size[1],
            ),
            generator=generator,
            dtype=self.mp_type,
        )
        local_view = torch.rand(
            (
                self.num_samples,
                self.channels,
                self.local_image_size[0],
                self.local_image_size[1],
            ),
            generator=generator,
            dtype=self.mp_type,
        )

        labels = torch.randint(
            0,
            self.num_labels,
            (self.num_samples,),
            dtype=torch.int32,
            generator=generator,
        )

        return {
            "global_view": global_view,
            "local_view": local_view,
            "labels": labels,
        }

    def __getitem__(self, index):
        index_data = {}

        # All crops have same tensors
        bd_shape = [self.num_local_crops, self.channels, *self.local_image_size]
        index_data["local_view"] = (
            self.data["local_view"][index].unsqueeze(0).broadcast_to(*bd_shape)
        )

        bd_shape = [
            self.num_global_crops,
            self.channels,
            *self.global_image_size,
        ]
        index_data["global_view"] = (
            self.data["global_view"][index].unsqueeze(0).broadcast_to(*bd_shape)
        )

        index_data["labels"] = self.data["labels"][index]

        return index_data

    def __len__(self):
        return self.num_samples
