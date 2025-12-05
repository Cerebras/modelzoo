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

import math
from typing import List, Literal, Optional, Tuple, Union

import torch
from pydantic import field_validator
from torch.utils import data

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.base_dataset import (
    DatasetConfig,
    SSLTransform,
)


class IJEPASyntheticDatasetConfig(DatasetConfig):
    name: Literal["IJEPASyntheticDataset"]

    transform: Optional[SSLTransform] = None

    channels: int = ...
    image_size: Union[int, Tuple[int, int]] = ...  # (H, W)
    patch_size: Union[int, Tuple[int, int]] = [16, 16]
    num_labels: int = ...
    num_samples: Union[int, List[int]] = (100,)

    num_encoder_masks: int = 1
    encoder_mask_scale: Tuple[float, float] = [0.2, 0.8]
    encoder_aspect_ratio: Tuple[float, float] = [1.0, 1.0]
    # Predictor
    num_predictor_masks: Optional[int] = 2
    predictor_mask_scale: Tuple[float, float] = [0.2, 0.8]
    predictor_aspect_ratio: Tuple[float, float] = [0.3, 3.0]
    min_mask_patches: int = 4
    seed: int = 1223

    @field_validator("image_size", mode="after")
    @classmethod
    def validate_image_size(cls, image_size, **kwargs):
        if isinstance(image_size, int):
            return (image_size, image_size)
        return image_size

    @property
    def __dataset_cls__(self):
        return IJEPASyntheticDataset


class IJEPASyntheticDataset(data.Dataset):
    def __init__(self, config: IJEPASyntheticDatasetConfig):
        if isinstance(config, dict):
            config = IJEPASyntheticDatasetConfig(**config)

        super().__init__()

        self.transform = config.transform

        self.image_height, self.image_width = config.image_size
        self.channels = config.channels
        self.seed = config.seed
        self.num_labels = config.num_labels
        self.num_samples = config.num_samples

        self.image_size = config.image_size

        self.mp_type = cstorch.amp.get_floating_point_dtype()

        self.height, self.width = (
            config.image_size[0] // config.patch_size[0],
            config.image_size[1] // config.patch_size[1],
        )
        self.num_patches = self.height * self.width

        self.max_num_mask_patches_predictor = self._max_mask_patches(
            config.predictor_mask_scale, config.predictor_aspect_ratio
        )

        self.max_num_mask_patches_encoder = self._max_mask_patches(
            config.encoder_mask_scale, config.encoder_aspect_ratio
        )

        self.num_samples = config.num_samples
        self.seed = config.seed
        self.min_mask_patches = config.min_mask_patches
        self.num_encoder_masks = config.num_encoder_masks
        self.num_predictor_masks = config.num_predictor_masks

        self.data = self._create_data()

    def _max_mask_patches(self, scale, aspect_ratio_scale):
        max_area = -float("inf")
        for sc in scale:
            for ar in aspect_ratio_scale:
                max_keep = int(self.height * self.width * sc)

                h = int(round(math.sqrt(max_keep * ar)))
                w = int(round(math.sqrt(max_keep / ar)))

                h = min(self.height - 1, h)
                w = min(self.width - 1, w)

                if h * w > max_area:
                    max_area = h * w

        return max_area

    def _create_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        # Image data
        image_data = torch.rand(
            (
                self.num_samples,
                self.channels,
                self.image_height,
                self.image_width,
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

        num_valid_mask_encoder = torch.randint(
            self.min_mask_patches,
            self.max_num_mask_patches_encoder,
            (self.num_samples, 1),
            dtype=torch.int32,
            generator=generator,
        )
        num_valid_mask_encoder = num_valid_mask_encoder.repeat(
            (1, self.num_encoder_masks)
        )

        encoder_mask_idx = torch.randint(
            0,
            self.num_patches,
            (
                self.num_samples,
                self.num_encoder_masks,
                self.max_num_mask_patches_encoder,
            ),
            dtype=torch.int64,
            generator=generator,
        )

        _mask_enc = torch.arange(self.max_num_mask_patches_encoder).reshape(
            1, 1, -1
        )
        encoder_mask_idx[
            _mask_enc
            >= num_valid_mask_encoder.reshape(
                self.num_samples, self.num_encoder_masks, 1
            )
        ] = 0

        num_valid_mask_predictor = torch.randint(
            self.min_mask_patches,
            self.max_num_mask_patches_predictor,
            (self.num_samples, 1),
            dtype=torch.int32,
            generator=generator,
        )
        num_valid_mask_predictor = num_valid_mask_predictor.repeat(
            (1, self.num_predictor_masks)
        )

        predictor_mask_idx = torch.randint(
            0,
            self.num_patches,
            (
                self.num_samples,
                self.num_predictor_masks,
                self.max_num_mask_patches_predictor,
            ),
            dtype=torch.int64,
            generator=generator,
        )

        _mask_pred = torch.arange(self.max_num_mask_patches_predictor).reshape(
            1, 1, -1
        )
        predictor_mask_idx[
            _mask_pred
            >= num_valid_mask_predictor.reshape(
                self.num_samples, self.num_predictor_masks, 1
            )
        ] = 0

        loss_mask = torch.ones(
            (
                self.num_samples,
                self.num_predictor_masks,
                self.max_num_mask_patches_predictor,
            ),
            dtype=torch.float32,
        )
        loss_mask[
            _mask_pred
            >= num_valid_mask_predictor.reshape(
                self.num_samples, self.num_predictor_masks, 1
            )
        ] = 0.0

        return {
            "image": image_data,
            "labels": labels,
            "num_valid_mask_encoder": num_valid_mask_encoder,
            "encoder_mask_idx": encoder_mask_idx,
            "num_valid_mask_predictor": num_valid_mask_predictor,
            "predictor_mask_idx": predictor_mask_idx,
            "loss_mask": loss_mask.unsqueeze(3),
        }

    def __getitem__(self, index):
        index_data = {}
        for k in self.data.keys():
            index_data[k] = self.data[k][index]

        return index_data

    def __len__(self):
        return self.num_samples
