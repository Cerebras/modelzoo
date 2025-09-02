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
import os

import torch
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms
from torchvision.utils import save_image

import cerebras.pytorch as cstorch
from cerebras.modelzoo.data.vision.transforms import create_transform
from cerebras.modelzoo.models.vision.generic_image_encoders.base.BaseSSLImageTransform import (
    BaseSSLImageTransform,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.config import (
    ImageRandomMultiCropTransformConfig,
)


class ImageRandomMultiCropTransform(BaseSSLImageTransform):
    def __init__(self, config: ImageRandomMultiCropTransformConfig):
        if isinstance(config, dict):
            config = ImageRandomMultiCropTransformConfig(**config)

        self.config = config

        self.global_num_crops, self.local_num_crops = (
            config.global_num_crops,
            config.local_num_crops,
        )
        self.total_crops = self.global_num_crops + self.local_num_crops
        self.global_image_size = config.global_image_size
        self.local_image_size = config.local_image_size

        self.global_crops_scale = config.global_crops_scale
        self.local_crops_scale = config.local_crops_scale

        self.interpolation = (
            Image.BICUBIC
            if config.interpolation_type == "bicubic"
            else Image.BILINEAR
        )

        self.addnl_transform = create_transform(
            {
                "name": "to_dtype",
                "mp_type": cstorch.amp.get_floating_point_dtype(),
            }
        )

        multi_transforms = self.create_multicrop_transform(
            config.multicrop_transform_list
        )

        self.global_transforms = multi_transforms[0 : self.global_num_crops]
        self.local_transforms = multi_transforms[
            self.global_num_crops : self.total_crops
        ]

    @property
    def output_keys(self):
        return self.config.output_keys

    def create_multicrop_transform(self, multicrop_transform_list):

        _transforms = []
        for _ in range(self.global_num_crops):
            _transforms.append(
                [
                    transforms.RandomResizedCrop(
                        self.global_image_size,
                        scale=self.global_crops_scale,
                        interpolation=self.interpolation,
                    )
                ]
            )

        for _ in range(self.local_num_crops):
            _transforms.append(
                [
                    transforms.RandomResizedCrop(
                        self.local_image_size,
                        scale=self.local_crops_scale,
                        interpolation=self.interpolation,
                    )
                ]
            )

        for crop_transform in multicrop_transform_list:  # per transform
            for i in range(self.total_crops):  # per crop
                kwargs = {"name": crop_transform["name"]}
                for k, val in crop_transform.items():
                    if k == "name":
                        continue
                    if isinstance(val, list) and len(val) == self.total_crops:
                        kwargs[k] = val[i]
                    else:
                        kwargs[k] = val

                tx = create_transform(kwargs)
                _transforms[i].append(tx)

        multi_transforms = []
        for _tx in _transforms:
            multi_transforms.append(
                transforms.Compose(_tx + [self.addnl_transform])
            )

        logging.debug(
            f"The following sequence is used to transform data:\n{multi_transforms}"
        )
        return multi_transforms

    def __call__(self, image):
        crops = []

        for global_tx in self.global_transforms:
            crops.append(global_tx(image))

        for local_tx in self.local_transforms:
            crops.append(local_tx(image))

        return crops

    def collate_fn(self, batch):
        #  batch: len of batch size of [([len(num_crops)], label)]
        data = {}

        global_views = []
        for i in range(self.global_num_crops):
            val = [images[i] for images, _ in batch]
            global_views.append(torch.stack(val))
        data["global_view"] = torch.stack(global_views, dim=1)

        local_view = []
        for i in range(self.local_num_crops):
            val = [images[i + self.global_num_crops] for images, _ in batch]
            local_view.append(torch.stack(val))
        data["local_view"] = torch.stack(local_view, dim=1)

        labels = [lbl for _, lbl in batch]
        data["labels"] = default_collate(labels)

        return data

    def visualize_transform(self, batch):
        batch_size = batch["global_view"].shape[0]
        image_dir = os.path.join(
            os.path.dirname(__file__), "visualize_ImageRandomMultiCropTransform"
        )
        logging.info(
            f"Batch visualization with `ImageRandomMultiCropTransform` saved at {image_dir}"
        )
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i in range(batch_size):
            img_path = os.path.join(image_dir, f"sample{i}_global_view.jpg")
            save_image(batch["global_view"][i], img_path)

            img_path = os.path.join(image_dir, f"sample{i}_local_view.jpg")
            save_image(batch["local_view"][i], img_path)
