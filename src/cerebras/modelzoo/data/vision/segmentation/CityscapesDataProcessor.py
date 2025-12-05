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

from typing import List, Literal

import torch
from torchvision import datasets, transforms

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.data.vision.segmentation.UNetDataProcessor import (
    UNetDataProcessor,
    UNetDataProcessorConfig,
)
from cerebras.modelzoo.data.vision.utils import create_worker_cache


class Cityscapes(datasets.Cityscapes):
    """Wrapper around torchvision.datasets.Cityscapes with sorted files for reproducibility"""

    def __init__(self, use_worker_cache=False, **kwargs):
        super(Cityscapes, self).__init__(**kwargs)
        if use_worker_cache and dist.is_streamer():
            if not cstorch.use_cs():
                raise RuntimeError(
                    "use_worker_cache not supported for non-CS runs"
                )
            else:
                self.root = create_worker_cache(self.root)

        self.images, self.targets = zip(*sorted(zip(self.images, self.targets)))


class CityscapesDataProcessorConfig(UNetDataProcessorConfig):
    data_processor: Literal["CityscapesDataProcessor"]

    use_worker_cache: bool = False

    max_image_shape: List[int] = [1024, 2048]

    split: Literal["train", "val"] = "train"
    "Dataset split."


class CityscapesDataProcessor(UNetDataProcessor):
    def __init__(self, config: CityscapesDataProcessorConfig):
        super(CityscapesDataProcessor, self).__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")

        self.use_worker_cache = config.use_worker_cache
        self.image_shape = config.image_shape  # of format (H, W, C)
        self._tiling_image_shape = self.image_shape  # out format: (H, W, C)

        # Tiling param:
        # If `image_shape` < 1K x 2K, do not tile.
        # If `image_shape` > 1K x 2K in any dimension,
        #   first resize image to min(img_shape, max_image_shape)
        #   and then tile to target height and width specified in yaml
        self.max_image_shape = config.max_image_shape

        self.image_shape = self._update_image_shape()
        (
            self.tgt_image_height,
            self.tgt_image_width,
            self.channels,
        ) = self.image_shape

    def _update_image_shape(self):

        # image_shape is of format (H, W, C)
        image_shape = []

        for i in range(2):
            image_shape.append(
                min(self.image_shape[i], self.max_image_shape[i])
            )
        image_shape = (
            image_shape + self.image_shape[-1:]
        )  # Output shape format (H, W, C)

        return image_shape

    def create_dataset(self):
        dataset = Cityscapes(
            root=self.data_dir,
            split=self.split,
            mode="fine",
            target_type="semantic",
            transforms=self.transform_image_and_mask,
            use_worker_cache=self.use_worker_cache,
        )
        return dataset

    def preprocess_mask(self, mask):

        # Refer to :
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L56-L99
        # Mapping all classes with `ignoreInEval`=True(from above link)
        # to background class with id 0

        def lookup_table(mask):
            # fmt: off
            lut = torch.tensor([
                0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
                16, 0, 0, 17, 18, 19,
                ],
                dtype=torch.uint8,
            )
            # fmt: on
            return lut[mask]

        # Resize
        resize_pil_transform = transforms.Resize(
            [self.tgt_image_height, self.tgt_image_width],
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        # converts to (C, H, W) format.
        to_tensor_transform = transforms.PILToTensor()

        # Convert to long for lookup
        convert_to_long_transform = transforms.Lambda(
            lambda x: x.to(torch.long)
        )

        # Map target ids based on lookup table
        lookup_table_transform = transforms.Lambda(lambda x: lookup_table(x))

        # Convert to mp type
        convert_to_mp_type_transform = transforms.Lambda(
            lambda x: x.to(self.mp_type)
        )

        tile_transform = self.get_tile_transform()

        transforms_list = [
            resize_pil_transform,
            to_tensor_transform,
            convert_to_long_transform,
            lookup_table_transform,
            convert_to_mp_type_transform,
            tile_transform,
        ]

        mask = transforms.Compose(transforms_list)(mask)

        return mask
