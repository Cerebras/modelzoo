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

import os
from typing import Any, Literal, Optional

import h5py
import numpy as np
from PIL import Image
from pydantic import Field
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)


class DSprites(VisionDataset):
    """
    dSprites is a dataset of 2D shapes procedurally generated from 6 ground
    truth independent latent factors. These factors are color, shape, scale,
    rotation, x and y positions of a sprite.

    All possible combinations of these latents are present exactly once,
    generating N = 737280 total images.

    ### Latent factor values
    *   Color: white
    *   Shape: square, ellipse, heart
    *   Scale: 6 values linearly spaced in [0.5, 1]
    *   Orientation: 40 values in [0, 2 pi]
    *   Position X: 32 values in [0, 1]
    *   Position Y: 32 values in [0, 1]

    We varied one latent at a time (starting from Position Y, then Position
    X, etc), and sequentially stored the images in fixed order. Hence the
    order along the first dimension is fixed and allows you to map back to
    the value of the latents corresponding to that image.

    We chose the latents values deliberately to have the smallest step
    changes while ensuring that all pixel outputs were different. No noise
    was added.
    """

    _file = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            os.path.join(root, "dsprites"),
            transform=transform,
            target_transform=target_transform,
        )
        if not os.path.exists(self.root):
            raise RuntimeError(
                f"Dataset not found. Download from "
                f"https://github.com/deepmind/dsprites-dataset/blob/master/{self._file}"
            )

        with h5py.File(os.path.join(self.root, self._file), "r") as fx:
            self.length = len(fx["imgs"])
        self.h5dataset = None
        self.images = None
        self.labels = None

    def __getitem__(self, index):
        # Workaround so that dataset is pickleable and allow for multiprocessing
        # See discussion: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.h5dataset is None:
            self.h5dataset = h5py.File(os.path.join(self.root, self._file), "r")
            self.images = self.h5dataset["imgs"]
            self.labels = self.h5dataset["latents"]["classes"]

        # image has shape (64, 64), expand and tile to (64, 64, 3)
        img = np.tile(np.expand_dims(self.images[index] * 255, -1), (1, 1, 3))
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


class DSpritesProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["DSpritesProcessor"]

    use_worker_cache: bool = ...

    split: Literal["train"] = "train"
    "Dataset split."

    num_classes: Optional[Any] = Field(None, deprecated=True)


class DSpritesProcessor(VisionClassificationProcessor):
    def __init__(self, config: DSpritesProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")
        self.num_classes = 16

    def create_dataset(self):
        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = DSprites(
            root=self.data_dir,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset
