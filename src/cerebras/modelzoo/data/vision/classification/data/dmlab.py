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

import numpy as np
from PIL import Image
from pydantic import Field
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)


class Dmlab(VisionDataset):
    """
    The Dmlab dataset contains frames observed by the agent acting in the DMLab
    environment, which are annotated by the distance between the agent and
    various objects present in the environment. The goal is to is to evaluate
    the ability of a visual model to reason about distances from the visual
    input in 3D environments.

    The Dmlab dataset consists of 360x480 color images in 6 classes. The classes
    are {close, far, very far} x {positive reward, negative reward} respectively.
    """

    def __init__(
        self, root, transform=None, target_transform=None, split="train"
    ):
        super().__init__(
            os.path.join(root, "dmlab"),
            transform=transform,
            target_transform=target_transform,
        )
        self.split = split if split != "val" else "validation"
        if not os.path.exists(os.path.join(self.root, self.split)):
            raise RuntimeError(
                "Dataset not found. Download from "
                "https://storage.googleapis.com/dmlab-vtab/dmlab.tar.gz"
            )

        self.categories = sorted(
            os.listdir(os.path.join(self.root, self.split))
        )

        self.img_files = []
        self.y = []
        for c in self.categories:
            imgs = os.listdir(os.path.join(self.root, self.split, c))
            self.img_files.extend(imgs)
            self.y.extend(len(imgs) * [c])

        if len(self.img_files) != len(self.y):
            raise RuntimeError(f"Image file length is different than labels")

    def __getitem__(self, index):
        img_file = os.path.join(
            self.root, self.split, self.y[index], self.img_files[index]
        )
        img = Image.fromarray(np.load(img_file).astype('uint8'), 'RGB')
        target = float(self.y[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_files)


class DmlabProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["DmlabProcessor"]

    use_worker_cache: bool = ...

    split: Literal["train", "val", "test"] = "train"
    "Dataset split."

    num_classes: Optional[Any] = Field(None, deprecated=True)


class DmlabProcessor(VisionClassificationProcessor):
    def __init__(self, config: DmlabProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")
        self.num_classes = 6

    def create_dataset(self):
        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = Dmlab(
            root=self.data_dir,
            transform=transform,
            target_transform=target_transform,
            split=self.split,
        )
        return dataset
