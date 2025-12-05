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


class Caltech101(VisionDataset):
    """
    Caltech-101 consists of pictures of objects belonging to 101 classes, plus
    one `background clutter` class. Each image is labelled with a single object.
    Each class contains roughly 40 to 800 images, totalling around 9k images.
    Images are of variable sizes, with typical edge lengths of 200-300 pixels.
    This version contains image-level labels only. The original dataset also
    contains bounding boxes.

    This version also adds the option to split Caltech-101 into trainval set and
    test set.The trainval set is classed balanced with <class_balance_count>
    random samples for each of the 101 classes. The remainder are added to the
    test set.
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        split=None,
        class_balance_count=None,
    ):
        super().__init__(
            os.path.join(root, "caltech101"),
            transform=transform,
            target_transform=target_transform,
        )
        if not os.path.exists(os.path.join(self.root, "101_ObjectCategories")):
            raise RuntimeError(
                "Dataset not found or corrupted. Download and extract from "
                "https://data.caltech.edu/records/20086"
            )

        self.categories = sorted(
            os.listdir(os.path.join(self.root, "101_ObjectCategories"))
        )
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        self.index = []
        self.y = []
        for i, c in enumerate(self.categories):
            n = len(
                os.listdir(os.path.join(self.root, "101_ObjectCategories", c))
            )
            if class_balance_count is not None:
                numpy_original_state = np.random.get_state()
                np.random.seed(1234)
                trainval_index = np.random.choice(
                    range(1, n + 1), class_balance_count, replace=False
                )
                np.random.set_state(numpy_original_state)

                test_index = [
                    idx for idx in range(1, n + 1) if idx not in trainval_index
                ]
                if split == "trainval":
                    self.index.extend(trainval_index)
                    self.y.extend(class_balance_count * [i])
                else:  # "test"
                    self.index.extend(test_index)
                    self.y.extend((n - class_balance_count) * [i])
            else:
                self.index.extend(range(1, n + 1))
                self.y.extend(n * [i])

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        )
        img = img.convert("RGB")
        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.index)


class Caltech101ProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["Caltech101Processor"]

    split: Literal["trainval", "test"] = "trainval"
    "Dataset split."

    num_classes: Optional[Any] = Field(None, deprecated=True)


class Caltech101Processor(VisionClassificationProcessor):
    def __init__(self, config: Caltech101ProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")

        self.num_classes = 101

    def create_dataset(self):
        use_training_transforms = self.split == "trainval"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = Caltech101(
            root=self.data_dir,
            transform=transform,
            target_transform=target_transform,
            split=None,
        )
        return dataset
