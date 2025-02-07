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

import json
import os
from typing import Any, Literal, Optional

from PIL import Image
from pydantic import Field
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)


class CLEVR(VisionDataset):
    """
    CLEVR is a diagnostic dataset that tests a range of visual reasoning
    abilities. It contains minimal biases and has detailed annotations
    describing the kind of reasoning each question requires.
    """

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            os.path.join(root, "clevr", "CLEVR_v1.0"),
            transform=transform,
            target_transform=target_transform,
        )
        if not os.path.exists(self.root):
            raise RuntimeError(
                "Dataset not found. Download and extract from "
                "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
            )
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))

        self._image_files = sorted(
            os.listdir(os.path.join(self.root, "images", self._split))
        )
        self._image_files = []
        self.labels = []
        if self._split != "test":
            with open(
                os.path.join(
                    self.root, "scenes", f"CLEVR_{self._split}_scenes.json"
                )
            ) as file:
                content = json.load(file)

            for scene in content["scenes"]:
                self._image_files.append(scene["image_filename"])
                self.labels.append(scene["objects"])
        else:
            self._image_files = sorted(
                os.listdir(os.path.join(self.root, "images", self._split))
            )
            # The CLEVR dataset does not have labels (answers) to the test set.
            self.labels = [[]] * len(self._image_files)

    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        label = self.labels[idx]

        image = Image.open(
            os.path.join(self.root, "images", self._split, image_file)
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self._image_files)


class CLEVRProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["CLEVRProcessor"]

    use_worker_cache: bool = ...

    split: Literal["train", "val", "test"] = "train"
    "Dataset split."

    num_classes: Optional[Any] = Field(None, deprecated=True)


class CLEVRProcessor(VisionClassificationProcessor):
    def __init__(self, config: CLEVRProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")

    def create_dataset(self):
        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = CLEVR(
            root=self.data_dir,
            split=self.split,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset
