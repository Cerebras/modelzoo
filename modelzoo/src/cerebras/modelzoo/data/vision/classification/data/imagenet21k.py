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
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)


class ImageNet21K(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )
        verify_str_arg(split, "split", ("train", "val"))

        database_path = os.path.join(
            self.root, f"ILSVRC2021winter_whole_map_{split}.txt"
        )
        self.database = json.load(open(database_path))

    def loader(self, path: str):
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index: int):
        filename, target = self.database[index]
        image = self.loader(os.path.join(self.root, filename))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


class ImageNet21KProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["ImageNet21KProcessor"]

    use_worker_cache: bool = ...

    split: Literal["train", "val"] = "train"
    "Dataset split."

    use_fake_data: Optional[Any] = Field(None, deprecated=True)
    num_classes: Optional[Any] = Field(None, deprecated=True)


class ImageNet21KProcessor(VisionClassificationProcessor):
    def __init__(self, config: ImageNet21KProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")
        self.num_classes = 19167

    def create_dataset(self):
        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )

        database_file = f"ILSVRC2021winter_whole_map_{self.split}.txt"
        if not os.path.isfile(os.path.join(self.data_dir, database_file)):
            raise RuntimeError(
                f"The database file is not present in the root directory. Check "
                "https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md "
                "for more details on preparing the dataset."
            )

        dataset = ImageNet21K(
            root=self.data_dir,
            split=self.split,
            transform=transform,
            target_transform=target_transform,
        )

        return dataset
