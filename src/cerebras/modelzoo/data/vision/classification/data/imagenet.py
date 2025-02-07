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

import torchvision
from pydantic import Field

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)
from cerebras.modelzoo.data.vision.utils import create_worker_cache


class ImageNet1KProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["ImageNet1KProcessor"]

    use_worker_cache: bool = False

    split: Literal["train", "val"] = "train"
    "Dataset split."

    use_fake_data: Optional[Any] = Field(None, deprecated=True)
    num_classes: Optional[Any] = Field(None, deprecated=True)


class ImageNet1KProcessor(VisionClassificationProcessor):
    def __init__(self, config: ImageNet1KProcessorConfig):
        if isinstance(config, dict):
            config = ImageNet1KProcessorConfig(**config)
        super().__init__(config)
        self.use_worker_cache = config.use_worker_cache
        self.split = config.split

        self.shuffle = self.shuffle and (self.split == "train")
        self.num_classes = 1000

    def create_dataset(self):
        if self.use_worker_cache and dist.is_streamer():
            if not cstorch.use_cs():
                raise RuntimeError(
                    "use_worker_cache not supported for non-CS runs"
                )
            else:
                self.data_dir = create_worker_cache(self.data_dir)

        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )

        if not os.path.isfile(os.path.join(self.data_dir, "meta.bin")):
            raise RuntimeError(
                "The meta file meta.bin is not present in the root directory. "
                "Check vision/pytorch/input/classification/data/README.md for "
                "more details on downloading the dataset."
            )

        if not os.path.isdir(os.path.join(self.data_dir, self.split)):
            raise RuntimeError(
                f"No directory {self.split} under root dir. Refer to "
                "vision/pytorch/input/classification/data/README.md on how to "
                "prepare the dataset."
            )

        dataset = torchvision.datasets.ImageNet(
            root=self.data_dir,
            split=self.split,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset
