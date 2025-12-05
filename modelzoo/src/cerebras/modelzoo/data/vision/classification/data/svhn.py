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

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)


class SVHNProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["SVHNProcessor"]

    use_worker_cache: bool = ...

    split: Literal["train", "test", "extra"] = "train"
    "Dataset split."

    num_classes: Optional[Any] = Field(None, deprecated=True)


class SVHNProcessor(VisionClassificationProcessor):
    def __init__(self, config: SVHNProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")
        self.num_classes = 10

    def create_dataset(self):
        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = torchvision.datasets.SVHN(
            root=os.path.join(self.data_dir, "SVHN"),
            split=self.split,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        return dataset
