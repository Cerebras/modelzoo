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

import torchvision

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
)


class StanfordCarsProcessor(Processor):
    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train", "test"]
        self.num_classes = 196

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = torchvision.datasets.StanfordCars(
            root=self.data_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        return dataset
