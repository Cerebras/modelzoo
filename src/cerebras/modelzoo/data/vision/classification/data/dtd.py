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

import numpy as np
import torchvision

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
    VisionSubset,
)


class DTDProcessor(Processor):
    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train", "val", "test"]
        self.num_classes = 47

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = torchvision.datasets.DTD(
            root=self.data_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        return dataset

    def create_vtab_dataset(self, use_1k_sample=True, seed=42):
        train_transform, train_target_transform = self.process_transform(
            use_training_transforms=True
        )
        eval_transform, eval_target_transform = self.process_transform(
            use_training_transforms=False
        )

        train_set = torchvision.datasets.DTD(
            root=self.data_dir,
            split="train",
            transform=train_transform,
            target_transform=train_target_transform,
            download=False,
        )
        val_set = torchvision.datasets.DTD(
            root=self.data_dir,
            split="val",
            transform=eval_transform,
            target_transform=eval_target_transform,
            download=False,
        )
        test_set = torchvision.datasets.DTD(
            root=self.data_dir,
            split="test",
            transform=eval_transform,
            target_transform=eval_target_transform,
            download=False,
        )

        if use_1k_sample:
            rng = np.random.default_rng(seed)
            sample_idx = self.create_shuffled_idx(len(train_set), rng)
            train_set = VisionSubset(train_set, sample_idx[:800])

            sample_idx = self.create_shuffled_idx(len(val_set), rng)
            val_set = VisionSubset(val_set, sample_idx[:200])

        return train_set, val_set, test_set
