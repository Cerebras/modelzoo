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


class EuroSATProcessor(Processor):
    """RGB version of the EuroSAT Dataset (num channel = 3)"""

    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train"]
        self.num_classes = 10

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = torchvision.datasets.EuroSAT(
            root=self.data_dir,
            transform=transform,
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

        dataset = torchvision.datasets.EuroSAT(
            root=self.data_dir,
            transform=None,
            download=False,
        )

        # EuroSAT only comes with a training set. Therefore, the training,
        # validation, and test sets are split out of the original training set.
        # By default, 60% is used as a new training split, 20% is used for
        # validation, and 20% is used for testing.
        split_percent = [60, 20, 20]
        train_set, val_set, test_set = self.split_dataset(
            dataset, split_percent, seed
        )

        if use_1k_sample:
            train_set.truncate_to_idx(800)
            val_set.truncate_to_idx(200)

        train_set.set_transforms(
            transform=train_transform, target_transform=train_target_transform
        )
        val_set.set_transforms(
            transform=eval_transform, target_transform=eval_target_transform
        )
        test_set.set_transforms(
            transform=eval_transform, target_transform=eval_target_transform
        )

        return train_set, val_set, test_set
