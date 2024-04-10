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
import torch
import torchvision

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
    VisionSubset,
)


class Flowers102Processor(Processor):
    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train", "val", "test"]
        self.num_classes = 102

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = torchvision.datasets.Flowers102(
            root=self.data_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        return dataset

    def create_vtab_dataset(
        self, use_1k_sample=True, train_split_percent=None, seed=42
    ):
        train_transform, train_target_transform = self.process_transform(
            use_training_transforms=True
        )
        eval_transform, eval_target_transform = self.process_transform(
            use_training_transforms=False
        )

        train_set = torchvision.datasets.Flowers102(
            root=self.data_dir,
            split="train",
            transform=train_transform,
            target_transform=train_target_transform,
            download=False,
        )
        val_set = torchvision.datasets.Flowers102(
            root=self.data_dir,
            split="val",
            transform=eval_transform,
            target_transform=eval_target_transform,
            download=False,
        )
        test_set = torchvision.datasets.Flowers102(
            root=self.data_dir,
            split="test",
            transform=eval_transform,
            target_transform=eval_target_transform,
            download=False,
        )

        rng = np.random.default_rng(seed)
        train_sample_idx = self.create_shuffled_idx(len(train_set), rng)
        val_sample_idx = self.create_shuffled_idx(len(val_set), rng)

        if use_1k_sample:
            train_set = VisionSubset(train_set, train_sample_idx[:800])
            val_set = VisionSubset(val_set, val_sample_idx[:200])
            return train_set, val_set, test_set
        else:
            if train_split_percent:
                # if train_split_percent is specified, the training set and
                # validation set are combined and then split according to the
                # specified percentage.
                split_percent = [train_split_percent, 100 - train_split_percent]
                train_set_splits = self.split_dataset(
                    train_set, split_percent, seed
                )
                val_set_splits = self.split_dataset(
                    val_set, split_percent, seed
                )

                # update transform so that the split dataset at index 0 uses
                # training transform while those at index 1 uses eval transform
                train_set_splits[0].set_transforms(train_transform)
                val_set_splits[0].set_transforms(train_transform)
                train_set_splits[1].set_transforms(eval_transform)
                val_set_splits[1].set_transforms(eval_transform)

                new_train_set = torch.utils.data.ConcatDataset(
                    [train_set_splits[0], val_set_splits[0]]
                )
                new_val_set = torch.utils.data.ConcatDataset(
                    [train_set_splits[1], val_set_splits[1]]
                )
                return new_train_set, new_val_set, test_set
            else:
                return train_set, val_set, test_set
