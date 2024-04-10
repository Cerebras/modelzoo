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

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
    VisionSubset,
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


class DmlabProcessor(Processor):
    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train", "val", "test"]
        self.num_classes = 6

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = Dmlab(
            root=self.data_dir,
            transform=transform,
            target_transform=target_transform,
            split=split,
        )
        return dataset

    def create_vtab_dataset(self, use_1k_sample=True, seed=42):
        train_transform, train_target_transform = self.process_transform(
            use_training_transforms=True
        )
        eval_transform, eval_target_transform = self.process_transform(
            use_training_transforms=False
        )

        train_set = Dmlab(
            root=self.data_dir,
            transform=train_transform,
            target_transform=train_target_transform,
            split="train",
        )
        val_set = Dmlab(
            root=self.data_dir,
            transform=eval_transform,
            target_transform=eval_target_transform,
            split="val",
        )
        test_set = Dmlab(
            root=self.data_dir,
            transform=eval_transform,
            target_transform=eval_target_transform,
            split="test",
        )

        if use_1k_sample:
            rng = np.random.default_rng(seed)
            sample_idx = self.create_shuffled_idx(len(train_set), rng)
            train_set = VisionSubset(train_set, sample_idx[:800])

            sample_idx = self.create_shuffled_idx(len(val_set), rng)
            val_set = VisionSubset(val_set, sample_idx[:200])

        return train_set, val_set, test_set
