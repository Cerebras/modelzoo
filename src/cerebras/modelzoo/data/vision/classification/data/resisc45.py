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

from PIL import Image
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
)


class Resisc45(VisionDataset):
    """
    RESISC45 dataset is a publicly available benchmark for Remote Sensing Image
    Scene Classification (RESISC), created by Northwestern Polytechnical
    University (NWPU). This dataset contains 31,500 images, covering 45 scene
    classes with 700 images in each class.
    URL: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html
    """

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(
            os.path.join(root, "resisc45"),
            transform=transform,
            target_transform=target_transform,
        )
        if not os.path.exists(self.root):
            raise RuntimeError(
                "Dataset not found. Download and extract from "
                "https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs"
            )

        self.categories = sorted(os.listdir(self.root))

        self.index = []
        self.y = []
        for i, c in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        category = self.categories[self.y[index]]
        img = Image.open(
            os.path.join(
                self.root,
                category,
                f"{category}_{self.index[index]:03d}.jpg",
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


class Resisc45Processor(Processor):
    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train"]
        self.num_classes = 45

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = Resisc45(
            root=self.data_dir,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset

    def create_vtab_dataset(self, use_1k_sample=True, seed=42):
        train_transform, train_target_transform = self.process_transform(
            use_training_transforms=True
        )
        eval_transform, eval_target_transform = self.process_transform(
            use_training_transforms=False
        )

        dataset = Resisc45(root=self.data_dir, transform=None)

        # Resisc45 only comes with a training set. Therefore, the training,
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
