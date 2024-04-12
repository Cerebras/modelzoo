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

import numpy as np
from PIL import Image
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
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


def _count_objects(target):
    """
    Predict the number of objects in the scene. Since the number of objects
    ranges from [3, 10], we subtract 3 to make class labels [0, 7]
    """
    return len(target) - 3


def _closest_object_distance(target):
    """
    Predict distance to the closest object in the scene. We bin the distances
    such that the distribution of classes is more or less balanced.
    """
    if len(target) == 0:
        return -1
    else:
        thresholds = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
        # pixel_coords is a list of 3 numbers (x, y, z) of a given object in the
        # scene. Index 2 corresponds to the z-axis value.
        dist = np.min(
            [target[i]["pixel_coords"][2] for i in range(len(target))]
        )
        return np.max(np.where(thresholds - dist < 0))


class CLEVRProcessor(Processor):
    _TASK_DICT = {
        "count": {"preprocess_fn": _count_objects, "num_classes": 8},
        "distance": {
            "preprocess_fn": _closest_object_distance,
            "num_classes": 6,
        },
    }

    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train", "val", "test"]
        self.allowable_task = self._TASK_DICT.keys()

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = CLEVR(
            root=self.data_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset

    def create_vtab_dataset(
        self,
        task="count",
        use_1k_sample=True,
        train_split_percent=None,
        seed=42,
    ):
        if task not in self.allowable_task:
            raise ValueError(
                f"Task {task} is not supported, choose from "
                f"{self.allowable_task} instead"
            )

        train_transform, train_target_transform = self.process_transform(
            use_training_transforms=True
        )
        eval_transform, eval_target_transform = self.process_transform(
            use_training_transforms=False
        )

        trainval_set = CLEVR(
            root=self.data_dir,
            split="train",
            transform=None,
            target_transform=self._TASK_DICT[task]["preprocess_fn"],
        )
        test_set = CLEVR(
            root=self.data_dir,
            split="test",
            transform=eval_transform,
            target_transform=self._TASK_DICT[task]["preprocess_fn"],
        )

        # By default, 90% of the official training split is used as a new
        # training split and the rest is used for validation
        train_percent = train_split_percent or 90
        val_percent = 100 - train_percent
        train_set, val_set = self.split_dataset(
            trainval_set, [train_percent, val_percent], seed
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

        return train_set, val_set, test_set
