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
from functools import partial

import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
)


class DSprites(VisionDataset):
    """
    dSprites is a dataset of 2D shapes procedurally generated from 6 ground
    truth independent latent factors. These factors are color, shape, scale,
    rotation, x and y positions of a sprite.

    All possible combinations of these latents are present exactly once,
    generating N = 737280 total images.

    ### Latent factor values
    *   Color: white
    *   Shape: square, ellipse, heart
    *   Scale: 6 values linearly spaced in [0.5, 1]
    *   Orientation: 40 values in [0, 2 pi]
    *   Position X: 32 values in [0, 1]
    *   Position Y: 32 values in [0, 1]

    We varied one latent at a time (starting from Position Y, then Position
    X, etc), and sequentially stored the images in fixed order. Hence the
    order along the first dimension is fixed and allows you to map back to
    the value of the latents corresponding to that image.

    We chose the latents values deliberately to have the smallest step
    changes while ensuring that all pixel outputs were different. No noise
    was added.
    """

    _file = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            os.path.join(root, "dsprites"),
            transform=transform,
            target_transform=target_transform,
        )
        if not os.path.exists(self.root):
            raise RuntimeError(
                f"Dataset not found. Download from "
                f"https://github.com/deepmind/dsprites-dataset/blob/master/{self._file}"
            )

        with h5py.File(os.path.join(self.root, self._file), "r") as fx:
            self.length = len(fx["imgs"])
        self.h5dataset = None
        self.images = None
        self.labels = None

    def __getitem__(self, index):
        # Workaround so that dataset is pickleable and allow for multiprocessing
        # See discussion: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.h5dataset is None:
            self.h5dataset = h5py.File(os.path.join(self.root, self._file), "r")
            self.images = self.h5dataset["imgs"]
            self.labels = self.h5dataset["latents"]["classes"]

        # image has shape (64, 64), expand and tile to (64, 64, 3)
        img = np.tile(np.expand_dims(self.images[index] * 255, -1), (1, 1, 3))
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


def _process_predicted_attribute_by_index(
    label_index, class_division_factor, label
):
    target = label[label_index]
    return np.floor(target / class_division_factor)


class DSpritesProcessor(Processor):
    _TASK_DICT = {
        "label_x_position": {
            "preprocess_fn": partial(_process_predicted_attribute_by_index, 4),
            "num_classes": 32,
        },
        "label_orientation": {
            "preprocess_fn": partial(_process_predicted_attribute_by_index, 3),
            "num_classes": 40,
        },
    }

    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train"]
        self.allowable_task = self._TASK_DICT.keys()
        self.num_classes = 16

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = DSprites(
            root=self.data_dir,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset

    def create_vtab_dataset(
        self,
        task="label_x_position",
        num_classes=16,
        use_1k_sample=True,
        seed=42,
    ):
        if task not in self.allowable_task:
            raise ValueError(
                f"Task {task} is not supported, choose from "
                f"{self.allowable_task} instead"
            )

        num_original_classes = self._TASK_DICT[task]["num_classes"]
        if num_classes is None:
            num_classes = num_original_classes
        if (
            not isinstance(num_classes, int)
            or num_classes <= 1
            or (num_classes > num_original_classes)
        ):
            raise ValueError(
                f"The number of classes should be None or in "
                f"[2, {num_original_classes}"
            )
        class_division_factor = float(num_original_classes) / num_classes
        target_transform = partial(
            self._TASK_DICT[task]["preprocess_fn"], class_division_factor
        )

        train_transform, train_tgt_transform = self.process_transform(
            use_training_transforms=True
        )
        eval_transform, eval_tgt_transform = self.process_transform(
            use_training_transforms=False
        )

        train_target_transform = transforms.Compose(
            [target_transform, train_tgt_transform]
        )
        eval_target_transform = transforms.Compose(
            [target_transform, eval_tgt_transform]
        )

        dataset = DSprites(
            root=self.data_dir,
            transform=None,
        )

        # DSprites only comes with a training set. Therefore, the training,
        # validation, and test sets are split out of the original training set.
        # By default, 80% is used as a new training split, 10% is used for
        # validation, and 10% is used for testing.
        split_percent = [80, 10, 10]
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
            eval_transform, target_transform=eval_target_transform
        )
        test_set.set_transforms(
            eval_transform, target_transform=eval_target_transform
        )

        return train_set, val_set, test_set
