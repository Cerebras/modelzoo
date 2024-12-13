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

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import cerebras.pytorch as cstorch
from cerebras.modelzoo.data.vision.segmentation.Hdf5BaseDataProcessor import (
    Hdf5BaseDataProcessor,
    Hdf5BaseDataProcessorConfig,
)
from cerebras.modelzoo.data.vision.segmentation.preprocessing_utils import (
    adjust_brightness_transform,
    normalize_tensor_transform,
    rotation_90_transform,
)


class Hdf5DataProcessorConfig(Hdf5BaseDataProcessorConfig):
    data_processor: Literal["Hdf5DataProcessor"]

    split: Literal["train", "val"] = "train"


class Hdf5DataProcessor(Hdf5BaseDataProcessor):
    def __init__(self, config: Hdf5DataProcessorConfig):
        super(Hdf5DataProcessor, self).__init__(config)
        self.split = config.split

    def _shard_files(self, is_training=False):
        # Features in HDF5 record files
        self.features_list = ["image", "label"]

        assert self.batch_size > 0, "Batch size should be positive."

        p = Path(self.data_dir)
        assert p.is_dir()

        files = sorted(p.glob('*.h5'))
        if not files:
            raise RuntimeError('No hdf5 datasets found')

        all_files = [str(file.resolve()) for file in files]

        self.all_files = []
        self.files_in_this_task = []
        self.num_examples = 0
        self.num_examples_in_this_task = 0
        for idx, file_path in enumerate(all_files):
            with h5py.File(file_path, mode='r') as h5_file:
                num_examples_in_file = h5_file.attrs["n_examples"]
                file_details = (file_path, num_examples_in_file)
                self.all_files.append(file_details)
                self.num_examples += num_examples_in_file
                if idx % self.num_tasks == self.task_id:
                    self.files_in_this_task.append(file_details)
                    self.num_examples_in_this_task += num_examples_in_file

        # Prevent CoW which is effectively copy on read behavior for PT,
        # see: https://github.com/pytorch/pytorch/issues/13246
        self.all_files = pd.DataFrame(
            self.all_files, columns=["file_path", "num_examples_in_file"]
        )
        self.files_in_this_task = pd.DataFrame(
            self.files_in_this_task,
            columns=["file_path", "num_examples_in_file"],
        )

    def _apply_normalization(self, x):
        return normalize_tensor_transform(
            x, normalize_data_method=self.normalize_data_method
        )

    def _load_buffer(self, data_partitions):
        for file_path, start_idx, num_examples in data_partitions:
            with h5py.File(file_path, mode='r') as h5_file:
                for idx in range(start_idx, start_idx + num_examples):
                    yield h5_file[f"example_{idx}"]

    def _maybe_shard_dataset(self, num_workers):
        per_worker_partition = {}
        idx = 0
        files = (
            self.all_files if self.disable_sharding else self.files_in_this_task
        )
        for _, row in files.iterrows():
            # Try to evenly distribute number of examples between workers
            file_path = row["file_path"]
            num_examples_in_file = row["num_examples_in_file"]
            num_examples_all_workers = [
                (num_examples_in_file // num_workers)
            ] * num_workers
            for i in range(num_examples_in_file % num_workers):
                num_examples_all_workers[i] += 1
            assert sum(num_examples_all_workers) == num_examples_in_file

            for file_idx in range(num_examples_in_file):
                per_worker_partition[idx] = (file_path, f"example_{file_idx}")
                idx += 1
        return per_worker_partition

    def __len__(self):
        if self.disable_sharding:
            return self.num_examples
        else:
            return self.num_examples_in_this_task

    def __getitem__(self, index):
        """Get item at a particular index"""
        file_path, sample_name = self.data_partitions[index]
        example_dict = {}
        with h5py.File(file_path, mode='r') as h5_file:
            example = h5_file[sample_name]
            for _, feature in enumerate(self.features_list):
                example_dict[feature] = torch.from_numpy(
                    np.array(example[feature])
                )
            image, label = self.transform_image_and_mask(
                example_dict["image"], example_dict["label"]
            )
        return image, label

    def transform_image_and_mask(self, image, mask):
        if self.normalize_data_method:
            image = self.normalize_transform(image)

        if self.augment_data:
            do_horizontal_flip = torch.rand(size=(1,)).item() > 0.5
            # n_rots in range [0, 3)
            n_rotations = torch.randint(low=0, high=3, size=(1,)).item()

            if self.tgt_image_height != self.tgt_image_width:
                # For a rectangle image
                n_rotations = n_rotations * 2

            augment_transform_image = self.get_augment_transforms(
                do_horizontal_flip=do_horizontal_flip,
                n_rotations=n_rotations,
                do_random_brightness=True,
            )
            augment_transform_mask = self.get_augment_transforms(
                do_horizontal_flip=do_horizontal_flip,
                n_rotations=n_rotations,
                do_random_brightness=False,
            )

            image = augment_transform_image(image)
            mask = augment_transform_mask(mask)

        # Handle dtypes and mask shapes based on `loss_type`
        # and `mixed_precsion`
        if self.loss_type == "bce":
            mask = mask.to(self.mp_type)
        elif self.loss_type == "multilabel_bce":
            mask = torch.squeeze(mask, 0)
            # Only long tensors are accepted by one_hot fcn.
            mask = mask.to(torch.long)

            # out shape: (H, W, num_classes)
            mask = torch.nn.functional.one_hot(
                mask, num_classes=self.num_classes
            )
            # out shape: (num_classes, H, W)
            mask = torch.permute(mask, [2, 0, 1])
            mask = mask.to(self.mp_type)

        elif self.loss_type == "ssce":
            # out shape: (H, W) with each value in [0, num_classes)
            mask = torch.squeeze(mask, 0)
            # TODO: Add MZ tags here when supported.
            # SW-82348 workaround: Pass `labels` in `int32``
            # PT crossentropy loss takes in `int64`,
            # view and typecast does not change the orginal `labels`.
            mask = mask.to(torch.int32)

        if cstorch.amp.mixed_precision():
            image = image.to(self.mp_type)

        return image, mask

    def get_augment_transforms(
        self, do_horizontal_flip, n_rotations, do_random_brightness
    ):
        augment_transforms_list = []
        if do_horizontal_flip:
            horizontal_flip_transform = transforms.Lambda(
                lambda x: transforms.functional.hflip(x)
            )
            augment_transforms_list.append(horizontal_flip_transform)

        if n_rotations > 0:
            rotation_transform = transforms.Lambda(
                lambda x: rotation_90_transform(x, num_rotations=n_rotations)
            )
            augment_transforms_list.append(rotation_transform)

        if do_random_brightness:
            brightness_transform = transforms.Lambda(
                lambda x: adjust_brightness_transform(x, p=0.5, delta=0.2)
            )
            augment_transforms_list.append(brightness_transform)

        return transforms.Compose(augment_transforms_list)
