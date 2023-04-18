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

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision import transforms

from modelzoo.vision.pytorch.unet.input.Hdf5BaseDataProcessor import (
    Hdf5BaseDataProcessor,
)
from modelzoo.vision.pytorch.unet.input.preprocessing_utils import (
    adjust_brightness_transform,
    normalize_tensor_transform,
    rotation_90_transform,
)


class Hdf5DataProcessor(Hdf5BaseDataProcessor):
    """
    A HDF5 dataset processor for UNet HDF dataset.
    Performs on-the-fly augmentation of image and labek.

    Functionality includes:
        Reading data from HDF5 documents
        Augmenting data

    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (str or list of str): Path to dataset HDF5 files
    - "num_classes (int): Maximum length of the sequence to generate
    - "image_shape" (int): Expected shape of output images and label, used in assert checks.
    - "loss" (str): Loss type, supported: {"bce", "multilabel_bce", "ssce"}
    - "normalize_data_method" (str): Can be one of {None, "zero_centered", "zero_one"}
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_buffer" (int): Size of shuffle buffer in samples.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of samples loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    """

    def _shard_files(self, is_training=False):
        # Features in HDF5 record files
        self.features_list = ["image", "label"]

        assert self.batch_size > 0, "Batch size should be positive."

        p = Path(self.data_dir)
        assert p.is_dir()

        files = sorted(p.glob('*.h5'))
        if not files:
            raise RuntimeError('No hdf5 datasets found')

        assert (
            len(files) >= self.num_tasks
        ), f"Number of h5 files {len(files)} should atleast be equal to the number of Slurm tasks {self.num_tasks}, to correctly shard the dataset between the streamers"

        # Shard H5 files between the tasks and resolve the paths
        files_in_this_task = [
            str(file.resolve())
            for file in files[self.task_id :: self.num_tasks]
        ]

        self.files_in_this_task = []
        self.num_examples_in_this_task = 0
        for file_path in files_in_this_task:
            with h5py.File(file_path, mode='r') as h5_file:
                num_examples_in_file = h5_file.attrs["n_examples"]
                self.files_in_this_task.append(
                    (file_path, num_examples_in_file)
                )
                self.num_examples_in_this_task += num_examples_in_file

        assert (
            self.num_examples_in_this_task >= self.num_workers * self.batch_size
        ), f"The number of examples on this worker={self.num_examples_in_this_task} is lesser than batch size(={self.batch_size}) * num_workers(={self.num_workers}). Please consider reducing the number of workers (or) increasing the number of samples in files (or) reducing the batch size"

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(self.files_in_this_task)

    def _apply_normalization(self, x):
        return normalize_tensor_transform(
            x, normalize_data_method=self.normalize_data_method
        )

    def _load_buffer(self, data_partitions):
        for file_path, start_idx, num_examples in data_partitions:
            with h5py.File(file_path, mode='r') as h5_file:
                for idx in range(start_idx, start_idx + num_examples):
                    yield h5_file[f"example_{idx}"]

    def _shard_dataset(self, worker_id, num_workers):
        per_worker_partition = []
        for file, num_examples_in_file in self.files_in_this_task:
            # Try to evenly distribute number of examples between workers
            num_examples_all_workers = [
                (num_examples_in_file // num_workers)
            ] * num_workers
            for i in range(num_examples_in_file % num_workers):
                num_examples_all_workers[i] += 1

            assert sum(num_examples_all_workers) == num_examples_in_file

            per_worker_partition.append(
                (
                    file,
                    sum(num_examples_all_workers[:worker_id])
                    if worker_id > 0
                    else 0,  # Start index
                    num_examples_all_workers[worker_id],  # Length of data chunk
                )
            )

        return per_worker_partition

    def __iter__(self):
        """
        Iterating over the data to construct input features.
        """
        for example in self._load_buffer(self.data_partitions):
            example_dict = {}
            for idx, feature in enumerate(self.features_list):
                example_dict[feature] = torch.from_numpy(
                    np.array(example[feature])
                )
            image, label = self.transform_image_and_mask(
                example_dict["image"], example_dict["label"]
            )

            yield image, label

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

        if self.mixed_precision:
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
