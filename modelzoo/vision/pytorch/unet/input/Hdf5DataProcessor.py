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

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.utils import BufferedShuffleDataset
from modelzoo.vision.pytorch.unet.input.preprocessing_utils import (
    adjust_brightness_transform,
    normalize_tensor_transform,
    rotation_90_transform,
)


class Hdf5DataProcessor(torch.utils.data.IterableDataset):
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

    def __init__(self, params):
        super(Hdf5DataProcessor, self).__init__()

        self.data_dir = params["data_dir"]
        self.num_classes = params["num_classes"]
        self.normalize_data_method = params.get("normalize_data_method")
        if self.normalize_data_method:
            # Normalize
            self.normalize_transform = transforms.Lambda(
                self._apply_normalization
            )

        self.image_shape = params["image_shape"]  # of format (H, W, C)
        (
            self.tgt_image_height,
            self.tgt_image_width,
            self.channels,
        ) = self.image_shape

        # TODO: copy loss param from model section in utils.py
        self.loss_type = params["loss"]

        self.shuffle_seed = params.get("shuffle_seed", None)
        if self.shuffle_seed:
            torch.manual_seed(self.shuffle_seed)

        self.augment_data = params.get("augment_data", True)
        self.batch_size = params["batch_size"]
        self.shuffle = params.get("shuffle", True)

        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)

        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

        # TODO: copy `mixed_precison` param from model section in utils.py
        self.mixed_precision = params.get("mixed_precision")
        if self.mixed_precision:
            self.mp_type = torch.float16
        else:
            self.mp_type = torch.float32

        # Features in HDF5 record files
        self.features_list = ["image", "label"]

        assert self.batch_size > 0, "Batch size should be positive."

        p = Path(self.data_dir)
        assert p.is_dir()

        files = sorted(p.glob('*.h5'))
        if not files:
            raise RuntimeError('No hdf5 datasets found')

        self.num_tasks = cm.num_streamers() if cm.is_streamer() else 1
        self.task_id = cm.get_streaming_rank() if cm.is_streamer() else 0

        assert (
            len(files) % self.num_tasks == 0
        ), f"Number of h5 files {len(files)} should be divisible by the number of Slurm tasks {self.num_tasks}, to correctly shard the dataset between the streamers"

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

    def __len__(self):
        """
        Returns the len of dataset on the task process
        """
        return self.num_examples_in_this_task

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single-process
            worker_id = 0
            num_workers = 1

        if self.shuffle_seed:
            # Use a unique seed for each worker.
            random.seed(self.shuffle_seed + worker_id)

        self.data_partitions = self._shard_dataset(worker_id, num_workers)

    def create_dataloader(self, is_training=True):
        """
        Classmethod to create the dataloader object.
        """
        data_loader = torch.utils.data.DataLoader(
            BufferedShuffleDataset(
                dataset=self, buffer_size=self.shuffle_buffer
            )
            if self.shuffle
            else self,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            worker_init_fn=self._worker_init_fn,
        )
        # set self.data_partitions in case self.num_workers == 0
        if self.num_workers == 0:
            self._worker_init_fn(0)
        return data_loader

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
