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
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torchvision import transforms

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import BufferedShuffleDataset
from cerebras.modelzoo.data.vision.segmentation.Hdf5BaseDataProcessor import (
    Hdf5BaseDataProcessorConfig,
)
from cerebras.modelzoo.data.vision.segmentation.preprocessing_utils import (
    normalize_tensor_transform,
)
from cerebras.modelzoo.data.vision.utils import create_worker_cache


class Hdf5BaseIterDataProcessorConfig(Hdf5BaseDataProcessorConfig):
    data_processor: Literal["Hdf5BaseIterDataProcessor"]


class Hdf5BaseIterDataProcessor(ABC, torch.utils.data.IterableDataset):
    """
    A HDF5 dataset processor for UNet HDF dataset.
    Performs on-the-fly augmentation of image and labek.

    Functionality includes:
        Reading data from HDF5 documents
        Augmenting data
    """

    def __init__(self, config: Hdf5BaseIterDataProcessorConfig):
        super(Hdf5BaseIterDataProcessor, self).__init__()

        use_worker_cache = config.use_worker_cache
        self.data_dir = config.data_dir
        if use_worker_cache and dist.is_streamer():
            if not cstorch.use_cs():
                raise RuntimeError(
                    "use_worker_cache not supported for non-CS runs"
                )
            else:
                self.data_dir = create_worker_cache(self.data_dir)

        self.num_classes = config.num_classes
        self.normalize_data_method = config.normalize_data_method
        if self.normalize_data_method:
            # Normalize
            self.normalize_transform = transforms.Lambda(
                self._apply_normalization
            )

        self.image_shape = config.image_shape  # of format (H, W, C)
        (
            self.tgt_image_height,
            self.tgt_image_width,
            self.channels,
        ) = self.image_shape

        self.loss_type = config.loss

        self.shuffle_seed = config.shuffle_seed
        if self.shuffle_seed is not None:
            torch.manual_seed(self.shuffle_seed)

        self.augment_data = config.augment_data
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.shuffle = config.shuffle

        self.shuffle_buffer = config.shuffle_buffer

        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        self.mp_type = cstorch.amp.get_floating_point_dtype()

        self.num_tasks = dist.num_streamers() if dist.is_streamer() else 1
        self.task_id = dist.get_streaming_rank() if dist.is_streamer() else 0

        # set later once processor gets a call to create a dataloader
        self.num_examples_in_this_task = 0
        self.files_in_this_task = []

    @abstractmethod
    def _shard_files(self, is_training=False):
        pass

    @abstractmethod
    def _load_buffer(self, data_partitions):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def _shard_dataset(self, worker_id, num_workers):
        pass

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

        if self.shuffle_seed is not None:
            # Use a unique seed for each worker.
            random.seed(self.shuffle_seed + worker_id)

        self.data_partitions = self._shard_dataset(worker_id, num_workers)

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        is_training = self.split == "train"
        self._shard_files(is_training)

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(self.files_in_this_task)

        data_loader = torch.utils.data.DataLoader(
            (
                BufferedShuffleDataset(
                    dataset=self, buffer_size=self.shuffle_buffer
                )
                if self.shuffle
                else self
            ),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=(
                self.prefetch_factor if self.num_workers > 0 else None
            ),
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            worker_init_fn=self._worker_init_fn,
        )
        # set self.data_partitions in case self.num_workers == 0
        if self.num_workers == 0:
            self._worker_init_fn(0)
        return data_loader

    def _apply_normalization(self, x):
        return normalize_tensor_transform(
            x, normalize_data_method=self.normalize_data_method
        )
