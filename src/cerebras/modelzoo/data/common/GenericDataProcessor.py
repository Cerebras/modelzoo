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

"""Pytorch Generic Dataloader"""

import random
from typing import Literal, Optional

import torch

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import (
    BufferedShuffleDataset,
    IterableDatasetSampler,
)
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.data.common.input_utils import num_tasks, task_id


class GenericDataProcessorConfig(DataConfig):
    data_processor: Literal["GenericDataProcessor"]

    batch_size: int = ...
    "The Batch size."

    shuffle: bool = False
    "Flag to enable data shuffling."

    shuffle_seed: Optional[int] = None
    "Shuffle seed."

    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."

    num_workers: int = 0
    "How many subprocesses to use for data loading."

    drop_last: bool = True
    """
    If True and the dataset size is not divisible by the batch size, the last
    incomplete batch will be dropped.
    """

    prefetch_factor: Optional[int] = 10
    "Number of batches loaded in advance by each worker."

    persistent_workers: bool = True
    """
    If True, the data loader will not shutdown the worker processes after a
    dataset has been consumed once.
    """


class GenericDataProcessor:
    """
    A Generic PyTorch Data Processor.

    Args:
        config: the configuration object for the data processor.
    """

    def __init__(self, config: GenericDataProcessorConfig, dataset):
        if isinstance(config, dict):
            config = GenericDataProcessorConfig(**config)

        super().__init__()

        self.batch_size = get_streaming_batch_size(config.batch_size)

        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        self.shuffle_buffer = config.shuffle_buffer
        if self.shuffle_buffer is None:
            self.shuffle_buffer = 10 * self.batch_size

        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        assert self.batch_size > 0, "Batch size should be positive."

        self.dataset = dataset

        if isinstance(self.dataset, torch.utils.data.IterableDataset):
            self.map_style_dataset = False
        else:
            self.map_style_dataset = True

        if not hasattr(self, "data_collator"):
            self.data_collator = None

        if self.shuffle and not self.map_style_dataset:
            # Distributed sampling of an IterableDataset between different tasks
            self.dataset = IterableDatasetSampler(
                self.dataset, world_size=num_tasks(), rank=task_id()
            )
            self.sampler = None

            self.dataset = BufferedShuffleDataset(
                dataset=self.dataset, buffer_size=self.shuffle_buffer
            )

        else:
            torch.manual_seed(self.shuffle_seed)

            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=num_tasks(),
                rank=task_id(),
                shuffle=self.shuffle,
                seed=self.shuffle_seed,
                drop_last=self.drop_last,
            )

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            # Use a unique seed for each worker.
            random.seed(self.shuffle_seed + worker_id)

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        # Seed BufferedShuffleDataset() in case of single-worker,
        # for multiple workers, using _worker_init_fn()
        if (
            self.num_workers == 0
            and self.shuffle_seed
            and not self.map_style_dataset
        ):
            random.seed(self.shuffle_seed)

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=(
                self.prefetch_factor if self.num_workers > 0 else None
            ),
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            collate_fn=self.data_collator,
            worker_init_fn=(
                self._worker_init_fn
                if self.num_workers > 0 and not self.map_style_dataset
                else None
            ),
        )
        return data_loader
