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

"""Pytorch HDF5 Dataloader"""

import random

import torch
from torch.utils.data import default_collate

from cerebras.modelzoo.common.pytorch_utils import BufferedShuffleDataset
from cerebras.modelzoo.data.common.HDF5IterableDataset import (
    RestartableDataLoader,
)


class HDF5IterableDataProcessor:
    """
    A HDF5 dataset processor. Loads data from HDF5 files.
    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "shuffle_buffer" (int): Size of shuffle buffer in samples.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of batches loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    """

    def __init__(self, params):
        super(HDF5IterableDataProcessor, self).__init__()

        self.batch_size = self.dataset.batch_size

        self.shuffle = params["shuffle"]
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)

        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        if self.num_workers == 0:
            self.prefetch_factor = None
        else:
            self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            # Use a unique seed for each worker.
            random.seed(self.shuffle_seed + worker_id)

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        # Seed BufferedShuffleDataset() in case of single-worker,
        # for multiple workers, using _worker_init_fn()
        if self.num_workers == 0 and self.shuffle_seed is not None:
            random.seed(self.shuffle_seed)

        if self.shuffle:
            dataloader_cls = torch.utils.data.DataLoader
            dataset = BufferedShuffleDataset(
                dataset=self.dataset, buffer_size=self.shuffle_buffer
            )
        else:
            dataloader_cls = RestartableDataLoader
            dataset = self.dataset

        data_loader = dataloader_cls(
            dataset,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            worker_init_fn=self._worker_init_fn
            if self.num_workers > 0 and self.shuffle_seed is not None
            else None,
        )

        return data_loader
