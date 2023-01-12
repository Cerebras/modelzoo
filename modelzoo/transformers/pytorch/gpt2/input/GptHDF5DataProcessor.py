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
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from modelzoo.common.pytorch.utils import BufferedShuffleDataset
from modelzoo.transformers.pytorch.input_utils import (
    num_tasks,
    shard_list_of_chunks_contiguous,
    task_id,
)


class GptHDF5DataProcessor(torch.utils.data.IterableDataset):
    """
    A HDF5 dataset processor for GPT pre-training.
    Performs on-the-fly processing of data from text.

    Functionality includes:
        Reading data from text documents
        Creating creating input sequences and masks, and
        autoregressive LM labels

    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (str or list of str): Path to dataset HDF5 files
    - "max_sequence_length (int): Maximum length of the sequence to generate
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_buffer" (int): Size of shuffle buffer in samples.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of batches loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    """

    def __init__(self, params):
        super(GptHDF5DataProcessor, self).__init__()

        self.data_dir = params["data_dir"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]

        self.shuffle = params["shuffle"]
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)

        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)
        self.dataloader_state = params.get('cerebras', {})

        # Features in HDF5 files
        self.features_list = ["input_ids", "attention_mask", "labels"]

        assert self.batch_size > 0, "Batch size should be positive."

        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]

        files = []
        for directory in self.data_dir:
            p = Path(directory)
            assert p.is_dir()
            files.extend(p.glob('*.h5'))

        files = sorted(files)
        if not files:
            raise RuntimeError('No hdf5 datasets found')

        self.num_tasks = num_tasks()
        self.task_id = task_id()

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

        # Single worker
        # laod dataloader state from previous run for restart
        self.dataloader_state_file = self.dataloader_state.get(
            'dataloader_state_file', None
        )
        self.num_workers_prev_state = self.dataloader_state.get(
            'num_workers', None
        )
        if self.num_workers_prev_state is not None:
            assert (
                self.num_workers == self.num_workers_prev_state
            ), "num_workers should be the same at the restart"

        # Sanity check whether or not "dataloader_state_file" is readable
        if self.dataloader_state_file is not None:
            assert os.path.isfile(
                self.dataloader_state_file
            ), f"Invalid dataloader state file: '{self.dataloader_state_file}'"
            with open(self.dataloader_state_file, 'r') as f:
                self.prev_worker_iter_index = int(f.readline())
        else:
            self.prev_worker_iter_index = 0

    def _load_buffer(self, data_partitions):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else:
            # 1 sub-process
            worker_id = 0

        restart_iter_partition_id = (
            0  # partition id should default to 0 if not reading iter from file
        )
        restart_iter_start_idx = 0  # start_idx should default to 0
        # Sanity check whether or not "dataloader_state_file" is readable
        if self.prev_worker_iter_index > 0:
            iters_until_current_partition = 0
            prev_partition_offset_start_idx = 0
            current_partition_offset_start_idx = 0
            for partition_idx, partition_specs in enumerate(data_partitions):
                start_idx = partition_specs[1]
                num_examples = partition_specs[2]

                if partition_idx > 0:
                    num_examples_prev_partition = (
                        data_partitions[partition_idx - 1][2]
                        - prev_partition_offset_start_idx
                    )
                    if (
                        num_examples_prev_partition
                        - (num_examples_prev_partition // self.batch_size)
                        * self.batch_size
                    ) > 0:
                        current_partition_offset_start_idx = self.batch_size - (
                            num_examples_prev_partition
                            - (num_examples_prev_partition // self.batch_size)
                            * self.batch_size
                        )
                    else:
                        current_partition_offset_start_idx = 0
                    prev_partition_offset_start_idx = (
                        current_partition_offset_start_idx
                    )
                    num_examples_curr_partition = (
                        num_examples - current_partition_offset_start_idx
                    )
                else:
                    num_examples_curr_partition = num_examples
                    current_partition_offset_start_idx = 0

                iters_until_current_partition += np.ceil(
                    num_examples_curr_partition / self.batch_size
                )
                if (
                    self.prev_worker_iter_index
                    <= iters_until_current_partition - 1
                ):
                    restart_iter_partition_id = partition_idx
                    restart_iter_start_idx = int(
                        self.batch_size
                        * (
                            self.prev_worker_iter_index
                            - (
                                iters_until_current_partition
                                - np.ceil(
                                    num_examples_curr_partition
                                    / self.batch_size
                                )
                            )
                        )
                    )

                    restart_iter_start_idx += current_partition_offset_start_idx

                    break

        for partition_idx, partition_specs in enumerate(
            data_partitions[restart_iter_partition_id:]
        ):
            file_path = partition_specs[0]
            start_idx_org = partition_specs[1]
            num_examples = partition_specs[2]
            if restart_iter_partition_id >= 0 and partition_idx == 0:
                start_idx = restart_iter_start_idx
            else:
                start_idx = start_idx_org
            with h5py.File(file_path, mode='r') as h5_file:
                for idx in range(
                    start_idx, start_idx_org + num_examples, self.batch_size
                ):
                    load_len = min(
                        self.batch_size, start_idx_org + num_examples - idx
                    )
                    load_data = h5_file["data"][idx : idx + load_len]
                    for i in range(load_len):
                        yield load_data[i]

    def __iter__(self):
        """
        Iterating over the data to construct input features.
        """
        for example in self._load_buffer(self.data_partitions):
            yield {
                feature: np.array(example[i], np.int32)
                for i, feature in enumerate(self.features_list)
            }

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

        self.data_partitions = shard_list_of_chunks_contiguous(
            self.files_in_this_task, worker_id, num_workers
        )

    def create_dataloader(self, is_training=True):
        """
        Classmethod to create the dataloader object.
        """
        data_loader = torch.utils.data.DataLoader(
            BufferedShuffleDataset(
                dataset=self, buffer_size=self.shuffle_buffer
            )
            if self.shuffle and not bool(self.dataloader_state)
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
        if self.num_workers == 0:
            self._worker_init_fn(0)

        return data_loader
