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

"""Pytorch HDF5 Dataloader."""

import random
from typing import Any, Literal, Optional

import torch
from pydantic import Field
from torch.utils.data import default_collate

from cerebras.modelzoo.common.pytorch_utils import BufferedShuffleDataset
from cerebras.modelzoo.data.common.config import GenericDataProcessorConfig
from cerebras.modelzoo.data.common.HDF5IterableDataset import (
    HDF5IterableDataset,
    HDF5IterableDatasetConfig,
)


class HDF5IterableDataProcessorConfig(
    GenericDataProcessorConfig, HDF5IterableDatasetConfig
):
    data_processor: Literal["HDF5IterableDataProcessor"]

    vocab_size: Optional[Any] = Field(None, deprecated=True)


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

    def __init__(self, config: HDF5IterableDataProcessorConfig):
        if isinstance(config, dict):
            config = HDF5IterableDataProcessorConfig(**config)

        super().__init__()

        # HDF5IterableDataset yields samples with the features `input_ids`,
        # `attention_mask`, and `labels`. In the case of gpt models, the name
        # `attention_mask` is a misnomer. In reality it acts as a loss mask
        # and its contents aren't used for any attention masking.
        self.dataset = HDF5IterableDataset(config)

        self.batch_size = self.dataset.batch_size

        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        self.shuffle_buffer = config.shuffle_buffer or (10 * self.batch_size)

        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        if self.num_workers == 0:
            self.prefetch_factor = None
        else:
            self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

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
            prefetch_factor=(
                self.prefetch_factor if self.num_workers > 0 else None
            ),
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            worker_init_fn=(
                self._worker_init_fn
                if self.num_workers > 0 and self.shuffle_seed is not None
                else None
            ),
        )

        return data_loader


class RestartableDataLoader(torch.utils.data.DataLoader):
    """
    This custom dataloader class specifies the  'state_dict', 'aggregate_state_dict',
    'load_state_dict' and 'deaggregate_state_dict' methods.
    These methods dictate what worker state information  is stored
    (local or global streaming info) and how it is to be aggregated and retrieved.
    To deterministically restart an instance of HDF5IterableDataset, it requires the
    number of samples already seen in the previous run. This info is stored in the
    `samples_streamed` key inside the worker state dict. Upon restart, the
    `load_state_dict` method sets the `samples_seen` class variable which determines
    the number of samples to be skipped.
    """

    def __init__(self, *args, **kwargs):
        # keep track of how many samples were streamed in the previous portion
        # of the run so that we can track cumulative samples streamed in the
        # state_dict
        self.previous_samples_streamed = 0
        super().__init__(*args, **kwargs)

    def state_dict(self):
        from cerebras.pytorch.distributed import get_worker_state

        worker_state = get_worker_state()
        return {
            "samples_streamed": worker_state.samples_streamed
            + self.previous_samples_streamed,
            "shard_index": self.dataset.shard_index,
        }

    def load_state_dict(self, state_dict):
        if (
            len(state_dict) != 2
            or "samples_streamed" not in state_dict
            or "shard_index" not in state_dict
        ):
            raise RuntimeError(
                "The state dict must contain keys `samples_streamed` and `shard_index`, "
                f"but found {state_dict.keys()}. This means that the dataloader "
                "state in the checkpoint being loaded from is not compatible "
                "with the dataloader currently in use. Consider re-running "
                "without loading the dataloader state."
            )
        self.previous_samples_streamed = state_dict["samples_streamed"]
        self.dataset.set_state(
            state_dict["samples_streamed"], state_dict["shard_index"]
        )

    def aggregate_state_dict(self, worker_states):
        worker_states.sort(key=lambda x: x["shard_index"])
        return {"all_worker_states": worker_states}

    def deaggregate_state_dict(self, aggregated_state_dict):
        if (
            len(aggregated_state_dict) != 1
            or "all_worker_states" not in aggregated_state_dict
        ):
            raise RuntimeError(
                "The aggregated state dict must contain a single key "
                f"'all_worker_states', found {aggregated_state_dict.keys()}. "
                "This means that the dataloader state in the checkpoint you are "
                "loading from is not compatible with the dataloader currently "
                "in use. Consider re-running without loading the dataloader "
                "state."
            )
        all_worker_states = aggregated_state_dict["all_worker_states"]
        # For deterministic restart to work, the num_tasks for the previous run should
        # match with the restart run's num_tasks. If that condition is not met,
        # the dataloader would start from sample `0`.
        num_tasks_prev_run = len(all_worker_states)
        num_tasks = self.dataset.num_tasks
        task_id = self.dataset.task_id
        if num_tasks != num_tasks_prev_run:
            raise RuntimeError(
                "Unable to deterministically restart the dataloader. The total number "
                f"of workers for the initial run, {num_tasks_prev_run}, does not match "
                f"number of workers in the current run, {num_tasks}. This is currently "
                "not supported by the dataloader for the `HDF5IterableDataset`. Please "
                "ensure that `num_csx * num_workers_per_csx` is fixed across runs, or "
                "consider opting out of loading dataloader state via `runconfig.load_checkpoint_states`."
            )

        worker_index_offset = None
        min_samples = None
        for i, sd in enumerate(all_worker_states):
            if (
                worker_index_offset is None
                or sd["samples_streamed"] < min_samples
            ):
                worker_index_offset = i
                min_samples = sd["samples_streamed"]

        return all_worker_states[(task_id + worker_index_offset) % num_tasks]
