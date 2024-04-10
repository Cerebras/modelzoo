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

"""PyTorch HDF5 Dataset"""

import json
import math
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.data.common.input_utils import (
    cluster_config,
    shard_list_of_chunks_contiguous,
)
from cerebras.pytorch.distributed import get_worker_state


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


class HDF5IterableDataset(torch.utils.data.IterableDataset):
    """
    A HDF5 dataset processor. Loads data from HDF5 files.
    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:
    - "data_dir" (str or list of str): Path to dataset HDF5 files
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "use_vsl" (bool): Flag to enable variable sequence length training.
       It requires the dataset to have two extra features: the
       `attention_span` of keys and the `position_ids` of tokens.
       Defaults to `False`.
    """

    def __init__(self, params):
        super(HDF5IterableDataset, self).__init__()

        self.data_dir = params["data_dir"]
        self.batch_size = get_streaming_batch_size(params["batch_size"])

        self.shuffle = params["shuffle"]
        self.shuffle_seed = params.get("shuffle_seed", None)

        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.use_vsl = params.get("use_vsl", False)

        self.num_feature_groups = 1
        # Load feature names from data_params.json, if present and
        # has the correct format (generated by HF > HDF5 converter script)
        if not isinstance(self.data_dir, list) and os.path.exists(
            os.path.join(self.data_dir, "data_params.json")
        ):
            try:
                with open(
                    os.path.join(self.data_dir, "data_params.json"), 'r'
                ) as _fin:
                    data_params = json.load(_fin)
                    if "features" in data_params:
                        self.features_list = data_params["features"]
                    elif "data_0_features" in data_params:
                        self.features_list = [data_params["data_0_features"]]
                        i = 1
                        while f"data_{i}_features" in data_params:
                            self.features_list.append(
                                data_params[f"data_{i}_features"]
                            )
                            i += 1
                        self.num_feature_groups = i
                    else:
                        self.features_list = [
                            "input_ids",
                            "attention_mask",
                            "labels",
                        ]
            except:
                self.features_list = ["input_ids", "attention_mask", "labels"]
        else:
            self.features_list = ["input_ids", "attention_mask", "labels"]

        if self.use_vsl:
            self.features_list = [
                "input_ids",
                "attention_mask",
                "labels",
                "attention_span",
                "position_ids",
            ]

        if self.batch_size <= 0:
            raise ValueError(
                f"Batch size should be a positive number, but got value {self.batch_size}."
            )

        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]

        self.files = []
        for directory in self.data_dir:
            p = Path(directory)
            if not p.is_dir():
                raise FileNotFoundError(
                    f"The path {directory} does not exist or is not a directory."
                )
            self.files.extend(p.glob('*.h5'))

        self.files = sorted(self.files)
        if not self.files:
            raise RuntimeError("No .h5 dataset files found.")

        cluster_spec, worker_spec = cluster_config()
        self.num_tasks = cluster_spec.num_tasks()
        self.task_id = worker_spec.rank

        # initialize state with 0 samples seen and shard_index = task_id
        self.set_state(0, self.task_id)

    def set_state(self, samples_seen, shard_index):
        """
        This method sets the state of the dataloader's samples_seen variable that controls
        how many samples are to be skipped for determinisitic restart.
        This is called by the load_state_dict method of the RestartableDataLoader.

        Args:
            samples_seen (int): number of samples streamed by the dataloader
            shard_index (int): the index of the shard of data that this worker
                is responsible for streaming
        """
        self._samples_seen = samples_seen
        self.shard_index = shard_index

        # Shard H5 files between the tasks and resolve the paths
        files_in_this_task = [
            str(file.resolve())
            for file in self.files[shard_index :: self.num_tasks]
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

    @property
    def samples_seen(self):
        return self._samples_seen % self.__len__()

    def _load_buffer(self, data_partitions):
        # partition id should default to 0 if not reading iter from file
        self.prev_worker_iter_index = self.samples_seen // self.batch_size
        restart_iter_partition_id = 0
        restart_iter_start_idx = 0  # start_idx should default to 0
        if self.prev_worker_iter_index > 0:
            # check total number of iterations/steps in the data partitions
            # This is required to determine the epoch of the restart iter
            worker_num_iters = self.num_examples_in_this_task // self.batch_size
            self.prev_worker_iter_index %= worker_num_iters
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
            if self.prev_worker_iter_index > 0:
                if restart_iter_partition_id >= 0 and partition_idx == 0:
                    start_idx = restart_iter_start_idx
                else:
                    start_idx = start_idx_org
            else:
                start_idx = start_idx_org
            with h5py.File(file_path, mode='r') as h5_file:
                if self.use_vsl and h5_file["data"].shape[1] != 5:
                    raise ValueError(
                        f"Expected all dataset H5 files to have 5 features for "
                        f"variable sequence length training, but got "
                        f"{h5_file['data'].shape[1]} features in {file_path}."
                    )
                for idx in range(
                    start_idx, start_idx_org + num_examples, self.batch_size
                ):
                    load_len = min(
                        self.batch_size, start_idx_org + num_examples - idx
                    )
                    if self.num_feature_groups == 1:
                        load_data = h5_file["data"][idx : idx + load_len]
                        for i in range(load_len):
                            yield load_data[i]
                    else:
                        load_data = [None] * self.num_feature_groups
                        for i in range(self.num_feature_groups):
                            load_data[i] = h5_file[f"data_{i}"][
                                idx : idx + load_len
                            ]
                        for i in range(load_len):
                            yield tuple(
                                [
                                    load_data[j][i]
                                    for j in range(self.num_feature_groups)
                                ]
                            )

        l = self.__len__()
        self._samples_seen = l * math.ceil((self._samples_seen + 1) / l)

    def __iter__(self):
        """
        Iterating over the data to construct input features.
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single-process
            worker_id = 0
            num_workers = 1

        data_partitions = shard_list_of_chunks_contiguous(
            self.files_in_this_task, worker_id, num_workers
        )

        for example in self._load_buffer(data_partitions):
            if self.num_feature_groups == 1:
                yield {
                    feature: np.array(example[i], np.int32)
                    for i, feature in enumerate(self.features_list)
                }
            else:
                sample = {}
                for j in range(self.num_feature_groups):
                    sample.update(
                        {
                            feature: np.array(example[j][i], np.int32)
                            for i, feature in enumerate(self.features_list[j])
                        }
                    )
                yield sample

    def __len__(self):
        """
        Returns the len of dataset on the task process
        """
        return self.num_examples_in_this_task
