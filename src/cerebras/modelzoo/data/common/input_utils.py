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

import math
import random

import numpy as np
import torch

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.pytorch.distributed.cluster_resolver import ClusterSpec, TaskSpec


def get_data_for_task(
    task_id,
    meta_data_values_cum_sum,
    num_examples_per_task,
    meta_data_values,
    meta_data_filenames,
):
    """
    Function to get distribute files with given number of examples such that each
    distributed task has access to exactly the same number of examples

    Args:
        task_id (int): Integer id for a task.
        meta_data_values_cum_sum (int): Cumulative sum of the file sizes in
            lines from meta data file.
        num_examples_per_task (int): Number of the examples specified per
            slurm task. Equal to `batch_size` * `num_batch_per_task`.
        meta_data_values (list[int]): List of the files sizes in lines in the
            meta data file.
        meta_data_filenames (list[str]): List with file names in the meta data
            file.

    Returns:
        list of tuples of length 3. The tuple contains at
        - index 0: filepath.
        - index 1: number of examples to be considered for this task_id.
        - index 2: start index in the file from where these
                    examples should be considered
        The list represents the files that should be considered for this task_id.
    """
    files_in_task = []

    # file where the split starts
    file_start_idx = np.min(
        np.where(meta_data_values_cum_sum > task_id * num_examples_per_task)[0]
    )
    # Index in file from where the examples should be considered for this task
    start_idx = (
        task_id * num_examples_per_task
        - meta_data_values_cum_sum[file_start_idx - 1]
        # -1 since len(`meta_data_values_cum_sum`) = len(`meta_data_values`) + 1
    )

    # Number of examples to pick from this file.
    # We do a `min` to handle a case where the file has
    # examples > num_examples_per_task
    num_examples = min(
        meta_data_values[file_start_idx - 1] - start_idx,
        num_examples_per_task,
    )
    files_in_task.append(
        (
            meta_data_filenames[file_start_idx - 1],
            num_examples,
            start_idx,
        )  # (file_path, num_examples, start_index)
    )

    if num_examples != num_examples_per_task:
        # If the file has fewer number of examples than
        # `num_examples_per_task`, continue through files
        # till we reach our required number of examples.

        indices = np.where(
            meta_data_values_cum_sum > (task_id + 1) * num_examples_per_task
        )[0]
        if indices.size != 0:
            file_end_idx = np.min(indices)
        else:
            file_end_idx = len(meta_data_values_cum_sum)

        for i in range(file_start_idx + 1, file_end_idx):
            files_in_task.append(
                (
                    meta_data_filenames[i - 1],
                    meta_data_values[i - 1],
                    0,
                )  # (file_path, num_examples, start_index)
            )

        # If the number of examples needed to fulfill
        # `num_examples_per_task`, falls in between a file
        num_end_examples = (
            task_id + 1
        ) * num_examples_per_task - meta_data_values_cum_sum[file_end_idx - 1]
        if num_end_examples > 0:
            files_in_task.append(
                (
                    meta_data_filenames[file_end_idx - 1],
                    num_end_examples,
                    0,
                )  # (file_path, num_examples, start_index)
            )

    assert (
        sum([num_examples for _, num_examples, _ in files_in_task])
        == num_examples_per_task
    ), f"Incorrect number of examples in the split with task_id {task_id}"

    return files_in_task


def is_distributed():
    """
    Returns True if DDP is enabled.
    """
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def task_id():
    if dist.is_streamer():
        return dist.get_streaming_rank()
    elif is_distributed():
        return dist.get_rank()
    else:
        return 0


def num_tasks():
    if dist.is_streamer():
        return dist.num_streamers()
    elif is_distributed():
        return dist.get_world_size()
    else:
        return 1


def cluster_config():
    """
    Returns (ClusterSpec, TaskSpec). The TaskSpec contains the following fields:
        - rank: the global rank of the current worker
        - local_rank: the rank of the current worker among workers who feed
            the same system as the current worker
        - wse_id: the index of the system that the current worker is
            associated with
    The ClusterSpec contains the following fields:
        - tasks: a list of TaskSpecs for each task running on the cluster
        - rank: the rank of the current process's task in the cluster
        - num_csx: the number of CSX systems in the cluster
        - num_workers_per_csx: the number of worker tasks per CSX
    If the current job is running on GPU instead of CS system, then
    the ranks and world sizes in the returned TaskSpec will be set to the GPU
    rank and world size.

    """
    if cstorch.use_cs() and dist.is_streamer():
        cluster_spec = dist.service_resolver().cluster_spec
        task_spec = cluster_spec.task()
        return cluster_spec, task_spec
    elif is_distributed():
        task_spec = TaskSpec(
            rank=dist.get_rank(),
            local_rank=dist.get_rank(),
            wse_id=0,
            node_name="unknown",
        )
        cluster_spec = ClusterSpec(
            [task_spec],
            dist.get_rank(),
            1,
            dist.get_world_size(),
        )
        return cluster_spec, task_spec
    else:
        task_spec = TaskSpec(
            rank=0, local_rank=0, wse_id=0, node_name="unknown"
        )
        cluster_spec = ClusterSpec([task_spec], 0, 1, 1)
        return cluster_spec, task_spec


class ShardedSampler(torch.utils.data.Sampler):
    """
    Modified from:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
    Sampler that restricts data loading to a subset of the dataset.

    Dataset is assumed to be of constant size.

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        shuffle (bool, optional): If `True` (default), sampler will shuffle
            the indices.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: `0`.
        drop_last (bool, optional): If `True`, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If `False`, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: `False`.
    """

    def __init__(self, dataset, shuffle=True, seed=None, drop_last=False):
        self.num_tasks = num_tasks()
        self.task_id = task_id()
        self.dataset = dataset
        self.dataset_len = len(self.dataset)
        self.drop_last = drop_last

        if cstorch.use_cs() and not self.drop_last:
            raise ValueError(
                "On CS2 we do not support unequal batch sizes so `drop_last` "
                "must be set to `True`."
            )
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_tasks:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each task receives the same amount of data when
            # using this sampler.
            self.num_samples = len(self.dataset) // self.num_tasks
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_tasks)
        self.total_size = self.num_samples * self.num_tasks
        self.shuffle = shuffle
        self.seed = seed
        self.indices = list(range(self.dataset_len))
        if not self.drop_last:
            # add extra samples to make it evenly divisible across tasks
            padding_indices_size = self.total_size - self.dataset_len
            # choose padding indices at random to reduce the chance of
            # reusing samples.
            random.seed(self.seed)
            padding_indices = random.sample(self.indices, padding_indices_size)
            self.indices += padding_indices
        else:
            # remove tail of data to make it evenly divisible.
            self.indices = self.indices[: self.total_size]
        assert len(self.indices) == self.total_size, (
            f"Total `indices` after dropping/padding indices must be equal "
            f"to `total_size` of the dataset. Received total indices: "
            f"`{len(self.indices)}` and total size is: `{self.total_size}`."
        )

    def __iter__(self):
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.indices)

        # subsample
        indices = self.indices[self.task_id : self.total_size : self.num_tasks]
        assert len(indices) == self.num_samples, (
            f"Total `indices` for tasks must be equal to `num_samples` in a "
            f"task. Received total indices: `{len(indices)}` and samples in "
            f"task are: `{self.num_samples}`."
        )

        yield from indices

    def __len__(self):
        return self.num_samples


def check_sharding_sanity(
    examples_per_file,
    batch_size,
    num_workers,
    drop_last,
):
    """Checks if with the given sharding, at least one batch is generated.

    Note that this method is operating based on how `shard_and_shuffle_data` is
    sharding the data across workers.

    :param list examples_per_file: Total examples per file for this task.
    :param int batch_size: Batch size of the model.
    :param int num_workers: Number of workers to use in the dataloader.
    :param bool drop_last: Boolean indicating whether the last incomplete batch
        of the dataloader is dropped.
    :raises ValueError: If no batches are generated with the given sharding.
    """
    if drop_last is False:
        return

    if num_workers == 0:
        total_samples = sum(examples_per_file)
        if total_samples < batch_size:
            raise ValueError(
                f"Task {task_id()} only generates {total_samples}, which "
                f"is fewer than a full batch of size {batch_size}. "
            )
        return

    examples_per_worker = [0] * num_workers
    for file_idx, examples_in_file in enumerate(examples_per_file):
        worker_id = file_idx % num_workers
        examples_per_worker[worker_id] += examples_in_file

    max_examples = max(examples_per_worker)

    if max_examples < batch_size:
        raise ValueError(
            f"Maximum number of samples generated in dataloader workers of "
            f"task {task_id()} is {max_examples}. Since {max_examples} is less "
            f"than batch size {batch_size} and `drop_last` is True, this task "
            f"will end up not producing any samples. Please specify a fewer "
            f"number of workers or tasks."
        )


def shard_list_contiguous(input_list, worker_id, num_workers):
    """
    Shards a list by splitting it into `num_workers` contiguous segments.
    Only the `worker_id`th shard is returned. If the length of the list is
    not divisible by the number of workers, the last worker will be assigned
    all remainder elements.

    Args:
        input_list (list): list to shard into contiguous segments
        worker_id (int): index of shard to return
        num_workers (int): number of shards to create

    Returns:
        A sublist of contiguous elements (`worker_id`'s shard)
    """

    assert num_workers <= len(input_list), (
        f"Number of processes should be less than number of files, "
        f"Got `num_workers` equal to {num_workers} and `num_files` equal to {len(input_list)}."
    )

    per_worker_num_files = len(input_list) // num_workers
    if worker_id < num_workers - 1:
        output_list = input_list[
            (worker_id * per_worker_num_files) : (
                (worker_id + 1) * per_worker_num_files
            )
        ]
    else:
        output_list = input_list[(worker_id * per_worker_num_files) :]
    return output_list


def shard_list_interleaved(input_list, worker_id, num_workers):
    """
    Shards a list by assigning consecutive elements to alternating workers
    (i.e. interleaving). If the length of the list is not divisible by the
    number of workers, the remainder elements are spread across a subset
    of the workers such that each worker in the subset receives 1 extra
    element.

    Args:
        input_list (list): list to shard in an interleaved fashion
        worker_id (int): index of shard to return
        num_workers (int): number of shards to create

    Returns:
        `worker_id`'s shard (a subset of `input_list`).
    """

    output_for_cur_worker = []

    if num_workers != 0:
        assert num_workers <= len(input_list), (
            f"Number of processes should be less than number of files, "
            f"Got `num_workers` equal to {num_workers} and `num_files` equal to {len(input_list)}."
        )

        # Gather files for the input worker based in the file index and
        # number of workers.
        for index, elm in enumerate(input_list):
            if index % num_workers == worker_id:
                output_for_cur_worker.append(elm)
    else:
        output_for_cur_worker = input_list

    return output_for_cur_worker


def shard_list_of_chunks_contiguous(
    input_list_of_chunks, worker_id, num_workers
):
    """
    Shards a list of chunks by distributing contiguous segments of each chunk
    across shards. If the chunk's length is not divisible by the
    number of workers, the remainder elements are spread across a subset
    of the workers such that each worker in the subset receives 1 extra
    element.

    Args:
        input_list (list of tuples): list of chunks to shard. List should be of format
            `[... (chunk_i, length_of_chunk_i), ...]`
        worker_id (int): index of shard to return
        num_workers (int): number of shards to create

    Returns:
        `worker_id`'s shard: a list of the same length as `input_list` of the
        format: `[... (chunk_i, shard_start_index_i, shard_length_i), ...]`
    """
    output_for_cur_worker = []
    for elm, chunk_length in input_list_of_chunks:
        # Try to evenly distribute chunk_length between workers
        chunk_length_per_worker = [(chunk_length // num_workers)] * num_workers
        for i in range(chunk_length % num_workers):
            chunk_length_per_worker[i] += 1

        assert sum(chunk_length_per_worker) == chunk_length

        output_for_cur_worker.append(
            (
                elm,
                (
                    sum(chunk_length_per_worker[:worker_id])
                    if worker_id > 0
                    else 0
                ),  # Start index
                chunk_length_per_worker[worker_id],  # Length of data chunk
            )
        )
    return output_for_cur_worker
