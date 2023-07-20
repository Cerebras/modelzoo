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

import itertools

import torch

from modelzoo.transformers.pytorch.input_utils import cluster_config


class CBSampler(torch.utils.data.Sampler):
    """
    A sampler to handle sharding, batching, and skipping of map style datasets
    intended for use on CSX. Sharding is performed in such a way that data order
    is independent of the number of systems being used and the number of workers
    per system.
    """

    def __init__(
        self,
        data_source,
        shuffle=True,
        seed=None,
        start_index=0,
        shard=True,
        batch_size=None,
        drop_last=True,
    ):
        """
        Create a sampler to handle shuffling in a deterministic and restartable
        way as well as sharding.

        Args:
            data_source (torch.utils.data.Dataset): dataset to sample from
            shuffle (bool): whether or not to shuffle the dataset
            seed (int): The seed used to make shuffling deterministic
            start_index (int): The index of the first sample to yield
            shard (bool): Whether or not to shard the dataset across Cerebras
                data streamer nodes
            batch_size (int): The batch size to use to compute sharded indices
                and group samples into batches. If `None`, no batching will be
                performed. This is the global batch size visible to the dataset
                rather than the microbatch size.
        """
        cluster_spec, _ = cluster_config()
        _num_systems = cluster_spec.num_csx
        if batch_size is not None and batch_size % _num_systems:
            raise ValueError(
                f"The global batch size must be a multiple of the number of "
                f"CS-2s being used. Got global batch size {batch_size} and "
                f"number of systems {_num_systems}."
            )
        if _num_systems > 1 and not drop_last:
            raise ValueError(
                f"`drop_last=False` is only supported on GPU. Please re-run "
                f"with `drop_last=True`."
            )
        microbatch_size = (
            batch_size // _num_systems if batch_size is not None else None
        )
        self.sampler = BaseSampler(
            data_source, shuffle=shuffle, seed=seed, start_index=start_index
        )
        if batch_size is not None:
            self.sampler = BatchSampler(
                self.sampler, microbatch_size, drop_last
            )
        if shard:
            self.sampler = Sharder(self.sampler)
        if batch_size is not None and _num_systems > 1:
            self.sampler = BatchAccumulator(self.sampler, _num_systems)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class BaseSampler(torch.utils.data.Sampler):
    """
    Handle shuffling and skipping
    """

    def __init__(
        self,
        data_source,
        num_samples=None,
        shuffle=True,
        seed=None,
        start_index=0,
    ):
        self.data_source = data_source
        self._num_samples = num_samples
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )
        self._num_samples_frozen = self.num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = start_index // self.num_samples
        self.start_index = start_index - self.num_samples * self.epoch

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        if self.num_samples != self._num_samples_frozen:
            raise RuntimeError(
                f"Data source passed into Sampler must have the same length "
                f"every epoch. Original length was {self._num_samples_frozen}, "
                f"new length is {self.num_samples}"
            )
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self.epoch)
            if self.num_samples >= len(self.data_source):
                perm = torch.randperm(self.num_samples, generator=gen)
            else:
                perm = torch.randperm(len(self.data_source), generator=gen)
                perm = perm[: self.num_samples]
            perm = perm[self.start_index :]
        else:
            perm = torch.arange(self.start_index, self.num_samples)
        yield from perm.tolist()
        self.epoch += 1
        self.start_index = 0

    def __len__(self):
        return self.num_samples - self.start_index


class Sharder(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        cluster_spec, worker_spec = cluster_config()

        self.task_id = (
            worker_spec.local_rank * cluster_spec.num_csx + worker_spec.wse_id
        )
        self.num_tasks = cluster_spec.num_workers_per_csx * cluster_spec.num_csx
        self.first_task = 0

    def __iter__(self):
        n = len(self.data_source)
        effective_task_id = (self.task_id - self.first_task) % self.num_tasks
        for i, x in enumerate(self.data_source):
            if i % self.num_tasks == effective_task_id:
                yield x
        self.first_task = (
            self.first_task + (n % self.num_tasks)
        ) % self.num_tasks

    def __len__(self):
        effective_task_id = (self.task_id - self.first_task) % self.num_tasks
        n = len(self.data_source)
        l = n // self.num_tasks
        if n % self.num_tasks > effective_task_id:
            l += 1
        return l


class BatchSampler(torch.utils.data.Sampler):
    """
    A slight modification of the PyTorch batch sampler such that any samples not
    yielded at the end of an epoch when `drop_last=True` will be yielded at the
    start of the next epoch. This is necessary for shard-invariance.

    Adapted from the PyTorch batch sampler
    """

    def __init__(self, sampler, batch_size, drop_last):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.leftover_samples = []

    def __iter__(self):
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = itertools.chain(self.leftover_samples, self.sampler)
            while True:
                try:
                    batch = []
                    for _ in range(self.batch_size):
                        batch.append(next(sampler_iter))
                    yield batch
                except StopIteration:
                    self.leftover_samples = batch
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return (
                len(self.sampler) + len(self.leftover_samples)
            ) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class BatchAccumulator(torch.utils.data.Sampler):
    """
    Accumulate neighboring batches into one single larger batch. This is the
    inverse operation to the splitting of batches into microbatches that
    happens when using multiple CSX systems.
    """

    def __init__(
        self, data_source, n_accum,
    ):
        """
        Assumes data_source is an iterator of batches where each batch has the
        same length (i.e. `drop_last=True`).
        """
        self.data_source = data_source
        self._n = n_accum
        self._next_batch = []

    def __iter__(self):
        data_iter = itertools.chain(self._next_batch, self.data_source)
        self._next_batch = []
        while True:
            try:
                for _ in range(self._n):
                    self._next_batch.append(next(data_iter))
                yield [x for batch in self._next_batch for x in batch]
                self._next_batch = []
            except StopIteration:
                break

    def __len__(self):
        return (len(self.data_source) + len(self._next_batch)) // self._n
