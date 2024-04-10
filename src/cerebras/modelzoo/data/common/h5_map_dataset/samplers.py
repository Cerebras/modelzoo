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
import math

import torch

from cerebras.modelzoo.data.common.input_utils import cluster_config


class PaddedSampler:
    # Arbitrary object to use as a placeholder for padding indices
    pad_index = object()


class CBSampler(torch.utils.data.Sampler, PaddedSampler):
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
        num_samples=None,
        pad_last=False,
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
                performed. When running on worker nodes, this should be the
                per-system batch size rather than the global batch size or the
                microbatch size. The per-system batch size is defined as
                `global_batch_size / num_csx` and can be found using the
                `modelzoo.common.utils.utils.get_streaming_batch_size`
                function. When running on the coordinator node, this should
                be the global batch size. Again, the `get_streaming_batch_size`
                function will return the appropriate result.
            num_samples (int): The number of samples to shuffle over. In multi-
                epoch training, it is common to set this to the total number
                of samples that you plan to see in your training run to get
                smoother loss curves and improved convergence.
            pad_last (bool): Flag to enable padding of the last batch so
                that the last batch has the same batch size as the rest of the
                batches. Only used if `batch_size` is not `None` and `drop_last`
                is `False`.
        """
        cluster_spec, _ = cluster_config()
        _num_systems = cluster_spec.num_csx
        if _num_systems > 1 and not drop_last:
            raise ValueError(
                f"`drop_last=False` is only supported on GPU. Please re-run "
                f"with `drop_last=True`."
            )
        self.sampler = BaseSampler(
            data_source,
            shuffle=shuffle,
            seed=seed,
            start_index=start_index,
            num_samples=num_samples,
        )
        if batch_size is not None:
            self.sampler = BatchSampler(
                self.sampler, batch_size, drop_last, pad_last
            )
        if shard:
            self.sampler = Sharder(self.sampler)

        self.kwargs = {
            "data_source": data_source,
            "shuffle": shuffle,
            "seed": seed,
            "shard": shard,
            "batch_size": batch_size,
            "drop_last": drop_last,
        }

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

    def set_state(self, start_index):
        """
        Sets the state of the sampler to continue deterministically from a prior
        run.

        Args:
            start_index: the total number of samples streamed globally across
                all workers from a previous run.
        """
        self.__init__(**self.kwargs, start_index=start_index)


class BaseSampler(torch.utils.data.Sampler, PaddedSampler):
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
            if self.num_samples > len(self.data_source):
                epochs = math.ceil(self.num_samples / len(self.data_source))
                perm = torch.cat(
                    [
                        torch.arange(len(self.data_source))
                        for _ in range(epochs - 1)
                    ]
                )
                perm = torch.cat(
                    (perm, torch.randperm(len(self.data_source), generator=gen))
                )
                perm = perm[: self.num_samples]
                indices = torch.randperm(self.num_samples, generator=gen)
                perm = perm[indices]
            else:
                perm = torch.randperm(len(self.data_source), generator=gen)
                perm = perm[: self.num_samples]
            perm = perm[self.start_index :]
            yield from perm.tolist()
        else:
            yield from range(self.start_index, self.num_samples)
        self.epoch += 1
        self.start_index = 0

    def __len__(self):
        return self.num_samples - self.start_index


class Sharder(torch.utils.data.Sampler, PaddedSampler):
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


class BatchSampler(torch.utils.data.Sampler, PaddedSampler):
    """
    A slight modification of the PyTorch batch sampler such that any samples not
    yielded at the end of an epoch when `drop_last=True` will be yielded at the
    start of the next epoch. This is necessary for shard-invariance.

    Adapted from the PyTorch batch sampler
    """

    def __init__(self, sampler, batch_size, drop_last, pad_last):
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
        self.pad_last = pad_last
        self.leftover_samples = []

        if len(self.sampler) < self.batch_size:
            self.leftover_samples = [s for s in self.sampler]

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
                if self.pad_last:
                    while idx_in_batch < self.batch_size:
                        batch[idx_in_batch] = self.pad_index
                        idx_in_batch += 1
                    yield batch
                else:
                    yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return (
                len(self.sampler) + len(self.leftover_samples)
            ) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
