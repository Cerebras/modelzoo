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

import torch
import torch.distributed as dist

from modelzoo.common.pytorch import cb_model as cm


def is_gpu_distributed():
    """
    Returns True if DDP is enabled
    """
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def task_id():
    if cm.is_streamer():
        return cm.get_streaming_rank()
    elif is_gpu_distributed():
        return dist.get_rank()
    else:
        return 0


def num_tasks():
    if cm.is_streamer():
        return cm.num_streamers()
    elif is_gpu_distributed():
        return dist.get_world_size()
    else:
        return 1


class ShardedSampler(torch.utils.data.Sampler):
    """
    Modified from:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
    Sampler that restricts data loading to a subset of the dataset.

    Dataset is assumed to be of constant size.

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        mode (modes): Instance of `modes` to indicate train or eval mode.
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

        if cm.use_cs() and not self.drop_last:
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


##### Experimental to reduce first batch loading times for MAP style only #####
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(
            self, 'batch_sampler', _RepeatSampler(self.batch_sampler)
        )
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
