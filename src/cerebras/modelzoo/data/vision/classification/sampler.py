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

import torch
import torch.distributed as dist


class RepeatedAugSampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        num_repeats=3,
        batch_size=256,
    ):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available!")
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError(
                f"num_repeats is set to {num_repeats}, but it should be an integer greater than 0"
            )
        if batch_size > len(dataset):
            raise RuntimeError(
                f"batch size is set to {batch_size}, but it is should be smaller "
                f"than the dataset size of {len(dataset)}"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(
            math.ceil(
                len(self.dataset) * float(num_repeats) / self.num_replicas
            )
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(
                len(self.dataset) // batch_size * batch_size / self.num_replicas
            )
        )
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        # Add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(
            indices, repeats=self.num_repeats, dim=0
        ).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
