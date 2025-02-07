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

import filecmp
import math
import os
import random
import shutil

import torch
import torch.distributed as dist
from tqdm import tqdm

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist


def is_gpu_distributed():
    """
    Returns True if DDP is enabled
    """
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def task_id():
    if dist.is_streamer():
        return dist.get_streaming_rank()
    elif is_gpu_distributed():
        return dist.get_rank()
    else:
        return 0


def num_tasks():
    if dist.is_streamer():
        return dist.num_streamers()
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


##### Experimental to reduce first batch loading times for MAP style only #####
class _RepeatSampler(object):
    """Sampler that repeats forever.

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


def _get_worker_cache_dir(src_dir):
    """Gets the path to worker cache dir corresponding to the src_dir"""
    src_dir = os.path.abspath(src_dir)
    cache_dir = os.path.normpath("/".join([dist.WORKER_CACHE_ROOT, src_dir]))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _same_dirs_shallow(src_dir, dest_dir):
    """Takes a directory comparison obj and does a shallow comparison
    between the dirs src_dir and dest_dir
    The shallow comparison does a recursive check of the following:
        1. Check if the dirs exist, if they don't then return False
        2. Check if the files have a diff, or if there
        are additional files for either of the two dirs, if different,
        return False.
        3. Repeat 1 and 2 on subdirs
    """

    def _same_dirs_shallow_helper(dcmp: filecmp.dircmp):
        if not os.path.exists(dcmp.left) or not os.path.exists(dcmp.right):
            return False
        if dcmp.left_only:
            # If diff consists of only broken
            # symlinks, then its a match
            parent = dcmp.left
            for left_file in dcmp.left_only:
                if os.path.isdir(
                    os.path.join(parent, left_file)
                ) or os.path.isfile(os.path.join(parent, left_file)):
                    return False
        if dcmp.diff_files or dcmp.right_only:
            return False
        for sub_dcmp in dcmp.subdirs.values():
            if not _same_dirs_shallow_helper(sub_dcmp):
                return False
        return True

    return _same_dirs_shallow_helper(filecmp.dircmp(src_dir, dest_dir))


def create_worker_cache(src_dir: str, force_overwrite: bool = False):
    """Checks for the dir in the worker_cache (SSD) on the worker node corresponding to the src_dir.
    If the directory exists and is same as the src_dir, it returns the dir path on worker_cache.
    Otherwise writes the directory to the worker_cache and returns the dir path.
    Writing to the cache can take a while, depending on the size of the src_dir:
    Displays a progress bar (in the worker logs) which shows progress of the cache
    Forces cache overwrite irrespective of a cache hit, when force_overwrite is True.
    """
    from filelock import FileLock

    if (
        os.path.commonprefix([src_dir, dist.WORKER_CACHE_ROOT])
        == dist.WORKER_CACHE_ROOT
    ):
        raise RuntimeError(
            f"Ensure that the src_dir path does not have "
            f"a worker_cache path prefix: {dist.WORKER_CACHE_ROOT}"
        )
    if not dist.is_streamer():
        raise RuntimeError(
            "Ensure that create_worker_cache is called only for a worker node."
        )
    dest_dir = _get_worker_cache_dir(src_dir)
    # Provide read/write permissions for the lock for all users
    with FileLock(f"{dest_dir}.lock", mode=0o666):
        if _same_dirs_shallow(src_dir, dest_dir) and not force_overwrite:
            print(f"WORKER CACHE HIT: Skipping overwrite")
        else:
            (
                is_limit_hit,
                dir_size,
                available_space_for_copy,
            ) = dist.hit_worker_cache_limit(src_dir, dest_dir)
            if is_limit_hit:
                raise RuntimeError(
                    f"Failed when copying the directory to the worker cache: {src_dir},"
                    f" directory size: {dir_size} exceeds the available space on worker cache: {available_space_for_copy}."
                    f"Please contact your system administrator to clear the worker cache."
                )
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            # copy dirs to destination
            # get the total number of files to copy
            total_files = sum(
                [len(files) for root, dirs, files in os.walk(src_dir)]
            )

            # copy directory with progress bar
            def copy2_with_progress(src_path, dst_path, update):
                # skip if its a broken symlink
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                    update(1)

            with tqdm(
                total=total_files,
                desc="Overwriting cache",
                unit="files",
                dynamic_ncols=True,
            ) as pbar:
                shutil.copytree(
                    src_dir,
                    dest_dir,
                    symlinks=False,
                    ignore=None,
                    ignore_dangling_symlinks=True,
                    copy_function=lambda f, d: copy2_with_progress(
                        f, d, pbar.update
                    ),
                )
    return dest_dir
