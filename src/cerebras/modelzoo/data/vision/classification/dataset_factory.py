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

import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision
from pydantic import PositiveInt
from torch.utils.data import Subset
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.vision import StandardTransform

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.vision.classification.mixup import (
    RandomCutmix,
    RandomMixup,
)
from cerebras.modelzoo.data.vision.classification.sampler import (
    RepeatedAugSampler,
)
from cerebras.modelzoo.data.vision.preprocessing import get_preprocess_transform
from cerebras.modelzoo.data.vision.transforms import LambdaWithParam
from cerebras.modelzoo.data.vision.utils import is_gpu_distributed, task_id


class VisionClassificationProcessorConfig(DataConfig):

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = "."
    """ The path to the data """

    image_size: List[int] = [224, 224]
    """ The size of the images in the dataset """

    num_classes: int = ...
    """ The number of classification classes in the dataset """

    batch_size: int = 128
    """ Global batch size for the dataloader """

    shuffle: bool = True
    """ Whether or not to shuffle the dataset. """

    shuffle_seed: Optional[int] = None
    """ The seed used for deterministic shuffling. """

    drop_last: bool = True
    """
    Similar to the PyTorch drop_last setting except that samples that when set
    to `True`, samples that would have been dropped at the end of one epoch are
    yielded at the start of the next epoch so that there is no data loss. This
    is necessary for a data ordering that is independent of the distributed
    setup being used.
    """

    num_workers: int = 0
    """ How many subprocesses to use for data loading """

    prefetch_factor: Optional[int] = 10
    """ Number of batches loaded in advance by each worker """

    persistent_workers: Optional[bool] = True
    """ Whether or not to keep workers persistent between epochs. """

    sampler: str = "random"
    """ Type of data sampler to use"""

    ra_sampler_num_repeat: PositiveInt = 3
    """ Number of repeats for Repeated Augmentation sampler."""

    mixup_alpha: float = 0.1
    """ Alpha parameter for the mixup transform."""

    cutmix_alpha: float = 0.1
    """ Alpha parameter for the cutmix transform."""

    noaugment: bool = False
    """ 
    Indicates to skip augmentation as part of preprocessing.
    """

    transforms: List[dict] = ...
    """ List of transforms for preprocessing """


class VisionClassificationProcessor:
    def __init__(self, config: VisionClassificationProcessorConfig):
        if isinstance(config, dict):
            config = VisionClassificationProcessorConfig(**config)
        # data settings
        self.data_dir = config.data_dir
        self.image_size = config.image_size
        self.num_classes = config.num_classes
        self.allowable_split = None

        # params for preprocessing dataset
        self.pp_params = dict()
        self.pp_params["noaugment"] = config.noaugment
        self.pp_params["transforms"] = config.transforms

        # params for data loader
        self.global_batch_size = config.batch_size
        self.batch_size = get_streaming_batch_size(self.global_batch_size)

        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        if self.shuffle_seed is not None:
            torch.manual_seed(self.shuffle_seed)
        self.drop_last = config.drop_last

        # multi-processing params.
        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers
        self.distributed = is_gpu_distributed()

        # sampler
        self.sampler = config.sampler
        self.ra_sampler_num_repeat = config.ra_sampler_num_repeat
        self.mixup_alpha = config.mixup_alpha
        self.cutmix_alpha = config.cutmix_alpha

    def create_dataloader(self):
        dataset = self.create_dataset()

        mixup_transforms = []
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(self.num_classes, p=1.0, alpha=self.mixup_alpha)
            )
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(self.num_classes, p=1.0, alpha=self.cutmix_alpha)
            )
        if mixup_transforms:
            mixup_fn = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixup_fn(*default_collate(batch))

        if self.distributed:
            # distributed samplers require a seed
            if self.shuffle_seed is None:
                self.shuffle_seed = 0

            if self.sampler == "repeated-aug":
                data_sampler = RepeatedAugSampler(
                    dataset,
                    shuffle=self.shuffle,
                    seed=self.shuffle_seed,
                    num_repeats=self.ra_sampler_num_repeat,
                    batch_size=self.batch_size,
                )
            else:
                data_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    shuffle=self.shuffle,
                    seed=self.shuffle_seed,
                )
        else:
            if self.shuffle:
                data_sampler = torch.utils.data.RandomSampler(
                    dataset, generator=self._generator_fn()
                )
            else:
                data_sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=data_sampler,
            num_workers=self.num_workers,
            pin_memory=self.distributed,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            worker_init_fn=self._worker_init_fn,
        )
        return dataloader

    def create_dataset(self):
        raise NotImplementedError(
            "create_dataset must be implemented in a child class!!"
        )

    def _get_target_transform(self, x, *args, **kwargs):
        return np.int32(x)

    def process_transform(self, use_training_transforms=True):
        if self.pp_params["noaugment"]:
            transform_specs = [
                {"name": "resize", "size": self.image_size},
                {"name": "to_tensor"},
            ]
            logging.warning(
                "User specified `noaugment=True`. The input data will only be "
                "resized to `image_size` and converted to tensor."
            )
            self.pp_params["transforms"] = transform_specs
        transform = get_preprocess_transform(self.pp_params)
        target_transform = LambdaWithParam(self._get_target_transform)

        return transform, target_transform

    def split_dataset(self, dataset, split_percent, seed):
        num_sample = len(dataset)
        rng = np.random.default_rng(seed)
        sample_idx = self.create_shuffled_idx(num_sample, rng)

        split_idx = [0]
        if sum(split_percent) != 100:
            raise ValueError(
                f"Sum of split percentage must be 100%! Got {sum(split_percent)}"
            )

        for sp in split_percent[:-1]:
            offset = num_sample * sp // 100
            new_end = split_idx[-1] + offset
            split_idx.append(new_end)
        split_idx.append(num_sample)

        return [
            VisionSubset(dataset, sample_idx[start:end])
            for start, end in zip(split_idx[:-1], split_idx[1:])
        ]

    def create_shuffled_idx(self, num_sample, rng):
        shuffled_idx = np.arange(num_sample)
        rng.shuffle(shuffled_idx)
        return shuffled_idx

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.shuffle_seed is not None:
            np.random.seed(self.shuffle_seed + worker_id)

    def _generator_fn(self):
        generator_fn = None

        if self.shuffle_seed is not None:
            seed = self.shuffle_seed + task_id()
            generator_fn = torch.Generator(device="cpu")
            generator_fn.manual_seed(seed)

        return generator_fn


class VisionSubset(Subset):
    def __init__(self, dataset, indices):
        assert isinstance(
            dataset, torchvision.datasets.VisionDataset
        ), f"Dataset must be type VisionDataset, but got {type(dataset)} instead."
        super().__init__(dataset, indices)

    def set_transforms(
        self, transforms=None, transform=None, target_transform=None
    ):
        """
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        """
        has_transforms = transforms is not None
        has_separate_transform = (
            transform is not None or target_transform is not None
        )
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can "
                "be passed as argument"
            )

        if has_separate_transform:
            # uses previous transform and target_transform if no new fcns are specified
            if transform is not None:
                self.dataset.transform = transform

            if target_transform is not None:
                self.dataset.target_transform = target_transform

            self.dataset.transforms = StandardTransform(
                transform, target_transform
            )
        elif has_transforms:
            self.dataset.transforms = transforms

    def truncate_to_idx(self, new_length):
        self.indices = self.indices[:new_length]
