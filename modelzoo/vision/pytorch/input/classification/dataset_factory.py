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

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.vision import StandardTransform

from modelzoo.vision.pytorch.input.classification.mixup import (
    RandomCutmix,
    RandomMixup,
)
from modelzoo.vision.pytorch.input.classification.preprocessing import (
    get_preprocess_transform,
)
from modelzoo.vision.pytorch.input.classification.sampler import (
    RepeatedAugSampler,
)
from modelzoo.vision.pytorch.input.classification.utils import (
    create_preprocessing_params_with_defaults,
)
from modelzoo.vision.pytorch.input.transforms import LambdaWithParam
from modelzoo.vision.pytorch.input.utils import is_gpu_distributed


class Processor:
    def __init__(self, params):
        # data settings
        self.data_dir = params.get("data_dir", ".")
        self.image_size = params.get("image_size", 224)
        self.num_classes = params.get("num_classes")
        self.allowable_split = None

        # params for preprocessing dataset
        self.pp_params = create_preprocessing_params_with_defaults(params)

        # params for data loader
        self.batch_size = params.get("batch_size", 128)
        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", 0)
        self.drop_last = params.get("drop_last", True)

        # multi-processing params.
        self.num_workers = params.get("num_workers", 0)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)
        self.distributed = is_gpu_distributed()

        # sampler
        self.sampler = params.get("sampler", "random")
        self.ra_sampler_num_repeat = params.get("ra_sampler_num_repeat", 3)
        self.mixup_alpha = params.get("mixup_alpha", 0.1)
        self.cutmix_alpha = params.get("cutmix_alpha", 0.1)

    def create_dataloader(self, dataset, is_training=False):
        assert (
            isinstance(dataset, torchvision.datasets.VisionDataset)
            or isinstance(dataset, VisionSubset)
            or isinstance(dataset, torch.utils.data.Subset)
        ), f"Got {type(dataset)} but dataset must be type VisionDataset, "
        "VisionSubset, or torch.utils.data.Subset"
        shuffle = self.shuffle and is_training

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
            if self.sampler == "repeated-aug":
                data_sampler = RepeatedAugSampler(
                    dataset,
                    shuffle=shuffle,
                    seed=self.shuffle_seed,
                    num_repeats=self.ra_sampler_num_repeat,
                    batch_size=self.batch_size,
                )
            else:
                data_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=shuffle, seed=self.shuffle_seed,
                )
        else:
            if shuffle:
                data_sampler = torch.utils.data.RandomSampler(dataset)
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
        )
        return dataloader

    def create_dataset(self, use_training_transforms=True, split="train"):
        raise NotImplementedError(
            "create_dataset must be implemented in a child class!!"
        )

    def _get_target_transform(self, x, *args, **kwargs):
        return np.int32(x)

    def process_transform(self, use_training_transforms=True):
        transform = get_preprocess_transform(
            self.image_size, self.pp_params, use_training_transforms
        )
        target_transform = LambdaWithParam(self._get_target_transform)

        return transform, target_transform

    def check_split_valid(self, split):
        if split not in self.allowable_split:
            raise ValueError(
                f"Dataset split {split} is invalid. Only values in "
                f"{self.allowable_split} are allowed."
            )

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
