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

import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from torchvision.datasets import DatasetFolder

import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.data.vision.diffusion.config import (
    DiffusionLatentImageNet1KProcessorConfig,
)
from cerebras.modelzoo.data.vision.diffusion.DiffusionBaseProcessor import (
    DiffusionBaseProcessor,
)
from cerebras.modelzoo.data.vision.utils import create_worker_cache

FILE_EXTENSIONS = (".npz", ".npy")


class CategoricalDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, probs=None, seed=None):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.probs = probs
        if self.probs is None:
            self.probs = [1 / self.num_datasets] * self.num_datasets

        if not isinstance(self.probs, torch.Tensor):
            self.probs = torch.tensor(self.probs)

        assert (
            len(self.probs) == self.num_datasets
        ), f"Probability values(={len(self.probs)}) != number of datasets(={self.num_datasets})"

        assert (
            torch.sum(self.probs) == 1.0
        ), f"Probability values don't add up to 1.0"

        self.len_datasets = [len(ds) for ds in self.datasets]
        self.max_len = max(self.len_datasets)
        if seed is None:
            # large random number chosen as `high` upper bound
            seed = torch.randint(0, 2147483647, (1,), dtype=torch.int64).item()
        self.seed = seed
        self.generator = None

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        if self.generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        # Pick a dataset
        ds_id = torch.multinomial(self.probs, 1, generator=self.generator)
        ds = self.datasets[ds_id]

        # get sample from dataset selected
        sample_id = idx % self.len_datasets[ds_id]
        return ds[sample_id]


class ImageNetLatentDataset(DatasetFolder):
    def __init__(
        self,
        root: str,
        split: str,
        latent_size: List,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        strict_check: Optional[bool] = False,
    ):
        split_folder = os.path.join(root, split)
        self.strict_check = strict_check
        self.latent_size = latent_size

        super().__init__(
            split_folder,
            self.loader,
            FILE_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=None,
        )

    def loader(self, path: str):
        data = np.load(path)
        return data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        latent = torch.from_numpy(sample["vae_output"])
        label = torch.from_numpy(sample["label"])

        assert (
            list(latent.shape) == self.latent_size
        ), f"Mismatch between shapes {latent.shape} vs expected shape:{self.latent_size}"

        if self.strict_check:
            assert (
                sample["dest_path"] == path
            ), f"Mismatch between image and latent files, please check data creation process."
            assert (
                label == target
            ), f"Mismatch between labels written to npz file and inferred according to folder structure"
        if self.transform is not None:
            latent = self.transform(latent)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return latent, target


class DiffusionLatentImageNet1KProcessor(DiffusionBaseProcessor):
    def __init__(self, config: DiffusionLatentImageNet1KProcessorConfig):
        if isinstance(config, dict):
            config = DiffusionLatentImageNet1KProcessorConfig(**config)
        super().__init__(config)

        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]
        self.num_classes = 1000

    def create_dataset(self):
        if self.use_worker_cache and dist.is_streamer():
            data_dir = []
            for _dir in self.data_dir:
                data_dir.append(create_worker_cache(_dir))
            self.data_dir = data_dir

        self.check_split_valid(self.split)
        transform, target_transform = self.process_transform()

        dataset_list = []
        for _dir in self.data_dir:
            if not os.path.isdir(os.path.join(_dir, self.split)):
                raise RuntimeError(f"No directory {self.split} under root dir")

            dataset_list.append(
                ImageNetLatentDataset(
                    root=_dir,
                    latent_size=[
                        2 * self.latent_channels,
                        self.latent_height,
                        self.latent_width,
                    ],
                    split=self.split,
                    transform=transform,
                    target_transform=target_transform,
                )
            )

        dataset = CategoricalDataset(dataset_list, seed=self.shuffle_seed)
        return dataset
