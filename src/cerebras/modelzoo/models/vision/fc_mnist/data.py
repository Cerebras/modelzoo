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

from typing import Any, Literal, Optional

import torch
from pydantic import Field, model_validator
from torchvision import datasets, transforms

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.pytorch_utils import SampleGenerator
from cerebras.modelzoo.config import DataConfig


class MNISTDataProcessorConfig(DataConfig):
    data_processor: Literal["MNISTDataProcessor"]
    """Name of the data processor."""

    split: Optional[Literal["train", "val"]] = None
    """The type of data split to use (train or validation)."""

    batch_size: int = ...
    """Batch size."""

    shuffle: bool = True
    """Whether to shuffle the data."""

    drop_last: bool = Field(True, validation_alias="drop_last_batch")
    """Whether to drop the last incomplete batch."""

    num_workers: int = 0
    """Number of dataloader subrpocesses to use for data loading."""

    use_fake_data: bool = False
    """Whether to use fake data (for testing)."""

    fake_data_seed: Optional[int] = None
    """The random seed to use when `use_fake_data` is True."""

    data_dir: str = "./"
    """The directory to place the downloaded data files."""

    to_float16: Optional[Any] = Field(default=None, deprecated=True)

    @model_validator(mode="after")
    def validate_train(self):
        if self.split is None and not self.use_fake_data:
            raise ValueError(
                "split must be provided when use_fake_data is False"
            )
        return self


class MNISTDataProcessor:
    def __init__(self, config: MNISTDataProcessorConfig):
        if isinstance(config, dict):
            config = MNISTDataProcessorConfig(**config)

        self.config = config

        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.shuffle = config.shuffle

        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        if self.config.use_fake_data:
            num_streamers = dist.num_streamers() if dist.is_streamer() else 1

            if self.config.fake_data_seed is not None:
                torch.manual_seed(self.config.fake_data_seed)
                data = (
                    torch.rand(self.batch_size, 1, 28, 28, dtype=self.dtype),
                    torch.randint(
                        0,
                        10,
                        size=(self.batch_size,),
                        dtype=torch.int32 if cstorch.use_cs() else torch.int64,
                    ),
                )
            else:
                data = (
                    torch.zeros(self.batch_size, 1, 28, 28, dtype=self.dtype),
                    torch.zeros(
                        self.batch_size,
                        dtype=torch.int32 if cstorch.use_cs() else torch.int64,
                    ),
                )
            return SampleGenerator(
                data=data,
                sample_count=60000 // self.batch_size // num_streamers,
            )
        else:
            train_dataset = datasets.MNIST(
                self.config.data_dir,
                train=(self.config.split == "train"),
                download=dist.is_master_ordinal(),
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        transforms.Lambda(
                            lambda x: torch.as_tensor(x, dtype=self.dtype)
                        ),
                    ]
                ),
                target_transform=(
                    transforms.Lambda(
                        lambda x: torch.as_tensor(x, dtype=torch.int32)
                    )
                    if cstorch.use_cs()
                    else None
                ),
            )

            train_sampler = None
            if (
                cstorch.use_cs()
                and dist.num_streamers() > 1
                and dist.is_streamer()
            ):
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=dist.num_streamers(),
                    rank=dist.get_streaming_rank(),
                    shuffle=self.shuffle,
                )

            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                drop_last=self.config.drop_last,
                shuffle=False if train_sampler else self.shuffle,
                num_workers=self.config.num_workers,
            )


class MNISTMapDataProcessorConfig(MNISTDataProcessorConfig):
    data_processor: Literal["MNISTMapDataProcessor"]
    """Name of the data processor."""

    nan_index: Optional[int] = None
    """Index of the batch to generate NaN data."""


class MnistMapFakeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        batch_size,
        sample_count,
        input_dtype,
        shuffle=True,
        nan_sample_index=None,
    ):
        import cerebras.pytorch as cstorch

        self.sample_count = sample_count
        self.sampler = cstorch.utils.data.DistributedSampler(
            self, batch_size=batch_size, shuffle=shuffle
        )

        self.input_dtype = input_dtype
        self.target_dtype = torch.int32 if cstorch.use_cs() else torch.int64
        self.nan_sample_index = nan_sample_index

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        return [
            (
                torch.full((1, 28, 28), torch.nan, dtype=self.input_dtype)
                if idx == self.nan_sample_index
                else torch.ones((1, 28, 28), dtype=self.input_dtype)
            ),
            torch.tensor(1, dtype=self.target_dtype),
        ]


class MNISTMapDataProcessor:
    def __init__(self, config: MNISTMapDataProcessorConfig):
        if isinstance(config, dict):
            config = MNISTMapDataProcessorConfig(**config)

        self.config = config

        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.shuffle = config.shuffle

        self.dtype = cstorch.amp.get_floating_point_dtype()

    def create_dataloader(self):
        from cerebras.modelzoo.data.common.restartable_dataloader import (
            RestartableDataLoader,
        )

        if self.config.nan_index is not None:
            # set shuffle = False because we want to generate NaN data at a specific index
            ds = MnistMapFakeDataset(
                self.batch_size, 100, self.dtype, False, self.config.nan_index
            )
        else:
            ds = MnistMapFakeDataset(
                self.batch_size, 100, self.dtype, self.shuffle
            )
        return RestartableDataLoader(ds, batch_sampler=ds.sampler)
