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

import random
from typing import Literal, Optional, Union

import numpy as np
import torch
from pydantic import Field
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.data.common.restartable_dataloader import (
    RestartableDataLoader,
)
from cerebras.modelzoo.data.vision.utils import is_gpu_distributed, task_id
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset import (
    ImageNetConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.base_dataset import (
    DatasetConfig,
    SSLTransform,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.Dinov1SyntheticDataset import (
    Dinov1SyntheticDatasetConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.dataset.IJEPASyntheticDataset import (
    IJEPASyntheticDatasetConfig,
)


class BaseImageDataProcessorConfig(DataConfig):
    data_processor: Literal["BaseImageDataProcessor"]
    "Name of the data processor. Must be set to `BaseImageDataProcessor`."

    ssl_transform: SSLTransform
    """
    An image tranform suitable for self-supervised learning.
    One of 
    ```
       Dinov2TransformConfig,
       ImageRandomMultiCropTransformConfig,
       MaskedPatchTransformConfig,
       MultiBlockMaskedContextImageTransformConfig,
    ```.
    """

    dataset: Annotated[
        Union[
            Dinov1SyntheticDatasetConfig,
            IJEPASyntheticDatasetConfig,
            ImageNetConfig,
        ],
        Field(discriminator=DatasetConfig.discriminator),
    ]
    "Dataset configuration to use with this DataProcessor."

    batch_size: int = 128
    "Batch size to serve to the model."

    shuffle: bool = True
    "Whether to shuffle the data."

    shuffle_seed: int = 1456354
    "Random seed to use when shuffling the data."

    drop_last: bool = True
    "Whether to drop the last batch if it is smaller than the batch size."

    num_samples: Optional[int] = None
    "Maximum number of samples to emit. No limit if None."

    pad_last: bool = False
    "Add padding to the last batch in the scenario where the dataset size isn't evenly divisible by the batch size."

    num_workers: int = 0
    "How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process."

    prefetch_factor: Optional[int] = 10
    "Number of samples loaded in advance by each worker."

    persistent_workers: bool = True
    "Determines if the subprocesses should be killed and recreated between each epoch."

    def post_init(self, context):
        model_config = context.get("model", {}).get("config")
        if model_config and hasattr(model_config, "validate_forward_args"):
            model_config.validate_forward_args(self.ssl_transform.output_keys)


class BaseImageDataProcessor:
    def __init__(self, config: BaseImageDataProcessorConfig):
        if isinstance(config, dict):
            config = BaseImageDataProcessorConfig(**config)

        self.ssl_transform_config = config.ssl_transform
        self.dataset_config = config.dataset

        # params for data loader
        # Note: when running via `run.py`,
        # these defaults with `.get` do not get used
        # These are set here if we want to quickly
        # see what the class does without setting too many params.
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        if self.shuffle_seed is not None:
            torch.manual_seed(self.shuffle_seed)
            np.random.seed(self.shuffle_seed)
            random.seed(self.shuffle_seed)
        self.drop_last = config.drop_last
        self.num_samples = config.num_samples
        self.pad_last = config.pad_last

        # multi-processing params.
        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers
        self.distributed = is_gpu_distributed()

    def create_dataset_with_transform(self):
        ssl_transform = self.ssl_transform_config()

        # Only apply SSL transform.
        # All transforms should be routed through registered SSL transform
        # assumption: Assumes all the dataset have `transform` object which
        # applies `transform` to image
        dataset = self.dataset_config.copy(
            update=dict(transform=self.ssl_transform_config)
        )()

        return dataset, ssl_transform

    def create_dataloader(self):

        dataset, ssl_transform_obj = self.create_dataset_with_transform()
        self.ssl_transform = ssl_transform_obj

        if self.distributed:
            # GPU
            # distributed samplers require a seed
            if self.shuffle_seed is None:
                self.shuffle_seed = 0

            data_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=self.shuffle,
                seed=self.shuffle_seed,
            )

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
                collate_fn=ssl_transform_obj.collate_fn,
            )
        else:
            # CSX
            data_sampler = cstorch.utils.data.DistributedSampler(
                dataset,
                shuffle=self.shuffle,
                seed=self.shuffle_seed,
                shard=True,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                num_samples=self.num_samples,
                pad_last=self.pad_last,
            )

            dataloader = RestartableDataLoader(
                dataset,
                batch_sampler=data_sampler,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                collate_fn=ssl_transform_obj.collate_fn,
                worker_init_fn=self._worker_init_fn,
            )

        return dataloader

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.shuffle_seed is not None:
            np.random.seed(self.shuffle_seed + worker_id)
            random.seed(self.shuffle_seed + worker_id)

    def _generator_fn(self):
        generator_fn = None

        if self.shuffle_seed is not None:
            seed = self.shuffle_seed + task_id()
            generator_fn = torch.Generator(device="cpu")
            generator_fn.manual_seed(seed)

        return generator_fn

    @staticmethod
    def visualize_batch(batch, transform_obj):
        transform_obj.visualize_transform(batch)
