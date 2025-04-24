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
import random
from abc import abstractmethod
from typing import List, Literal, Optional, Union

import torch
from torchvision import transforms

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.vision.segmentation.preprocessing_utils import (
    normalize_tensor_transform,
)
from cerebras.modelzoo.data.vision.utils import (
    FastDataLoader,
    create_worker_cache,
    num_tasks,
    task_id,
)


class Hdf5BaseDataProcessorConfig(DataConfig):
    data_processor: Literal["Hdf5BaseDataProcessor"]

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    """Path to dataset HDF5 files"""

    use_worker_cache: bool = False

    num_classes: Optional[int] = None
    """Maximum length of the sequence to generate"""

    loss: Optional[Literal["bce", "multilabel_bce", "ssce", "ssce_dice"]] = None

    normalize_data_method: Optional[
        Literal["zero_centered", "zero_one", "standard_score"]
    ] = None
    """Data normalization method"""

    shuffle: bool = True
    """"Flag to enable data shuffling"""

    shuffle_buffer: Optional[int] = None
    """Size of shuffle buffer in samples."""

    shuffle_seed: Optional[int] = None
    """Shuffle seed"""

    augment_data: bool = True

    image_shape: Optional[List[int]] = None

    batch_size: int = ...
    """Batch size"""

    num_workers: int = 0
    """How many subprocesses to use for data loading."""

    drop_last: bool = True
    """If True and the dataset size is not divisible by the
    batch size, the last incomplete batch will be dropped."""

    prefetch_factor: Optional[int] = 10
    """Number of samples loaded in advance by each worker."""

    persistent_workers: bool = True
    """If True, the data loader will not shutdown the worker
    processes after a dataset has been consumed once."""

    use_fast_dataloader: bool = False

    duplicate_act_worker_data: bool = False

    def post_init(self, context):
        if self.shuffle_buffer is None:
            self.shuffle_buffer = 10 * self.batch_size

        model_config = context.get("model", {}).get("config")

        if model_config is not None:
            if hasattr(model_config, "image_shape"):
                self.image_shape = model_config.image_shape

            if hasattr(model_config, "num_classes"):
                self.num_classes = model_config.num_classes

            if "loss" in self.model_fields_set:
                logging.warning(
                    "Loss cannot be set in data configuration. "
                    "Defaulting to value in model configuration."
                )
            if hasattr(model_config, "loss"):
                self.loss = model_config.loss

        if any(
            x is None
            for x in [
                self.image_shape,
                self.num_classes,
                self.loss,
            ]
        ):
            raise ValueError(
                "image_size, num_classes and loss must "
                "be configured from the model config."
            )


class Hdf5BaseDataProcessor(torch.utils.data.Dataset):
    """
    A HDF5 dataset processor for UNet HDF dataset.
    Performs on-the-fly augmentation of image and labek.

    Functionality includes:
        Reading data from HDF5 documents
        Augmenting data
    """

    def __init__(self, config: Hdf5BaseDataProcessorConfig):
        super(Hdf5BaseDataProcessor, self).__init__()

        use_worker_cache = config.use_worker_cache
        self.data_dir = config.data_dir
        if use_worker_cache and dist.is_streamer():
            if not cstorch.use_cs():
                raise RuntimeError(
                    "use_worker_cache not supported for non-CS runs"
                )
            else:
                self.data_dir = create_worker_cache(self.data_dir)

        self.num_classes = config.num_classes
        self.normalize_data_method = config.normalize_data_method
        if self.normalize_data_method:
            # Normalize
            self.normalize_transform = transforms.Lambda(
                self._apply_normalization
            )

        self.image_shape = config.image_shape  # of format (H, W, C)
        (
            self.tgt_image_height,
            self.tgt_image_width,
            self.channels,
        ) = self.image_shape

        self.loss_type = config.loss

        self.shuffle_seed = config.shuffle_seed
        if self.shuffle_seed is not None:
            torch.manual_seed(self.shuffle_seed)

        self.augment_data = config.augment_data
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.shuffle = config.shuffle

        self.shuffle_buffer = config.shuffle_buffer

        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        self.mp_type = cstorch.amp.get_floating_point_dtype()

        self.num_tasks = dist.num_streamers() if dist.is_streamer() else 1
        self.task_id = dist.get_streaming_rank() if dist.is_streamer() else 0

        # set later once processor gets a call to create a dataloader
        self.num_examples = 0
        self.files_in_this_task = []

        self.use_fast_dataloader = config.use_fast_dataloader

        # Each activation worker can access entire dataset when True
        self.duplicate_act_worker_data = config.duplicate_act_worker_data
        self.disable_sharding = False

    @abstractmethod
    def _shard_files(self, is_training=False):
        pass

    @abstractmethod
    def _load_buffer(self, data_partitions):
        pass

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def _shard_dataset(self, worker_id, num_workers):
        pass

    def __len__(self):
        """
        Returns the len of dataset on the task process.
        """
        return self.num_examples

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single-process
            worker_id = 0
            num_workers = 1

        self.data_partitions = self._maybe_shard_dataset(num_workers)

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        is_training = self.split == "train"
        self._shard_files(is_training)
        generator_fn = torch.Generator(device="cpu")

        if self.batch_size > self.num_examples // num_tasks():
            print(
                f"Dataset size: {len(self)} too small for num_tasks: {num_tasks()} and batch_size: {self.batch_size}, using duplicate data for activation workers..."
            )
            self.disable_sharding = True

        if self.shuffle:
            if self.duplicate_act_worker_data or self.disable_sharding:
                if self.shuffle_seed is None:
                    seed = task_id()
                else:
                    seed = self.shuffle_seed + task_id()
                random.seed(seed)
                random.shuffle(self.files_in_this_task)
                generator_fn.manual_seed(seed)
            data_sampler = torch.utils.data.RandomSampler(
                self, generator=generator_fn
            )
        else:
            data_sampler = torch.utils.data.SequentialSampler(self)

        if self.use_fast_dataloader:
            dataloader_fn = FastDataLoader
            print("-- Using FastDataLoader -- ")
        else:
            dataloader_fn = torch.utils.data.DataLoader

        data_loader = dataloader_fn(
            self,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=(
                self.prefetch_factor if self.num_workers > 0 else None
            ),
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            worker_init_fn=self._worker_init_fn,
            sampler=data_sampler,
        )
        # set self.data_partitions in case self.num_workers == 0
        if self.num_workers == 0:
            self._worker_init_fn(0)
        return data_loader

    def _apply_normalization(self, x):
        return normalize_tensor_transform(
            x, normalize_data_method=self.normalize_data_method
        )
