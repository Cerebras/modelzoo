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

from typing import Any, Callable, Literal, Optional

from pydantic import Field

from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.data.common.h5_map_dataset import (
    MLlamaHDF5Dataset,
    MLlamaHDF5DatasetConfig,
)
from cerebras.modelzoo.data.common.restartable_dataloader import (
    RestartableDataLoader,
)


class MLlamaHDF5MapDataProcessorConfig(MLlamaHDF5DatasetConfig, DataConfig):
    data_processor: Literal["MLlamaHDF5MapDataProcessor"]

    dataset_map_fn: Optional[Callable] = None

    num_workers: int = 0
    """The number of PyTorch processes used in the dataloader."""

    prefetch_factor: int = 10
    """The number of batches to prefetch in the dataloader."""

    persistent_workers: bool = True
    """Whether or not to keep workers persistent between epochs."""

    bos_token_id: Optional[Any] = Field(default=None)
    """The token id for the beginning of sentence token."""

    pos_token_id: Optional[Any] = Field(default=None)
    """The token id for the position token."""

    pad_token_id: Optional[Any] = Field(default=None)
    """The token id for the padding token."""

    micro_batch_size: Optional[Any] = Field(default=None)
    """The micro batch size for mixed precision training."""

    mixed_precision: Optional[Any] = Field(default=None)
    """Whether or not to use mixed precision training."""

    max_image_tiles: Optional[Any] = Field(default=None)
    """The maximum number of image tiles to use."""

    max_num_images: Optional[Any] = Field(default=None)
    """The maximum number of images to use."""

    prob_of_image: Optional[Any] = Field(default=None)
    """The probability of using an image."""

    size: Optional[Any] = Field(default=None)
    """The size of the image tile"""

    def post_init(self, context):
        super().post_init(context)
        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False


class MLlamaHDF5MapDataProcessor:
    def __init__(self, config: MLlamaHDF5MapDataProcessorConfig):
        if isinstance(config, dict):
            config = MLlamaHDF5MapDataProcessorConfig(**config)

        self.config = config

        self.dataset = MLlamaHDF5Dataset(config)

        if not self.dataset.by_sample:
            raise NotImplementedError(
                "Training with 'corpus' format data is not currently supported. "
                "Please switch to 'sample' format."
            )
        if config.use_vsl:
            raise NotImplementedError(
                "Variable sequence length (VSL) training is not "
                "currently supported for MLlama."
            )

        features_list = [
            "input_ids",
            "attention_mask",
            "labels",
            "pixel_values",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "num_tiles",
        ]
        if config.dataset_map_fn is not None:
            self.dataset.map(config.dataset_map_fn)
        else:
            self.dataset.map(
                lambda x: {
                    feature: x[idx] if idx < len(x) else None
                    for idx, feature in enumerate(features_list)
                }
            )

    def create_dataloader(self):
        return RestartableDataLoader(
            self.dataset,
            batch_sampler=self.dataset.sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
            collate_fn=self.dataset.collate_fn,
        )
