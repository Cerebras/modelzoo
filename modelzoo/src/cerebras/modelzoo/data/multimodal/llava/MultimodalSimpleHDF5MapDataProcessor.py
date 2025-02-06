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
    MultimodalSimpleHDF5Dataset,
    MultimodalSimpleHDF5DatasetConfig,
)
from cerebras.modelzoo.data.common.restartable_dataloader import (
    RestartableDataLoader,
)


class MultimodalSimpleHDF5MapDataProcessorConfig(
    MultimodalSimpleHDF5DatasetConfig, DataConfig
):
    data_processor: Literal["MultimodalSimpleHDF5MapDataProcessor"]

    # TODO: Make the Callable type more specific
    dataset_map_fn: Optional[Callable] = None

    # TODO: Make the Callable type more specific
    dataset_map_fn: Optional[Callable] = None

    num_workers: int = 0
    """ The number of PyTorch processes used in the dataloader. """

    prefetch_factor: Optional[int] = 10
    """ The number of batches to prefetch in the dataloader. """

    persistent_workers: bool = True
    """ Whether or not to keep workers persistent between epochs. """

    vocab_size: Optional[Any] = Field(default=None, deprecated=True)
    noaugment: Optional[Any] = Field(default=None, deprecated=True)
    bos_token_id: Optional[Any] = Field(default=None, deprecated=True)
    pos_token_id: Optional[Any] = Field(default=None, deprecated=True)
    pad_token_id: Optional[Any] = Field(default=None, deprecated=True)
    micro_batch_size: Optional[Any] = Field(default=None, deprecated=True)
    mixed_precision: Optional[Any] = Field(default=None, deprecated=True)
    fp16_type: Optional[Any] = Field(default=None, deprecated=True)

    def post_init(self, context):
        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False


class MultimodalSimpleHDF5MapDataProcessor:
    def __init__(self, config: MultimodalSimpleHDF5MapDataProcessorConfig):
        if isinstance(config, dict):
            config = MultimodalSimpleHDF5MapDataProcessorConfig(**config)

        self.config = config

        self.dataset = MultimodalSimpleHDF5Dataset(config)

        if not self.dataset.by_sample:
            raise NotImplementedError(
                "Training with 'corpus' format data is not currently supported "
                "Please switch to 'sample' format."
            )
        if config.use_vsl:
            raise NotImplementedError(
                "Variable sequence length (VSL) training is not"
                "currently supported."
            )

        features_list = [
            "text_input_ids",  # input_ids <-> text_input_ids
            "loss_mask",  # input_mask <-> loss_mask
            "labels",
            "key_padding_mask",  # attention_mask <-> key_padding_mask
            "token_modality_idx",
        ]
        if config.dataset_map_fn is not None:
            self.dataset.map(config.dataset_map_fn)
        else:
            self.dataset.map(
                lambda x: {
                    feature: x[idx] for idx, feature in enumerate(features_list)
                }
            )

    def create_dataloader(self):
        return RestartableDataLoader(
            self.dataset,
            batch_sampler=self.dataset.sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
        )
