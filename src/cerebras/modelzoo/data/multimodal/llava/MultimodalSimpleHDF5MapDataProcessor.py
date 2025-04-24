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
    "The name of the dataprocessor. Must be set to `MultimodalSimpleHDF5MapDataProcessor`."

    # TODO: Make the Callable type more specific
    dataset_map_fn: Optional[Callable] = None
    """
    Optional lambda function that can be applied on top of data read from H5 files.
    For ex: dataset_map_fn = lambda h5_sample: h5_sample + 1
    In this example, we assume that each sample in H5 file is a tensor.
    the map function reads a sample tensor from H5 file
    and adds one to it and returns the new sample.
    """

    num_workers: int = 0
    "The number of PyTorch processes used in the dataloader."

    prefetch_factor: Optional[int] = 10
    "The number of batches to prefetch in the dataloader."

    persistent_workers: bool = True
    """
    If True, the data loader will not shut down the worker processes 
    after a dataset has been consumed once. 
    This allows to maintain the workers Dataset instances alive.
    """

    vocab_size: Optional[Any] = Field(default=None, deprecated=True)
    "Size of vocabulary used in language model backbone of Multimodal Simple model"

    noaugment: Optional[Any] = Field(default=None, deprecated=True)
    "If True, do not augment the image"

    micro_batch_size: Optional[Any] = Field(default=None, deprecated=True)
    "Batch size to be used in a single forward pass of the model without gradient accumulation"

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
