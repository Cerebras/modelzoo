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

from typing import Any, Callable, List, Literal, Optional

import numpy as np
from pydantic import Field
from torch.utils.data.dataloader import default_collate

from cerebras.modelzoo.data.common.h5_map_dataset import HDF5Dataset
from cerebras.modelzoo.data.common.HDF5DataProcessor import (
    HDF5DataProcessorConfig,
)
from cerebras.modelzoo.data.common.restartable_dataloader import (
    RestartableDataLoader,
)
from cerebras.modelzoo.data.nlp.gpt.data_transforms import (
    DataTransform,
    wrap_collate_fn,
)


class GptHDF5MapDataProcessorConfig(HDF5DataProcessorConfig):
    data_processor: Literal["GptHDF5MapDataProcessor"]

    dataset_map_fn: Optional[Callable] = None

    data_transforms: Optional[List[DataTransform]] = None

    num_workers: int = 0
    """ The number of PyTorch processes used in the dataloader. """

    prefetch_factor: Optional[int] = 10
    """ The number of batches to prefetch in the dataloader. """

    persistent_workers: bool = True
    """ Whether or not to keep workers persistent between epochs. """

    vocab_size: Optional[Any] = Field(default=None, deprecated=True)
    repeat: Optional[Any] = Field(default=None, deprecated=True)

    def post_init(self, context):
        super().post_init(context)
        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False
        model_config = context.get('model', {}).get('config')

        return_masks = False
        enable_memory_tokens = False
        if model_config is not None:
            if (
                hasattr(model_config, "memory_tokens_config")
                and model_config.memory_tokens_config.memory_tokens_enabled
            ):

                enable_memory_tokens = True
                memory_tokens_config = model_config.memory_tokens_config
                attn_memory_chunk_size = (
                    memory_tokens_config.attn_memory_chunk_size
                )
                num_memory_tokens_per_chunk = (
                    memory_tokens_config.num_memory_tokens_per_chunk
                )
                memory_token_id = memory_tokens_config.memory_token_id
                return_masks = (
                    return_masks
                    or memory_tokens_config.add_qkv_memory_weights
                    or memory_tokens_config.add_extra_embedding
                )

            if hasattr(model_config, "transformer_decoder"):
                for layer_config in model_config.transformer_decoder.layers:
                    if (
                        hasattr(layer_config.self_attn, "memory_tokens_config")
                        and layer_config.self_attn.memory_tokens_config.memory_tokens_enabled
                    ):

                        enable_memory_tokens = True
                        memory_tokens_config = (
                            layer_config.self_attn.memory_tokens_config
                        )
                        attn_memory_chunk_size = (
                            memory_tokens_config.attn_memory_chunk_size
                        )
                        num_memory_tokens_per_chunk = (
                            memory_tokens_config.num_memory_tokens_per_chunk
                        )
                        memory_token_id = memory_tokens_config.memory_token_id
                        return_masks = (
                            return_masks
                            or memory_tokens_config.add_qkv_memory_weights
                            or memory_tokens_config.add_extra_embedding
                        )

        if enable_memory_tokens:
            if not self.data_transforms:
                self.data_transforms = []

            self.data_transforms.append(
                {
                    'name': 'add_memory_tokens',
                    'attn_memory_chunk_size': attn_memory_chunk_size,
                    'num_memory_tokens_per_chunk': num_memory_tokens_per_chunk,
                    'memory_token_id': memory_token_id,
                    'return_masks': return_masks,
                }
            )


class GptHDF5MapDataProcessor:
    """
    A map style dataset for GPT style models.

    Supports data saved on disk in either of the following formats:
        - `(num_tokens,)`, i.e. a set of documents tokenized and concatenated.
            We refer to this as the 'corpus' format in what follows.
        - `(num_sequences, 3, sequence_length)`, i.e. data that has already
            been preprocessed into sequences. We refer to this as the
            'sample' format in what follows.

    Args:
        config: The config used to configure the data processor.
    """

    def __init__(self, config: GptHDF5MapDataProcessorConfig):
        if isinstance(config, dict):
            config = GptHDF5MapDataProcessorConfig(**config)

        self.config = config

        # Note: attention_mask is a misnomer and serves as a loss mask in the
        # model itself. This naming will change in 2.0.
        self.dataset = HDF5Dataset(config)

        features_list = ["input_ids", "attention_mask", "labels"]
        if self.dataset.use_vsl:
            if self.dataset.by_sample:
                features_list.extend(["attention_span", "position_ids"])
            else:
                raise NotImplementedError(
                    "Variable sequence length (VSL) training is not "
                    "currently supported with 'corpus' format data. Please "
                    "switch to 'sample' format data to use VSL."
                )

        if self.config.dataset_map_fn:
            self.dataset.map(self.config.dataset_map_fn)
        elif self.dataset.by_sample:
            self.dataset.map(
                lambda x: {
                    feature: x[idx] for idx, feature in enumerate(features_list)
                }
            )
        else:
            self.dataset.map(
                lambda x: {
                    "input_ids": x[:-1],
                    "labels": x[1:],
                    "attention_mask": np.ones_like(x[:-1]),
                }
            )

        data_transforms = self.config.data_transforms
        self.collate_fn = default_collate
        if data_transforms is not None:
            self.collate_fn = wrap_collate_fn(
                self.collate_fn,
                data_transforms,
                self.dataset.use_vsl,
            )

    def create_dataloader(self):
        return RestartableDataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=self.dataset.sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
        )
