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

"""Pytorch HuggingFace Eli5 Iterable Dataloader."""

from typing import Literal

from cerebras.modelzoo.data.common.input_utils import num_tasks
from cerebras.modelzoo.data_preparation.huggingface.HuggingFace_Eli5 import (
    HuggingFace_Eli5,
)
from cerebras.modelzoo.data_preparation.huggingface.HuggingFaceDataProcessor import (
    HuggingFaceDataProcessor,
    HuggingFaceDataProcessorConfig,
)


class HuggingFaceIterableDataProcessorEli5Config(
    HuggingFaceDataProcessorConfig
):
    data_processor: Literal["HuggingFaceIterableDataProcessorEli5"]

    split: str = "train"


class HuggingFaceIterableDataProcessorEli5(HuggingFaceDataProcessor):
    """
    A HuggingFace Eli5 Iterable Data Processor.

    Args:
        config: The configuration object
    """

    def __init__(self, config: HuggingFaceIterableDataProcessorEli5Config):
        if isinstance(config, dict):
            config = HuggingFaceIterableDataProcessorEli5Config(**config)

        self.dataset, self.data_collator = HuggingFace_Eli5(
            split=config.split, num_workers=config.num_workers
        )

        # Convert to an IterableDataset
        self.dataset = self.dataset.to_iterable_dataset(
            num_shards=(num_tasks() * config.num_workers)
        )

        # The super class will take care of sharding the dataset and creating the dataloader
        super().__init__(config, self.dataset)
