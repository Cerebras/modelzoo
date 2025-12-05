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

"""Pytorch GPT2/3 Dataloader."""

import logging

from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessor,
)
from cerebras.modelzoo.data.nlp.gpt.config import GptHDF5DataProcessorConfig


class GptHDF5DataProcessor(HDF5IterableDataProcessor):
    """
    A HDF5 dataset processor for GPT pre-training.
    Loads data from HDF5 files.

    Args:
        config: The configuration object for the GPT HDF5 data processor.
    """

    def __init__(self, config: GptHDF5DataProcessorConfig):
        if isinstance(config, dict):
            config = GptHDF5DataProcessorConfig(**config)

        if config.max_sequence_length is not None:
            logging.warning(
                "`max_sequence_length` is not used in for GptHDF5DataProcessor "
                "as it expects the data to be pre tokenized to a desired MSL, "
                "please remove it from the supplied config."
            )

        # The super class will take care of sharding the dataset and creating the dataloader
        super().__init__(config)
