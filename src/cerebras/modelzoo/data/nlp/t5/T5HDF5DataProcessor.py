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

"""Pytorch T5/Transformer Dataloader."""

from typing import Literal

from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessor,
    HDF5IterableDataProcessorConfig,
)


class T5HDF5DataProcessorConfig(HDF5IterableDataProcessorConfig):
    data_processor: Literal["T5HDF5DataProcessor"]


class T5HDF5DataProcessor(HDF5IterableDataProcessor):
    """
    A HDF5 dataset processor for T5 training.
    Loads data from HDF5 files.
    """

    def __init__(self, config: T5HDF5DataProcessorConfig):
        if isinstance(config, dict):
            config = T5HDF5DataProcessorConfig(**config)

        super().__init__(config)
