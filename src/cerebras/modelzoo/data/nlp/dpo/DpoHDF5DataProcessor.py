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

"""Pytorch DPO Dataloader."""

from typing import List, Literal

from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessor,
    HDF5IterableDataProcessorConfig,
)


class DpoHDF5DataProcessorConfig(HDF5IterableDataProcessorConfig):
    data_processor: Literal["DpoHDF5DataProcessor"]

    features_list: List[str] = [
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
    ]
    "List of features to include in the batch"


class DpoHDF5DataProcessor(HDF5IterableDataProcessor):
    """
    A HDF5 dataset processor for DPO.
    Loads data from HDF5 files.
    """

    def __init__(self, config: DpoHDF5DataProcessorConfig):
        if isinstance(config, dict):
            config = DpoHDF5DataProcessorConfig(**config)

        super().__init__(config)
