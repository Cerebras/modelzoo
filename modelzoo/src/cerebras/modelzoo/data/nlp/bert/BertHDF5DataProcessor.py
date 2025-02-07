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

"""
Processor for PyTorch BERT training.
"""

import json
import os
from typing import Any, Callable, Literal, Optional

from pydantic import Field

from cerebras.modelzoo.data.common.h5_map_dataset import MLMHDF5Dataset
from cerebras.modelzoo.data.common.HDF5DataProcessor import (
    HDF5DataProcessorConfig,
)
from cerebras.modelzoo.data.common.restartable_dataloader import (
    RestartableDataLoader,
)


class BertHDF5DataProcessorConfig(HDF5DataProcessorConfig):
    data_processor: Literal["BertHDF5DataProcessor"]

    dataset_map_fn: Optional[Callable] = None

    num_workers: int = 0
    "The number of PyTorch processes used in the dataloader."

    prefetch_factor: Optional[int] = 10
    "The number of batches to prefetch in the dataloader."

    persistent_workers: bool = True
    "Whether or not to keep workers persistent between epochs."

    # The following fields are deprecated and unused.
    # They will be removed in the future once all configs have been fixed
    vocab_size: Optional[Any] = Field(default=None, deprecated=True)

    def post_init(self, context):
        super().post_init(context)

        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False


class BertHDF5DataProcessor:
    def __init__(self, config: BertHDF5DataProcessorConfig):
        self.dataset = MLMHDF5Dataset(config)
        features_list = {
            "data": ["input_ids", "attention_mask"],
            "labels": ["labels"],
        }

        data_params_path = os.path.join(
            self.dataset.data_dir, "data_params.json"
        )

        self.mlm = False

        with open(data_params_path, 'r') as file:
            data_params = json.load(file)
            dataset_params = data_params.get("dataset", None)
            mlm_with_gather = dataset_params.get("mlm_with_gather", False)
            training_objective = dataset_params.get("training_objective", None)
            self.mlm = (
                (training_objective == 'mlm')
                if training_objective is not None
                else False
            )

        if self.mlm and mlm_with_gather:
            features_list["labels"].extend(
                ["masked_lm_positions", "masked_lm_mask"]
            )

        if config.use_vsl:
            if self.dataset.by_sample:
                features_list["data"].extend(["attention_span", "position_ids"])
            else:
                raise NotImplementedError(
                    "Variable sequence length (VSL) training is not "
                    "currently supported with 'corpus' format data. Please "
                    "switch to 'sample' format data to use VSL."
                )

        if config.dataset_map_fn is not None:
            self.dataset.map(config.dataset_map_fn)
        elif self.dataset.by_sample:
            self.dataset.map(
                lambda x: {
                    feature: x[key][idx]
                    for key, value in features_list.items()
                    for idx, feature in enumerate(value)
                }
            )
        else:
            raise NotImplementedError(
                "MLM mode is not "
                "currently supported with 'corpus' format data. Please "
                "switch to 'sample' format data to use MLM."
            )

        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

    def create_dataloader(self):
        return RestartableDataLoader(
            self.dataset,
            batch_sampler=self.dataset.sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
