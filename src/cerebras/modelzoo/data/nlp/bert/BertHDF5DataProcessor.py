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

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.common.h5_map_dataset import MLMHDF5Dataset
from cerebras.modelzoo.data.common.restartable_dataloader import (
    RestartableDataLoader,
)


@registry.register_datasetprocessor("BertHDF5DataProcessor")
class BertHDF5DataProcessor:
    def __init__(self, params):
        self.dataset = MLMHDF5Dataset(params)
        use_vsl = params.get("use_vsl", False)
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

        if use_vsl:
            if self.dataset.by_sample:
                features_list["data"].extend(["attention_span", "position_ids"])
            else:
                raise NotImplementedError(
                    "Variable sequence length (VSL) training is not "
                    "currently supported with 'corpus' format data. Please "
                    "switch to 'sample' format data to use VSL."
                )

        if "dataset_map_fn" in params:
            self.dataset.map(params["dataset_map_fn"])
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

        self.num_workers = params.get("num_workers", 0)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)
        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False

    def create_dataloader(self):
        return RestartableDataLoader(
            self.dataset,
            batch_sampler=self.dataset.sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
