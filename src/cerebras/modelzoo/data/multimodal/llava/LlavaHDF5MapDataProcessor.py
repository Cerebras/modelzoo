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

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.common.h5_map_dataset import (
    MultiModalHDF5Dataset,
    RestartableDataLoader,
)


@registry.register_datasetprocessor("LlavaHDF5MapDataProcessor")
class LlavaHDF5MapDataProcessor:
    def __init__(self, params):
        self.dataset = MultiModalHDF5Dataset(params)

        if not self.dataset.by_sample:
            raise NotImplementedError(
                "Training with 'corpus' format data is not currently supported "
                "Please switch to 'sample' format."
            )
        if params.get("use_vsl", False):
            raise NotImplementedError(
                "Variable sequence length (VSL) training is not"
                "currently supported."
            )

        features_list = [
            "text_input_ids",  # input_ids <-> text_input_ids
            "loss_mask",  # input_mask <-> loss_mask
            "labels",
            "attention_span",
            "position_ids",
            "key_padding_mask",  # attention_mask <-> key_padding_mask
        ]
        if "dataset_map_fn" in params:
            self.dataset.map(params["dataset_map_fn"])
        else:
            self.dataset.map(
                lambda x: {
                    feature: x[idx] for idx, feature in enumerate(features_list)
                }
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
