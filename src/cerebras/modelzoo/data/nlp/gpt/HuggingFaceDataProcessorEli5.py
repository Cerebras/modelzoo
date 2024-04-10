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

"""Pytorch HuggingFace Eli5 map-style Dataloader"""

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data_preparation.huggingface.HuggingFace_Eli5 import (
    HuggingFace_Eli5,
)
from cerebras.modelzoo.data_preparation.huggingface.HuggingFaceDataProcessor import (
    HuggingFaceDataProcessor,
)


@registry.register_datasetprocessor("HuggingFaceDataProcessorEli5")
class HuggingFaceDataProcessorEli5(HuggingFaceDataProcessor):
    """
    A HuggingFace Eli5 map-style Data Processor.
    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of batches loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    """

    def __init__(self, params):
        num_workers = params.get("num_workers", 0)
        split = params["split"]

        self.dataset, self.data_collator = HuggingFace_Eli5(
            split=split, num_workers=num_workers
        )

        # The super class will take care of sharding the dataset and creating the dataloader
        super().__init__(params)
