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

"""Pytorch GPT2/3 Dataloader"""
import logging

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessor,
)
from cerebras.modelzoo.data.common.HDF5IterableDataset import (
    HDF5IterableDataset,
)


@registry.register_datasetprocessor("GptHDF5DataProcessor")
class GptHDF5DataProcessor(HDF5IterableDataProcessor):
    """
    A HDF5 dataset processor for GPT pre-training.
    Loads data from HDF5 files.
    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:
    - "data_dir" (str or list of str): Path to dataset HDF5 files
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_buffer" (int): Size of shuffle buffer in samples.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of batches loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    - "use_vsl" (bool): Flag to enable variable sequence length training.
       It requires the dataset to have two extra features: the
       `attention_span` of keys and the `position_ids` of tokens.
       Defaults to `False`.
    """

    def __init__(self, params):
        if params.get("max_sequence_length") is not None:
            logging.warning(
                "`max_sequence_length` is not used in for GptHDF5DataProcessor "
                "as it expects the data to be pre tokenized to a desired MSL, "
                "please remove it from the supplied config."
            )

        # HDF5IterableDataset yields samples with the features `input_ids`,
        # `attention_mask`, and `labels`. In the case of gpt models, the name
        # `attention_mask` is a misnomer. In reality it acts as a loss mask
        # and its contents aren't used for any attention masking.
        self.dataset = HDF5IterableDataset(params)

        # The super class will take care of sharding the dataset and creating the dataloader
        super().__init__(params)
