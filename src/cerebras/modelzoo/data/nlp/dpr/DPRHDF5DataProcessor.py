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

"""Pytorch DPR Dataloader"""

import torch
from torch.utils.data import default_collate

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessor,
)
from cerebras.modelzoo.data.common.HDF5IterableDataset import (
    HDF5IterableDataset,
)


# Abstracted collate function which allows control over concat dimension.
# See https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
# for original collate_tensor_fn implementation.
def collate_tensor_fn(batch, dim):
    elem = batch[0]
    out = None
    if elem.is_nested:
        raise RuntimeError(
            "Batches of nested tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        shape = elem.shape[:dim] + (len(batch),) + elem.shape[dim:]
        out = elem.new(storage).resize_(*shape)
    return torch.stack(batch, dim, out=out)


@registry.register_datasetprocessor("DPRHDF5DataProcessor")
class DPRHDF5DataProcessor(HDF5IterableDataProcessor):
    """
    A HDF5 dataset processor for DPR training.
    Loads data from HDF5 files.

    Note that this data-processor aims to return twice as many context-input-ids
    as question-input-ids because of hard-negatives. We cannot return tensors
    with different batch-dimensions, so we return [batch_size, 2, MSL] instead of
    [2*batch_size, MSL]. Then we reshape in the model code; please see the
    comment in dpr_model.py for more details.

    :param params: training input parameters for creating dataset.
    :type params: dict
    It should contain the following fields:
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
    """

    def __init__(self, params):
        self.dataset = HDF5IterableDataset(params)

        # The super class will take care of sharding the dataset and creating the dataloader
        super().__init__(params)

    @staticmethod
    def collate_fn(batch):
        """
        :param batch: Contains the input-ids/attention-mask/token-type-ids
            for a batch of questions and their corresponding documents
        :type batch: list[dict]

        This method collects examples and creates the labels for the batch.
        See the comments in dpr_model.py for more details on the shapes,
        however as a brief description, each incoming batch is a list of
        dictionaries as follows:
        [
            {
                question_input_ids: (MSL, ),
                questions_attention_mask: (MSL, ),
                questions_token_type_ids: (MSL, ),
                ctx_input_ids: (num_context, MSL),
                ctx_attention_mask: (num_context, MSL),
                ctx_token_type_ids: (num_context, MSL)
            },
            ...
        ]
        """
        batch = default_collate(batch)
        batch["labels"] = (
            torch.arange(batch["questions_input_ids"].shape[0]) * 2
        )
        return batch
