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

"""Pytorch DPR Dataloader."""

from typing import Literal

import torch
from torch.utils.data import default_collate

from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessor,
    HDF5IterableDataProcessorConfig,
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


class DPRHDF5DataProcessorConfig(HDF5IterableDataProcessorConfig):
    data_processor: Literal["DPRHDF5DataProcessor"]


class DPRHDF5DataProcessor(HDF5IterableDataProcessor):
    """
    A HDF5 dataset processor for DPR training.
    Loads data from HDF5 files.

    Note that this data-processor aims to return twice as many context-input-ids
    as question-input-ids because of hard-negatives. We cannot return tensors
    with different batch-dimensions, so we return [batch_size, 2, MSL] instead of
    [2*batch_size, MSL]. Then we reshape in the model code; please see the
    comment in dpr_model.py for more details.

    """

    name: Literal["DPRHDF5DataProcessor"]

    def __init__(self, config: DPRHDF5DataProcessorConfig):
        if isinstance(config, dict):
            config = DPRHDF5DataProcessorConfig(**config)

        super().__init__(config)

    def collate_fn(self, batch):
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
        See the comment in dpr_model.py and dpr_loss.py for more details on the shapes.
        """
        batch = default_collate(batch)
        batch_size = batch["questions_input_ids"].shape[0]
        num_context = batch["ctx_input_ids"].shape[1]
        batch["labels"] = torch.arange(batch_size) * num_context
        batch["context_labels"] = torch.arange(batch_size)

        return batch
