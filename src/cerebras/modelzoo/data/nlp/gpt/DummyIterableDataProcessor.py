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

"""Pytorch Generic Iterable Dataloader"""

import numpy as np
import torch

from cerebras.modelzoo.data.common.GenericDataProcessor import (
    GenericDataProcessor,
    GenericDataProcessorConfig,
)


class DummyIterableDataset(torch.utils.data.IterableDataset):
    """
    A Dummy iterable torch.utils.data.IterableDataset.
    """

    def __init__(self):
        self.length = 10000
        self.max_seq_len = 128
        self.vocab_size = 32000
        np.random.seed(seed=0)
        self.data = dict()

        input_mask = np.zeros((self.length, self.max_seq_len), dtype=np.int32)
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.length):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            input_mask[i, start_idx : self.max_seq_len] = 1
        self.data["attention_mask"] = 1 - input_mask

        self.data["input_ids"] = np.random.randint(
            low=0,
            high=self.vocab_size,
            size=(self.length, self.max_seq_len),
            dtype=np.int32,
        ) * (1 - input_mask)

        super(DummyIterableDataset, self).__init__()

    def __iter__(self):
        for idx in range(self.length):
            feature = {
                "input_ids": self.data["input_ids"][idx],
                "attention_mask": self.data["attention_mask"][idx],
                "labels": self.data["input_ids"][idx],
            }
            yield feature


class DummyIterableDataProcessor(GenericDataProcessor):

    def __init__(self, config: GenericDataProcessorConfig):
        super().__init__(config, DummyIterableDataset())


class DummyTinyIterableDataset(torch.utils.data.IterableDataset):
    """
    A Dummy iterable torch.utils.data.IterableDataset.
    """

    def __init__(self):
        self.length = 9
        self.max_seq_len = 128
        self.vocab_size = 32000
        np.random.seed(seed=0)
        self.data = dict()

        input_mask = np.zeros((self.length, self.max_seq_len), dtype=np.int32)
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.length):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            input_mask[i, start_idx : self.max_seq_len] = 1
        self.data["attention_mask"] = 1 - input_mask

        self.data["input_ids"] = np.random.randint(
            low=0,
            high=self.length,
            size=(self.length, self.max_seq_len),
            dtype=np.int32,
        ) * (1 - input_mask)

        super(DummyTinyIterableDataset, self).__init__()

    def __iter__(self):
        for idx in range(self.length):
            feature = {
                "input_ids": self.data["input_ids"][idx],
                "attention_mask": self.data["attention_mask"][idx],
                "labels": self.data["input_ids"][idx],
            }
            yield feature


class DummyTinyIterableDataProcessor(GenericDataProcessor):
    """
    A Generic PyTorch Data Processor.
    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "shuffle_buffer" (int): Size of shuffle buffer in samples.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of batches loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    """

    def __init__(self, params):
        self.dataset = DummyTinyIterableDataset()
        super().__init__(params)
