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
Processors for synthetic data for DPO Training
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.data.common.input_utils import is_distributed


class DPODataset(Dataset):
    """
    A class representing a DPODataset inheriting torch.utils.data.Dataset.
    """

    def __init__(self, data, data_processor):
        self.data = data
        self.length = data_processor.num_examples
        super(DPODataset, self).__init__()

    def __getitem__(self, index):
        feature = {
            "chosen_input_ids": self.data["chosen_input_ids"][index],
            "chosen_attention_mask": self.data["chosen_attention_mask"][index],
            "chosen_labels": self.data["chosen_labels"][index],
            "rejected_input_ids": self.data["rejected_input_ids"][index],
            "rejected_attention_mask": self.data["rejected_attention_mask"][
                index
            ],
            "rejected_labels": self.data["rejected_labels"][index],
        }
        return feature

    def __len__(self):
        return self.length


class DPOSyntheticDataProcessor:
    """
    Synthetic dataset generator.

    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:

    - "num_examples (int): Number of training examples
    - "vocab_size" (int): Vocabulary size
    - "max_seq_length (int): Maximum length of the sequence to generate
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    """

    def __init__(self, params):
        self.num_examples = params["num_examples"]
        self.vocab_size = params["vocab_size"]
        self.max_seq_len = params["max_sequence_length"]
        self.batch_size = get_streaming_batch_size(params["batch_size"])
        self.shuffle = params["shuffle"]
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.input_pad_id = params.get("input_pad_id", None)
        self.label_pad_id = params.get("label_pad_id", None)

        self.sampler = params.get("sampler", None)
        self.batch_sampler = params.get("batch_sampler", None)
        self.num_workers = params.get("num_workers", 8)
        self.pin_memory = params.get("pin_memory", False)
        self.drop_last = params.get("drop_last", True)
        self.timeout = params.get("timeout", 0)

        assert self.batch_size > 0, "Batch size should be positive."

        if self.drop_last and self.num_examples < self.batch_size:
            raise ValueError(
                f"This dataset does not return any batches because number of "
                f"examples in the dataset ({self.num_examples}) is less than "
                f"the batch size ({self.batch_size}) and `drop_last` is True."
            )

    def create_dataloader(self):
        """
        Create dataloader.

        :returns: dataloader
        """
        np.random.seed(seed=0)
        data = dict()

        chosen_input_mask = np.zeros(
            (self.num_examples, self.max_seq_len), dtype=np.int32
        )
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.num_examples):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            chosen_input_mask[i, start_idx : self.max_seq_len] = 1
        data["chosen_attention_mask"] = 1 - chosen_input_mask

        data["chosen_input_ids"] = (
            np.random.randint(
                low=0,
                high=self.vocab_size,
                size=(self.num_examples, self.max_seq_len),
                dtype=np.int32,
            )
            * data["chosen_attention_mask"]
        )

        data["chosen_labels"] = data["chosen_input_ids"]

        rejected_input_mask = np.zeros(
            (self.num_examples, self.max_seq_len), dtype=np.int32
        )
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.num_examples):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            rejected_input_mask[i, start_idx : self.max_seq_len] = 1
        data["rejected_attention_mask"] = 1 - rejected_input_mask

        data["rejected_input_ids"] = (
            np.random.randint(
                low=0,
                high=self.vocab_size,
                size=(self.num_examples, self.max_seq_len),
                dtype=np.int32,
            )
            * data["rejected_attention_mask"]
        )

        data["rejected_labels"] = data["rejected_input_ids"]

        dataset = DPODataset(data, self)

        if is_distributed():
            assert self.sampler is None, "Cannot use sampler in config with DDP"
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=self.shuffle,
                seed=self.shuffle_seed,
            )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
        )
