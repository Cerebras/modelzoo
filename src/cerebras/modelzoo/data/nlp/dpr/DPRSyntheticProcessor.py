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
Processors for synthetic data for DPR Training
"""
import copy

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.data.common.input_utils import is_distributed


# TODO(SW-106180): Until we have support for arange along batch dim
# on wafer we need to create labels in the data-loader
def collate_w_arange(batch):
    """
    Each incoming batch is a list of dictionaries as follows:
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
    See the comment in dpr_model.py for more details on the shapes.
    """
    batch = default_collate(batch)
    batch["labels"] = torch.arange(batch["questions_input_ids"].shape[0]) * 2

    return batch


class DPRDataset(Dataset):
    """
    A class representing a DPRDataset inheriting torch.utils.data.Dataset.
    """

    def __init__(self, data, data_processor):
        self.data = data
        self.length = data_processor.num_examples
        super(DPRDataset, self).__init__()

    def __getitem__(self, index):
        feature = {
            "questions_input_ids": self.data["questions_input_ids"][index],
            "ctx_input_ids": self.data["ctx_input_ids"][index],
            "questions_attention_mask": self.data["questions_attention_mask"][
                index
            ],
            "ctx_attention_mask": self.data["ctx_attention_mask"][index],
            "questions_token_type_ids": self.data["questions_token_type_ids"][
                index
            ],
            "ctx_token_type_ids": self.data["ctx_token_type_ids"][index],
        }
        return feature

    def __len__(self):
        return self.length


class DPRSyntheticDataProcessor:
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

        questions_input_mask = np.zeros(
            (self.num_examples, self.max_seq_len), dtype=np.int32
        )
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.num_examples):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            questions_input_mask[i, start_idx : self.max_seq_len] = 1
        data["questions_attention_mask"] = 1 - questions_input_mask
        data["questions_token_type_ids"] = copy.deepcopy(
            data["questions_attention_mask"]
        )

        data["questions_input_ids"] = np.random.randint(
            low=0,
            high=self.vocab_size,
            size=(self.num_examples, self.max_seq_len),
            dtype=np.int32,
        ) * (1 - questions_input_mask)

        num_ctx = 2
        # We return context tensors of shape [batch-size, 2, MSL] so that each
        # question has positive document and hard-negative document. See the
        # comment in dpr_model.py for more details. It's possible to consider
        # using more than one hard-negative, but we opt for one currently because
        # DPR showed that it didn't change significantly change ML-quality.
        ctx_input_mask = np.zeros(
            (self.num_examples, num_ctx, self.max_seq_len), dtype=np.int32
        )
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.num_examples):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            ctx_input_mask[i, :, start_idx : self.max_seq_len] = 1
        data["ctx_attention_mask"] = 1 - ctx_input_mask
        data["ctx_token_type_ids"] = copy.deepcopy(data["ctx_attention_mask"])

        data["ctx_input_ids"] = np.random.randint(
            low=0,
            high=self.vocab_size,
            size=(self.num_examples, num_ctx, self.max_seq_len),
            dtype=np.int32,
        ) * (1 - ctx_input_mask)

        dataset = DPRDataset(data, self)

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
            collate_fn=collate_w_arange,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
        )
