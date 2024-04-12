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

import random

import torch
from torch.utils.data import DataLoader, Dataset

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size

# vocabulary that represents the English alphabet a to z
VOCABS = [i for i in range(26)]


class AlphabetDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, unpacked_data):
        tensor_data = torch.tensor(unpacked_data, dtype=torch.int32)
        attention_mask = torch.ones(
            (tensor_data.shape[0], self.seq_length), dtype=torch.int32
        )
        return {
            "input_ids": tensor_data[:, :-1].contiguous(),
            "target_ids": tensor_data[:, 1:].contiguous(),
            "attention_mask": attention_mask,
        }


def train_input_dataloader(params):
    seed = params["runconfig"]["seed"]
    torch.manual_seed(seed)
    input_params = params["train_input"]
    num_samples = input_params["num_samples"]
    seq_length = input_params["seq_length"]
    batch_size = get_streaming_batch_size(input_params["batch_size"])
    train_data = []
    for _ in range(num_samples):
        start_index = random.randint(0, len(VOCABS) - seq_length - 1)
        end_index = start_index + seq_length + 1
        train_data.append(VOCABS[start_index:end_index])
    train_dataset = AlphabetDataset(train_data, seq_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
    )
    return train_dataloader


def eval_input_dataloader(params):
    seed = params["runconfig"]["seed"]
    torch.manual_seed(seed)
    input_params = params["eval_input"]
    num_samples = input_params["num_samples"]
    seq_length = input_params["seq_length"]
    batch_size = input_params["batch_size"]
    test_data = []
    for _ in range(num_samples):
        start_index = random.randint(0, len(VOCABS) - seq_length - 1)
        end_index = start_index + seq_length + 1
        test_data.append(VOCABS[start_index:end_index])

    test_dataset = AlphabetDataset(test_data, seq_length)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=test_dataset.collate,
    )
    return test_dataloader
