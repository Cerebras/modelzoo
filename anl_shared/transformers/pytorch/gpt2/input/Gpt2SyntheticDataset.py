# BEGIN_CEREBRAS_ONLY
"""
Processors for synthetic data for GPT-2
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class Gpt2Dataset(Dataset):
    """
    A class representing a GPT2Dataset inheriting torch.utils.data.Dataset.
    """

    def __init__(self, data, data_processor):
        self.data = data
        self.length = data_processor.num_examples
        super(Gpt2Dataset, self).__init__()

    def __getitem__(self, index):
        feature = {
            "input_ids": self.data["input_ids"][index],
            "attention_mask": self.data["attention_mask"][index],
            "labels": self.data["input_ids"][index],
        }
        return feature

    def __len__(self):
        return self.length


class Gpt2SyntheticDataProcessor:
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
        self.batch_size = params["batch_size"]
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

    def create_dataloader(self, is_training=True):
        """
        Create dataloader.
        :returns: dataloader
        """
        np.random.seed(seed=0)
        data = dict()

        input_mask = np.zeros(
            (self.num_examples, self.max_seq_len), dtype=np.int32
        )
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.num_examples):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            input_mask[i, start_idx : self.max_seq_len] = 1
        data["attention_mask"] = 1 - input_mask

        data["input_ids"] = np.random.randint(
            low=0,
            high=self.vocab_size,
            size=(self.num_examples, self.max_seq_len),
            dtype=np.int32,
        ) * (1 - input_mask)

        dataset = Gpt2Dataset(data, self)
        
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
