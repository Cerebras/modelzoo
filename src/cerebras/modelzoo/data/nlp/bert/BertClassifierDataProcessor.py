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
Data loader for SST2 and MNL (GLUE tasks).
"""

import abc
import csv
import os

import numpy as np
import torch

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.common.input_utils import ShardedSampler
from cerebras.modelzoo.data.nlp.bert.bert_utils import build_vocab
from cerebras.modelzoo.data_preparation.nlp.tokenizers.Tokenization import (
    FullTokenizer,
)

MNLI_LABEL_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


class ClassifierDataset(torch.utils.data.Dataset):
    """
    Base class for dataset that load their raw data from TSV files.
    Child classes must provide read_tsv.

    Args:
        params (dict): List of training input parameters for creating dataset.
        is_training (bool): Indicator for training or validation dataset.
    """

    def __init__(self, params, is_training):
        self.batch_size = get_streaming_batch_size(params["batch_size"])
        self.data_dir = params["data_dir"]
        self.is_training = is_training

        self.vocab_file = params["vocab_file"]
        self.do_lower = params.get("do_lower", False)
        # Get special tokens
        self.special_tokens = {
            "oov_token": "[UNK]",
            "class_token": "[CLS]",
            "document_separator_token": "[SEP]",
        }
        if self.do_lower:
            self.special_tokens = {
                key: value.lower() for key, value in self.special_tokens.items()
            }
        self.tokenizer = FullTokenizer(self.vocab_file, self.do_lower)
        self.vocab, self.vocab_size = build_vocab(
            self.vocab_file, self.do_lower, self.special_tokens["oov_token"]
        )
        # Init tokens_to_id converter.
        self.tokens_to_id = self.vocab.forward

        # Getting indices for special tokens.
        self.special_tokens_indices = {
            key: self.tokens_to_id([value])[0]
            for key, value in self.special_tokens.items()
        }

        # Padding indices.
        self.labels_pad_id = params.get("labels_pad_id", 0)
        self.input_pad_id = params.get("input_pad_id", 0)
        self.attn_mask_pad_id = params.get("attn_mask_pad_id", 0)

        self.max_sequence_length = params["max_sequence_length"]

    def encode_sequence(self, text1, text2=None):
        """
        Tokenizes a single text (if text2 is None) or a pair of texts.
        Truncates and adds special tokens as needed.

        Args:
            text1 (str): First text to encode.
            text2 (str): Second text to encode or `None`.

        Returns:
            A list for `input_ids`, `segment_ids` and `attention_mask`.
            - input_ids (np.array[int.32]): Numpy array with input token indices.
                Shape: (`max_sequence_length`).
            - segment_ids (np.array[int.32]): Numpy array with segment indices.
                Shape: (`max_sequence_length`).
            - attention_mask (np.array[int.32]): Numpy array with input masks.
                Shape: (`max_sequence_length`).
        """
        special_tokens_count = 3 if text2 else 2  # [CLS], [SEP], Optional[SEP]
        max_num_tokens = self.max_sequence_length - special_tokens_count
        # Tokenize and truncate
        tokenized_text1 = self.tokenizer.tokenize(text1.strip())
        tokenized_text2 = (
            self.tokenizer.tokenize(text2.strip()) if text2 else None
        )
        if text2:
            # Truncate the tokens one at a time from the end
            total_len = len(tokenized_text1) + len(tokenized_text2)
            while total_len > max_num_tokens:
                if len(tokenized_text1) > len(tokenized_text2):
                    tokenized_text1.pop()
                else:
                    tokenized_text2.pop()
                total_len -= 1
        else:
            tokenized_text1 = tokenized_text1[:max_num_tokens]
        # convert to ids
        cls_token_id = self.special_tokens_indices["class_token"]
        sep_token_id = self.special_tokens_indices["document_separator_token"]
        token_ids1 = self.tokens_to_id(tokenized_text1)
        input_ids = [cls_token_id] + token_ids1 + [sep_token_id]
        if text2:
            token_ids2 = self.tokens_to_id(tokenized_text2)
            input_ids = input_ids + token_ids2 + [sep_token_id]
        meaningful_tokens_count = len(input_ids)
        pad_count = self.max_sequence_length - meaningful_tokens_count
        input_ids = input_ids + [self.input_pad_id] * pad_count
        # attention mask
        attention_mask = (
            np.ones((self.max_sequence_length,), dtype=np.int32)
            * self.attn_mask_pad_id
        )
        attention_mask[:meaningful_tokens_count] = 1
        # segment ids
        segment_ids = np.zeros((self.max_sequence_length,), dtype=np.int32)
        if text2:
            text2_start = len(token_ids1)
            text2_end = text2_start + len(token_ids2)
            segment_ids[text2_start:text2_end] = 1

        return input_ids, segment_ids, attention_mask

    def read_tsv(self):
        raise NotImplementedError


class SST2Dataset(ClassifierDataset):
    """
    SST2 dataset processor for sentiment analysis.

    Args:
        params (dict): List of training input parameters for creating dataset.
        is_training (bool): Indicator for training or validation dataset.
    """

    def __init__(self, params, is_training):
        super(SST2Dataset, self).__init__(params, is_training)
        self.raw_data = np.array(self.read_tsv())
        self.num_examples = len(self.raw_data)
        self.num_batches = self.num_examples // self.batch_size
        assert self.num_batches > 0, (
            "Dataset does not contain enough samples for one batch, please "
            "choose a smaller batch size."
        )

    def read_tsv(self):
        fname = "train" if self.is_training else "dev"
        tsv_file = os.path.join(self.data_dir, f"{fname}.tsv")
        data = []
        with open(tsv_file, "r") as fid:
            csv_reader = csv.DictReader(
                fid, delimiter="\t", quoting=csv.QUOTE_NONE
            )
            for row in csv_reader:
                sst_data = [row["sentence"], row["label"]]
                data.append(sst_data)
        return data

    def __getitem__(self, idx):
        """
        For each text, raw_label sample in the data do:
            1. Tokenize and truncate
            2. Add special tokens
            3. Convert tokens to ids
            4. Create attention mask
            5. Create a feature dict with:
                - input_ids: np.array[int32] input tokens indices
                    shape: (max_sequence_length, )
                attention_mask:  np.array[int32] attention masks
                    shape: (max_sequence_length, )
                token_type_ids:  np.array[int32] segment ids
                    shape: (max_sequence_length, )
                labels: int32 scalar indicating the sentiment

        Returns:
            A dict with features.
        """
        text, raw_label = self.raw_data[idx]
        (
            input_ids,
            segment_ids,
            attention_mask,
        ) = self.encode_sequence(text)

        features = {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "attention_mask": attention_mask,
            "token_type_ids": segment_ids,
            "labels": np.array(int(raw_label), dtype=np.int32),
        }
        return features

    def __len__(self):
        return self.num_examples


class MNLIDataset(ClassifierDataset):
    """
    SST2 dataset processor for sentiment analysis.

    Args:
        params (dict): List of training input parameters for creating dataset.
        is_training (bool): Indicator for training or validation dataset.
    """

    def __init__(self, params, is_training):
        super(MNLIDataset, self).__init__(params, is_training)
        self.raw_data = np.array(self.read_tsv())
        self.num_examples = len(self.raw_data)
        self.num_batches = self.num_examples // self.batch_size
        assert self.num_batches > 0, (
            "Dataset does not contain enough samples for one batch, please "
            "choose a smaller batch size."
        )

    def read_tsv(self):
        fnames = ["train"]
        if not self.is_training:
            # MNLI has two validation sets:
            #     - the matched set comes from the same domains as training set
            #     - the mismatched set comes from different domains
            fnames = ["dev_matched", "dev_mismatched"]
        data = []
        for fname in fnames:
            tsv_file = os.path.join(self.data_dir, f"{fname}.tsv")
            with open(tsv_file, "r") as fid:
                csv_reader = csv.DictReader(
                    fid, delimiter="\t", quoting=csv.QUOTE_NONE
                )
                # During eval we concatenate the two validation sets. Before
                # doing so, we give each example an "is_matched" label so
                # that during eval we can measure matched and mismatched
                # accuracies separately.
                is_matched = 0 if "mismatched" in fname else 1
                for row in csv_reader:
                    mnli_data = [
                        row["sentence1"],
                        row["sentence2"],
                        row["gold_label"],
                        is_matched,
                    ]
                    data.append(mnli_data)
        return data

    def __getitem__(self, idx):
        """
        For each text, raw_label sample in the data do:
            1. Tokenize sentence a and sentence b, truncate
            2. Add special tokens
            3. Convert tokens to ids
            4. Create attention mask
            5. Create a feature dict with:
                - input_ids: np.array[int32] input tokens indices
                    shape: (max_sequence_length, )
                attention_mask:  np.array[int32] attention masks
                    shape: (max_sequence_length, )
                token_type_ids:  np.array[int32] segment ids
                    shape: (max_sequence_length, )
                labels: int32 scalar indicating the sentiment

        Returns:
            A dict with features.
        """
        text1, text2, raw_label, is_matched = self.raw_data[idx]
        (
            input_ids,
            segment_ids,
            attention_mask,
        ) = self.encode_sequence(text1, text2)

        features = {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "attention_mask": attention_mask,
            "token_type_ids": segment_ids,
            "labels": np.array(MNLI_LABEL_IDX[raw_label], dtype=np.int32),
        }
        # Add a field for is_matched on validation set
        if not self.is_training:
            features["is_matched"] = np.array(is_matched, dtype=np.int32)
            is_mismatched = 1 - np.array(is_matched, dtype=np.int32)
            features["is_mismatched"] = is_mismatched.astype(np.int32)
        return features

    def __len__(self):
        return self.num_examples


class DataProcessor(abc.ABC):
    """
    Base class for processors that load their raw data from TFDS.
    Child classes must provide map_fn, name.

    Args:
        data_params (dict): Input parameters for creating dataset.
            Expects the following fields:
                - "vocab_file" (str): Path to the vocab file.
                - "data_dir" (str): Path to directory containing the TF Records.
                - "batch_size" (int): Batch size.
                - "max_sequence_length" (int): Maximum length of the sequence.
                - "shuffle" (bool): Flag to enable data shuffling.
                - "shuffle_seed" (int): Shuffle seed.
                - "shuffle_buffer" (int): Shuffle buffer size.
                - "do_lower" (bool): Flag to lower case the texts.
                - "num_workers" (int):  How many subprocesses to use for data loading.
                - "drop_last" (bool): If True and the dataset size is not divisible
                    by the batch size, the last incomplete batch will be dropped.
                - "prefetch_factor" (int): Number of samples loaded in advance by each worker.
                - "persistent_workers" (bool): If True, the data loader will not shutdown
                    the worker processes after a dataset has been consumed once.
        model_params (dict, optional): Model parameters for creating the dataset, unused.
    """

    def __init__(self, data_params, model_params) -> None:
        self.data_params = data_params
        self.model_params = model_params
        self.batch_size = get_streaming_batch_size(data_params["batch_size"])
        self.shuffle = data_params.get("shuffle", True)
        self.shuffle_seed = data_params.get("shuffle_seed", None)
        self.num_workers = data_params.get("num_workers", 0)
        self.drop_last = data_params.get("drop_last", True)
        self.prefetch_factor = data_params.get("prefetch_factor", 10)
        self.persistent_workers = data_params.get("persistent_workers", True)
        self.dataset = None

    @abc.abstractmethod
    def create_dataset(self, is_training):
        raise NotImplementedError(
            "Please override this method in the base class to create the dataset."
        )

    def create_dataloader(self, is_training=True):
        self.create_dataset(is_training=is_training)
        assert self.dataset, "Unexpected error, dataset is None."
        sharded_sampler = ShardedSampler(
            self.dataset,
            self.shuffle,
            self.shuffle_seed,
            self.drop_last,
        )
        if self.num_workers:
            # prefetch factor only allowed with `num_workers > 0`
            return torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=sharded_sampler,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, drop_last=self.drop_last
        )


@registry.register_datasetprocessor("SST2DataProcessor")
class SST2DataProcessor(DataProcessor):
    """
    The data processor responsible for creating the SST2 dataloader instance.
    """

    def __init__(self, data_params, model_params) -> None:
        super(SST2DataProcessor, self).__init__(data_params, model_params)

    def create_dataset(self, is_training=True):
        self.dataset = SST2Dataset(self.data_params, is_training)


@registry.register_datasetprocessor("MNLIDataProcessor")
class MNLIDataProcessor(DataProcessor):
    """
    The data processor responsible for creating the MNLI dataloader instance.
    """

    def __init__(self, data_params, model_params) -> None:
        super(MNLIDataProcessor, self).__init__(data_params, model_params)

    def create_dataset(self, is_training=True):
        self.dataset = MNLIDataset(self.data_params, is_training)
