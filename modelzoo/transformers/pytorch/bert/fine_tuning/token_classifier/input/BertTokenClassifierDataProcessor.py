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
Processor for PyTorch BERT fine tuning - Token classifier.
"""
import csv
import json
import os

import numpy as np
import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.transformers.pytorch.bert.input.utils import (
    build_vocab,
    get_meta_data,
    shard_and_shuffle_data,
)
from modelzoo.transformers.pytorch.input_utils import (
    check_sharding_sanity,
    get_data_for_task,
    task_id,
)


class BertTokenClassifierDataProcessor(torch.utils.data.IterableDataset):
    """
    Reads csv file containing the input token ids, and label_ids.
    Creates attention_masks and sedment_ids on the fly
    :param <dict> params: dict containing input parameters for creating dataset.
        Expects the following fields:
            - "vocab_file" (str): Path to the vocab file.
            - "label_vocab_file" (str): Path to json file with class name to class index.
            - "data_dir" (str): Path to directory containing the CSV files.
            - "batch_size" (int): Batch size.
            - "max_sequence_length" (int): Maximum length of the sequence.
            - "repeat" (bool): Flag to enable data repeat.
            - "do_lower" (bool): Flag to lower case the texts.
            - "shuffle" (bool): Flag to enable data shuffling.
            - "shuffle_seed" (int): Shuffle seed.
            - "shuffle_buffer" (int): Shuffle buffer size.
            - "num_workers" (int):  How many subprocesses to use for data loading.
            - "drop_last" (bool): If True and the dataset size is not divisible
                by the batch size, the last incomplete batch will be dropped.
            - "prefetch_factor" (int): Number of samples loaded in advance by each worker.
            - "persistent_workers" (bool): If True, the data loader will not shutdown
                the worker processes after a dataset has been consumed once.
    :pram model_params (dict): Model parameters for creating the dataset.
        Expects the following to be defined:
        - "include_padding_in_loss" (bool): If set to true then a loss mask will be 
            generated such that padding tokens will be included in the loss calculation.
    """

    def __init__(self, params, model_params):
        super(BertTokenClassifierDataProcessor, self).__init__()

        self.use_cs = cm.use_cs()
        # Input params.
        self.meta_data = get_meta_data(params["data_dir"])

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0]
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.batch_size = params["batch_size"]

        self.num_batches = self.num_examples // self.batch_size
        assert (
            self.num_batches > 0
        ), "Dataset does not contain enough samples for one batch. Please choose a smaller batch size"
        self.num_tasks = cm.num_streamers() if cm.is_streamer() else 1
        self.num_batch_per_task = self.num_batches // self.num_tasks

        assert (
            self.num_batch_per_task > 0
        ), "Dataset cannot be evenly distributed across the given tasks. Please choose fewer tasks to run with"

        self.num_examples_per_task = self.num_batch_per_task * self.batch_size
        self.files_in_task = get_data_for_task(
            task_id(),
            self.meta_data_values_cum_sum,
            self.num_examples_per_task,
            self.meta_data_values,
            self.meta_data_filenames,
        )

        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)
        self.repeat = params.get("repeat", False)
        self.mask_whole_word = params.get("mask_whole_word", False)
        self.do_lower = params.get("do_lower", False)

        # Multi-processing params.
        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

        # Check that our sharding will produce at least one batch
        check_sharding_sanity(
            [num_examples for _, num_examples, _ in self.files_in_task],
            self.batch_size,
            self.num_workers,
            self.drop_last,
        )

        self.special_tokens = {
            "oov_token": "[UNK]",
            "class_token": "[CLS]",
            "pad_token": "[PAD]",
            "document_separator_token": "[SEP]",
        }
        if self.do_lower:
            self.special_tokens = {
                key: value.lower() for key, value in self.special_tokens.items()
            }

        # Get vocab file and size.
        self.vocab_file = params["vocab_file"]
        self.vocab, self.vocab_size = build_vocab(
            self.vocab_file, self.do_lower, self.special_tokens["oov_token"]
        )
        self.label_vocab_file = params["label_vocab_file"]
        if not os.path.exists(self.label_vocab_file):
            raise FileNotFoundError(f"{self.label_vocab_file} not found.")
        with open(self.label_vocab_file, "r") as labelmap_fid:
            self.label_map = json.load(labelmap_fid)

        # Init tokenizer.
        self.tokenize = self.vocab.forward

        # Getting indices for special tokens.
        self.special_tokens_indices = {
            key: self.tokenize([value])[0]
            for key, value in self.special_tokens.items()
        }

        # Padding indices.
        # See https://huggingface.co/transformers/glossary.html#labels.
        self.labels_pad_id = params.get(
            "labels_pad_id", self.special_tokens_indices["pad_token"]
        )
        self.input_pad_id = params.get(
            "input_pad_id", self.special_tokens_indices["pad_token"]
        )
        self.attn_mask_pad_id = params.get(
            "attn_mask_pad_id", self.special_tokens_indices["pad_token"]
        )
        assert all(
            pad >= 0
            for pad in [
                self.labels_pad_id,
                self.input_pad_id,
                self.attn_mask_pad_id,
            ]
        ), (
            f"All padding must be non-negative, got"
            f" `labels_pad_id` = {self.labels_pad_id}, `input_pad_id` = {self.input_pad_id},"
            f" `attn_mask_pad_id` = {self.attn_mask_pad_id}."
        )

        self.max_sequence_length = params["max_sequence_length"]

        self.include_padding_in_loss = model_params.get(
            "include_padding_in_loss"
        )

        # Store params.
        self.mp_type = (
            half_dtype_instance.half_dtype
            if params.get("mixed_precision")
            else torch.float32
        )
        self.data_buffer = []
        self.csv_files_per_task_per_worker = []
        self.processed_buffers = 0

    def create_dataloader(self, is_training=True):
        """
        Classmethod to create the dataloader object.
        """
        if self.num_workers:
            dataloader = torch.utils.data.DataLoader(
                self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, drop_last=self.drop_last,
            )

        return dataloader

    def load_buffer(self):
        """
        Generator to read the data in chunks of size of `data_buffer`.

        :returns: Yields the data stored in the `data_buffer`.
        """
        self.data_buffer = []
        while self.processed_buffers < len(self.csv_files_per_task_per_worker):
            (
                current_file_path,
                num_examples,
                start_id,
            ) = self.csv_files_per_task_per_worker[self.processed_buffers]
            with open(current_file_path, "r", newline="") as fid:
                data_reader = csv.DictReader(
                    fid, delimiter=",", quoting=csv.QUOTE_MINIMAL
                )
                for row_id, row in enumerate(data_reader):
                    if start_id <= row_id < start_id + num_examples:
                        self.data_buffer.append(row)
                    else:
                        continue

                    if len(self.data_buffer) == self.shuffle_buffer:
                        if self.shuffle:
                            self.rng.shuffle(self.data_buffer)

                        for ind in range(len(self.data_buffer)):
                            yield self.data_buffer[ind]
                        self.data_buffer = []

                self.processed_buffers += 1

        if self.shuffle:
            self.rng.shuffle(self.data_buffer)

        for ind in range(len(self.data_buffer)):
            yield self.data_buffer[ind]
        self.data_buffer = []

    def __len__(self):
        """
        Returns the length of the dataset on task process.
        """
        return self.num_examples_per_task

    def __iter__(self):
        """
        Iterator over the data to construct input features.

        :return: A tuple with training features:
            * np.array[int.32] input_ids: Numpy array with input token indices.
                Shape: (`max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels.
               Shape: (`max_sequence_length`).
            * np.array[int.32] attention_mask
               Shape: (`max_sequence_length`).
            * np.array[int.32] token_type_ids: Numpy array with segment indices.
               Shape: (`max_sequence_length`).
        """
        (
            self.processed_buffers,
            self.csv_files_per_task_per_worker,
            self.shuffle_seed,
            self.rng,
        ) = shard_and_shuffle_data(
            self.files_in_task, self.shuffle, self.shuffle_seed,
        )

        # Iterate over the data rows to create input features.
        for data_row in self.load_buffer():
            # `data_row` is a dict with keys: ["tokens", "labels""].
            tokens_list, labels_list = parse_ner_row(data_row, self.do_lower)
            features = create_ner_features(
                tokens_list,
                labels_list,
                self.label_map,
                self.max_sequence_length,
                self.input_pad_id,
                self.attn_mask_pad_id,
                self.labels_pad_id,
                self.include_padding_in_loss,
                self.tokenize,
            )

            yield features


def parse_ner_row(data_row, do_lower=False):
    """
    Postprocessing of a row in the CSV file.
    :param: dict data_row: dictionary with an input text tokens and labels.

    :return: tuple: (list of parsed tokens, List of labels).
    """
    tokens = data_row["tokens"].split()
    tokens_list = (
        list(map(lambda token: token.lower(), tokens)) if do_lower else tokens
    )
    labels_list = data_row["labels"].split()
    return tokens_list, labels_list


def create_ner_features(
    tokens_list,
    labels_list,
    label_map,
    max_sequence_length,
    input_pad_id,
    attn_mask_pad_id,
    labels_pad_id,
    include_padding_in_loss,
    tokenize,
):
    """
    Creates the features dict for token classifier model.

    :param list tokens_list: Tokens to process
    :param list labels_list: Labels to process
    :param dict label_map: Dictionary mapping label to int
    :param int max_sequence_length: Maximum sequence length.
    :param int input_pad_id: Input sequence padding id.
    :param int attn_mask_pad_id: Attention mask padding id.
    :param int labels_pad_id: Labels padding id.
    :param bool include_padding_in_loss: Flag to generate loss mask.
    :param callable tokenize: Method to tokenize the input sequence.

    :returns: dict for features which includes keys:
            * 'input_ids': Numpy array with input token indices.
                shape: (`max_sequence_length`), dtype: int32.
            * 'attention_mask': Numpy array with attention mask.
                shape: (`max_sequence_length`), dtype: int32.
            * 'loss_mask': Numpy array equal to attention mask if 
                `include_padding_in_loss` is False, else all ones.
                shape: (`max_sequence_length`), dtype: int32.
            * 'token_type_ids': Numpy array with segment ids.
                shape: (`max_sequence_length`), dtype: int32.
            * 'labels': Numpy array with labels.
                shape: (`max_sequence_length`), dtype: int32.
    """
    input_ids = np.ones((max_sequence_length,), dtype=np.int32) * input_pad_id
    attention_mask = (
        np.ones((max_sequence_length,), dtype=np.int32) * attn_mask_pad_id
    )
    loss_mask = np.ones((max_sequence_length,), dtype=np.int32)

    # Convert tokens to integer ids.
    token_ids = tokenize(tokens_list)
    input_ids[0 : len(token_ids)] = token_ids
    attention_mask[0 : len(token_ids)] = 1

    label_ids = np.ones((max_sequence_length,), dtype=np.int32) * labels_pad_id
    labels = [label_map[label] for label in labels_list]
    label_ids[0 : len(token_ids)] = labels

    # only one segment, so segment ids is all zeros
    segment_ids = np.zeros((max_sequence_length,), dtype=np.int32)

    # loss mask
    if not include_padding_in_loss:
        loss_mask = attention_mask.copy()
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "token_type_ids": segment_ids,
        "labels": label_ids,
    }
    return features
