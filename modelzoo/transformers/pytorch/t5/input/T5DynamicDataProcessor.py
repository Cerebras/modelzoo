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
Processor for PyTorch T5 training.
"""
import csv
import os
import random
import sys
from functools import partial

import numpy as np
import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.input_utils import bucketed_batch
from modelzoo.transformers.pytorch.bert.input.utils import build_vocab
from modelzoo.transformers.pytorch.input_utils import (
    get_data_for_task,
    num_tasks,
    shard_list_interleaved,
    task_id,
)
from modelzoo.transformers.pytorch.t5.input.utils import (
    concatenate_documents,
    construct_denoising_objective,
    flat_map,
    get_raw_sequence_lengths,
    pad_t5_input_features,
    parse_text,
    select_random_chunk,
    shuffle,
    split_sequences,
)


class T5DynamicDataProcessor(torch.utils.data.IterableDataset):
    """
    Reads text files containing the input text tokens, adds extra ids for
    language modeling task on the fly.
    :param str src_vocab_file: Path to file containing tokens of vocabulary, 
    one token per line.
    :param str src_data_dir: Path to directory containing the output of 
    preprocess.sh, with all the files of tokenized data.
    :param int batch_size: Number of sequences per batch. Note that it is 
    different between systems. 
    :param bool shuffle, optional: If true the data will be shuffled before passing 
    into the model. Recommended for training. Can be set to False for 
    debugging.
    :param int shuffle_seed, optional: Sets random seed for the order of 
    data shuffling. Allows for reproducibility while still shuffling data. 
    :param int shuffle_buffer: Size of buffer used to store data before 
    shuffling 
    :param int extra_ids, optional: Number of sentinel tokens for T5 objective 
    :param int src_max_sequence_length, optional: Largest possible sequence 
    length for the input. If longer it will be truncated. All other sequences 
    padded to this length. 
    :param int tgt_max_sequence_length, optional: Largest possible sequence 
    length for the labels. If longer it will be truncated. All other sequences
    padded to this length.
    :param int num_workers, optional: Number of processes that move data to the
    accelerator system, so that the system doesn't process data faster than 
    it receives it.
    :param bool drop_last, optional: If the last batch is not the full size,
    i.e. the dataset could not divide evenly into the batch-size, do not
    use the last batch.
    :param int prefetch_factor, optional: Number of batch loaded in advance 
    by each worker. 
    :param bool persistent_workers, optional: If set, workers will not be 
    shutdown after going through the dataset once. 
    :param bool do_lower, optional: If set, will lowercase all tokens in vocabulary. 
    T5's vocabulary is cased so this is not recommended.
    :param list buckets, optional: A list of boundaries for sequence lengths 
    to bucket together in order to speed up VTS/VSL.
    :param bool dynamic_loss_weight, optional: If set, will divide the loss 
    for a token by the length of the sequence that the token comes from. 
    :param bool pack_sequences, optional: If set, will concatenate sequences   
    so that computation is performed on real data rather than padding
    :param int num_documents_to_concatenate, optional: Specifies how many
    documents to pack together
    :param str oov_token, optional: Token for out-of-vocabulary words/sub-words
    :param str sos_token, optional: Token for start-of-sequence
    :param str eos_token, optional: Token for end-of-sequence
    :param str pad_token, optional: Token for padding  
    :param int labels_pad_id, optional: Can set specific padding for labels
    :param int input_pad_id, optional: Can set specific padding for inputs
    :param bool mixed_precision, optional: If set, will use float16 rather 
    than float32 when possible
    """

    def __init__(self, params):
        super(T5DynamicDataProcessor, self).__init__()

        self.use_cs = cm.use_cs()
        # Input params.
        self.meta_data = self.get_meta_data(params["src_data_dir"])

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0].
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.batch_size = params["batch_size"]

        self.num_batches = self.num_examples // self.batch_size

        assert (
            self.num_batches > 0
        ), "Dataset does not contain enough samples for one batch. Please choose a smaller batch size"

        self.num_tasks = num_tasks()
        self.task_id = task_id()
        self.num_batch_per_task = self.num_batches // self.num_tasks

        assert (
            self.num_batch_per_task > 0
        ), "Dataset cannot be evenly distributed across the given tasks. Please choose fewer tasks to run with."

        self.num_examples_per_task = self.num_batch_per_task * self.batch_size
        self.files_in_task = get_data_for_task(
            self.task_id,
            self.meta_data_values_cum_sum,
            self.num_examples_per_task,
            self.meta_data_values,
            self.meta_data_filenames,
        )
        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        np.random.seed(self.shuffle_seed)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)
        self.do_lower = params.get("do_lower", False)

        # buckets must be a list of boundaries or None to disable. Supplying an
        # integer for uniform bucketing is not currently supported.
        self.buckets = params.get("buckets", None)
        assert self.buckets is None or (
            self.buckets and isinstance(self.buckets, list)
        ), f"buckets may be None or a non-empty list of boundaries. Got {self.buckets}"
        # Compute a loss weight for every batch to use in the averaging of
        # per-token loss across the batch.
        self.dynamic_loss_weight = params.get("dynamic_loss_weight")
        self.pack_sequences = params.get("pack_sequences", False)
        self.num_documents_to_concatenate = params.get(
            "num_documents_to_concatenate", 128
        )

        # Multi-processing params.
        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

        # Get special tokens and tokens that should not be masked.
        self.special_tokens = {
            "oov_token": params.get("oov_token", "<unk>"),
            "sos_token": params.get("sos_token", "<s>"),
            "eos_token": params.get("eos_token", "</s>"),
            "pad_token": params.get("pad_token", "<pad>"),
        }
        if self.do_lower:
            self.special_tokens = {
                key: value.lower() for key, value in self.special_tokens.items()
            }

        # Get vocab file and size.
        self.src_vocab_file = params["src_vocab_file"]
        self.src_vocab_file, self.src_vocab_size = build_vocab(
            self.src_vocab_file, self.do_lower, self.special_tokens["oov_token"]
        )
        # Updating input and model params to account extra ids
        # for T5 Language Modeling task.
        extra_ids = params.get("extra_ids", 0)
        self.src_vocab_size += extra_ids

        # Init tokenizer.
        self.src_tokenize = self.src_vocab_file.forward

        # Getting indices for special tokens.
        self.special_tokens_indices = {
            key: self.src_tokenize([value])[0]
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
        self.attn_mask_pad_id = 0
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
        # Max sequence lengths size params.
        self.src_max_sequence_length = params["src_max_sequence_length"]
        self.tgt_max_sequence_length = params["tgt_max_sequence_length"]

        # Store params.
        self.data_buffer = []
        self.text_files_per_task_per_worker = []
        self.processed_buffers = 0

    def get_meta_data(self, data_dir):
        """
        Read data from meta files.
        :param str data_dir: Path to the input directory.
        :return: Processed meta data.
        """
        if not isinstance(data_dir, list):
            data_dir = [data_dir]

        meta_data = {}
        for file_name in data_dir:
            meta_file = os.path.join(file_name, "meta.dat")
            assert os.path.exists(
                meta_file
            ), f"Meta file is missing in the input directory: {data_dir}."

            with open(meta_file, "r") as fin:
                for line in fin.readlines():
                    line = line.strip().split()
                    # Format: src_file_path src_file_size
                    meta_data[os.path.join(file_name, line[0])] = int(line[1])
        return meta_data

    def load_buffer(self):
        """
        Generator to read samples of data.

        :returns: Yields data samples, one at a time.

        """
        self.processed_buffers = 0
        # The csv file might contain very huge fields, therefore increase the `field_size_limit`.
        csv.field_size_limit(sys.maxsize)
        while self.processed_buffers < len(self.text_files_per_task_per_worker):
            (
                current_file_path,
                num_examples,
                start_id,
            ) = self.text_files_per_task_per_worker[self.processed_buffers]

            with open(current_file_path, "r", newline="") as fin:
                # Some fields contain zero bytes.
                data_reader = csv.reader(
                    (x.replace("\0", "") for x in fin), delimiter="\n"
                )

                for row_id, row in enumerate(data_reader):
                    if row_id < start_id:
                        continue
                    if row_id >= start_id + num_examples:
                        break
                    yield row

                self.processed_buffers += 1

    def get_single_item(self):
        """
        Iterating over the data to construct input features.

        :return: A dict with training features:
            * np.array[int.32] input_ids: Numpy array with encoder input token indices.
                Shape: (`src_max_sequence_length`).
            * np.array[int.32] decoder_input_ids: Numpy array with decoder input token indices.
                Shape: (`tgt_max_sequence_length`).
            * np.array[int.32] attention_mask: Numpy array with attention mask for encoder.
               Shape: (`src_max_sequence_length`).
            * np.array[int.32] decoder_attention_mask: Numpy array with attention mask for decoder.
               Shape: (`tgt_max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels for teacher forcing mode.
               Shape: (`tgt_max_sequence_length`).
        """
        # Shard the data across multiple processes.

        max_raw_sequence_len, max_target_len = get_raw_sequence_lengths(
            self.src_max_sequence_length
        )
        if max_target_len > self.tgt_max_sequence_length:
            raise ValueError(
                f"Actual target sequence length must be less than max target "
                f"sequence length. Got {max_target_len} > "
                f"{self.tgt_max_sequence_length}. Please increase the max "
                f"target sequence length."
            )

        dataset = self.load_buffer()

        # parse, tokenize, and select chunks from the input documents
        dataset = map(
            lambda x: parse_text("".join(x), do_lower=self.do_lower), dataset
        )
        dataset = map(self.src_tokenize, dataset)
        dataset = map(select_random_chunk, dataset)

        # pack sequences to reduce padding
        if self.pack_sequences:
            if self.shuffle:
                # shuffle before concatenation so that you get variety in which
                # documents get concatenated
                dataset = shuffle(
                    dataset, self.shuffle_buffer // self.batch_size, self.rng
                )
            dataset = concatenate_documents(
                dataset,
                num_to_concatenate=self.num_documents_to_concatenate,
                pad_id=self.input_pad_id,
            )

        # split documents into sequences and format for input to T5 model
        dataset = flat_map(
            partial(split_sequences, length=max_raw_sequence_len), dataset
        )
        dataset = filter(len, dataset)
        if self.shuffle:
            # shuffle after `split_sequences` so that sequences from the same
            # document aren't always consecutive
            dataset = shuffle(dataset, self.shuffle_buffer, self.rng)
        dataset = map(
            partial(
                construct_denoising_objective,
                vocab_size=self.src_vocab_size,
                sos_token=self.special_tokens_indices["sos_token"],
                eos_token=self.special_tokens_indices["eos_token"],
            ),
            dataset,
        )
        dataset = map(
            lambda features: pad_t5_input_features(
                src_max_sequence_length=self.src_max_sequence_length,
                tgt_max_sequence_length=self.tgt_max_sequence_length,
                input_pad_id=self.input_pad_id,
                attn_mask_pad_id=self.attn_mask_pad_id,
                labels_pad_id=self.labels_pad_id,
                features=features,
            ),
            dataset,
        )

        return dataset

    def element_length_fn(self, features):
        """
        Takes a single sample and returns the sequence length of that sample
        to be used for VTS bucketing.
        """
        return np.sum(features["attention_mask"])

    def __iter__(self):
        batched_dataset = bucketed_batch(
            self.get_single_item(),
            self.batch_size,
            buckets=self.buckets,
            element_length_fn=self.element_length_fn,
            drop_last=self.drop_last,
            seed=self.shuffle_seed,
        )
        for batch in batched_dataset:
            if self.dynamic_loss_weight:
                scale = self.batch_size / torch.sum(
                    batch["decoder_attention_mask"]
                )
                batch["loss_weight"] = scale.expand(self.batch_size, 1)
            yield batch

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single-process
            worker_id = 0
            num_workers = 1

        self.processed_buffers = 0
        if self.shuffle_seed is not None:
            self.shuffle_seed += worker_id + 1
        self.rng = random.Random(self.shuffle_seed)

        # Shard the data across multiple processes.
        self.text_files_per_task_per_worker = shard_list_interleaved(
            self.files_in_task, worker_id, num_workers
        )
        if self.shuffle:
            self.rng.shuffle(self.text_files_per_task_per_worker)

    def create_dataloader(self, is_training=True):
        """
        Classmethod to create the dataloader object.
        """
        if not self.num_workers:
            self.prefetch_factor = 2
            self.persistent_workers = False
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=None,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            worker_init_fn=self._worker_init_fn,
        )

        if self.num_workers == 0:
            self._worker_init_fn(0)

        return dataloader
