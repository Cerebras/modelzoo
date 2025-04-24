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
from typing import Any, List, Literal, Optional

import numpy as np
import torch
from pydantic import Field, model_validator

from cerebras.modelzoo.common.input_utils import (
    bucketed_batch,
    get_streaming_batch_size,
)
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import AliasedPath, ValidatedPath
from cerebras.modelzoo.data.common.input_utils import (
    get_data_for_task,
    num_tasks,
    shard_list_interleaved,
    task_id,
)
from cerebras.modelzoo.data.nlp.bert.bert_utils import build_vocab
from cerebras.modelzoo.data.nlp.t5.t5_utils import (
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


class T5DynamicDataProcessorConfig(DataConfig):
    data_processor: Literal["T5DynamicDataProcessor"]

    src_vocab_file: AliasedPath = ...
    "Path to file containing tokens of vocabulary, one token per line."

    src_data_dir: ValidatedPath = ...
    """
    Path to directory containing the output of preprocess.sh, with all the files
    of tokenized data.
    """

    batch_size: int = ...
    "Number of sequences per batch. Note that it is different between systems."

    shuffle: bool = True
    """
    If true the data will be shuffled before passing into the model. Recommended
    for training. Can be set to False for debugging.
    """

    shuffle_seed: Optional[int] = None
    """
    Sets random seed for the order of data shuffling. Allows for reproducibility
    while still shuffling data.
    """

    shuffle_buffer: Optional[int] = None
    "Size of buffer used to store data before shuffling"

    extra_ids: int = 0
    "Number of sentinel tokens for T5 objective"

    src_max_sequence_length: int = ...
    """
    Largest possible sequence length for the input. If longer it will be
    truncated. All other sequences padded to this length.
    """

    tgt_max_sequence_length: int = ...
    """
    Largest possible sequence length for the labels. If longer it will be
    truncated. All other sequences padded to this length.
    """

    num_workers: int = 0
    """
    Number of processes that move data to the accelerator system, so that the
    system doesn't process data faster than it receives it.
    """

    drop_last: bool = True
    """
    If the last batch is not the full size, i.e. the dataset could not divide
    evenly into the batch-size, do not use the last batch.
    """

    prefetch_factor: Optional[int] = 10
    "Number of batch loaded in advance by each worker."

    persistent_workers: bool = True
    "If set, workers will not be shutdown after going through the dataset once."

    do_lower: bool = False
    """
    If True, will lowercase all tokens in vocabulary. T5's vocabulary is cased
    so this is not recommended.
    """

    buckets: Optional[List[int]] = None
    """
    A list of boundaries for sequence lengths to bucket together in order to
    speed up VTS/VSL.
    """

    dynamic_loss_weight: bool = False
    """
    If set, will divide the loss for a token by the length of the sequence that
    the token comes from.
    """

    pack_sequences: bool = False
    """
    If set, will concatenate sequences so that computation is performed on real
    data rather than padding
    """

    num_documents_to_concatenate: int = 128
    "Specifies how many documents to pack together"

    oov_token: str = "<unk>"
    "Token for out-of-vocabulary words/sub-words"

    sos_token: str = "<s>"
    "Token for start-of-sequence"

    eos_token: str = "</s>"
    "Token for end-of-sequence"

    pad_token: str = "<pad>"
    "Token for padding"

    labels_pad_id: Optional[str] = None
    "Can set specific padding for labels"

    input_pad_id: Optional[str] = None
    "Can set specific padding for inputs"

    vocab_size: Optional[Any] = Field(None, deprecated=True)
    tgt_vocab_file: Optional[Any] = Field(None, deprecated=True)
    tgt_data_dir: Optional[Any] = Field(None, deprecated=True)

    def post_init(self, context):
        if self.shuffle_buffer is None:
            self.shuffle_buffer = 10 * self.batch_size

        if self.do_lower:
            self.do_lower = False
            self.oov_token = self.oov_token.lower()
            self.sos_token = self.sos_token.lower()
            self.eos_token = self.eos_token.lower()
            self.pad_token = self.pad_token.lower()

    @model_validator(mode="after")
    def validate_max_sequence_length(self, info):
        if info.context:
            model_config = info.context.get("model", {}).get("config")
            if model_config:
                if (
                    model_config.src_max_position_embeddings
                    != self.src_max_sequence_length
                ):
                    raise ValueError(
                        f"src_max_sequence_length in data config ({self.src_max_sequence_length}) "
                        f"does not match src_max_position_embeddings in model config "
                        f"({model_config.src_max_position_embeddings})"
                    )

                if (
                    model_config.tgt_max_position_embeddings
                    != self.tgt_max_sequence_length
                ):
                    raise ValueError(
                        f"tgt_max_sequence_length in data config ({self.tgt_max_sequence_length}) "
                        f"does not match tgt_max_position_embeddings in model config "
                        f"({model_config.tgt_max_position_embeddings})"
                    )

        return self


class T5DynamicDataProcessor(torch.utils.data.IterableDataset):
    """
    Reads text files containing the input text tokens, adds extra ids for
    language modeling task on the fly.

    Args:
        config: Configuration for the data processor
    """

    def __init__(self, config: T5DynamicDataProcessorConfig):
        if isinstance(config, dict):
            config = T5DynamicDataProcessorConfig(**config)

        super().__init__()

        # Input params.
        self.meta_data = self.get_meta_data(config.src_data_dir)

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0].
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.batch_size = get_streaming_batch_size(config.batch_size)

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
        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        self.np_rng = np.random.default_rng(self.shuffle_seed)
        self.shuffle_buffer = config.shuffle_buffer
        self.do_lower = config.do_lower

        # buckets must be a list of boundaries or None to disable. Supplying an
        # integer for uniform bucketing is not currently supported.
        self.buckets = config.buckets

        # Compute a loss weight for every batch to use in the averaging of
        # per-token loss across the batch.
        self.dynamic_loss_weight = config.dynamic_loss_weight
        self.pack_sequences = config.pack_sequences
        self.num_documents_to_concatenate = config.num_documents_to_concatenate

        # Multi-processing params.
        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        # Get special tokens and tokens that should not be masked.
        self.special_tokens = {
            "oov_token": config.oov_token,
            "sos_token": config.sos_token,
            "eos_token": config.eos_token,
            "pad_token": config.pad_token,
        }

        # Get vocab file and size.
        self.src_vocab_file = config.src_vocab_file
        self.src_vocab_file, self.src_vocab_size = build_vocab(
            self.src_vocab_file, self.do_lower, self.special_tokens["oov_token"]
        )
        # Updating input and model params to account extra ids
        # for T5 Language Modeling task.
        extra_ids = config.extra_ids
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
        self.labels_pad_id = config.labels_pad_id
        self.input_pad_id = config.input_pad_id

        if self.labels_pad_id is None:
            self.labels_pad_id = self.special_tokens_indices["pad_token"]
        if self.input_pad_id is None:
            self.input_pad_id = self.special_tokens_indices["pad_token"]

        self.attn_mask_pad_id = 0

        # Max sequence lengths size params.
        self.src_max_sequence_length = config.src_max_sequence_length
        self.tgt_max_sequence_length = config.tgt_max_sequence_length

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
        dataset = map(partial(select_random_chunk, rng=self.np_rng), dataset)

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
                rng=self.np_rng,
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
        self.np_rng = np.random.default_rng(self.shuffle_seed)

        # Shard the data across multiple processes.
        self.text_files_per_task_per_worker = shard_list_interleaved(
            self.files_in_task, worker_id, num_workers
        )
        if self.shuffle:
            self.rng.shuffle(self.text_files_per_task_per_worker)

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        if not self.num_workers:
            self.prefetch_factor = None
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
