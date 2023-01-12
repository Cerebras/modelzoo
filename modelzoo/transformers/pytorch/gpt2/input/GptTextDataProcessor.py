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

from modelzoo.common.pytorch.utils import BufferedShuffleDataset
from modelzoo.transformers.pytorch.gpt2.input.data_processor_utils import (
    training_data_generator,
)
from modelzoo.transformers.pytorch.input_utils import (
    num_tasks,
    shard_list_contiguous,
    task_id,
)


class GptTextDataProcessor(torch.utils.data.IterableDataset):
    """
    A text dataset processor for GPT pre-training.
    Performs on-the-fly processing of data from text.

    Functionality includes:
        Reading data from text documents
        Creating creating input sequences and masks, and
        autoregressive LM labels

    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:

    - "metadata_files" (str or list of str): A string or strings list each
      pointing to a metadata file. A metadata file contains file paths for
      flat text cleaned documents. It has one file path per line.
      The cleaned cleaned files have one paragraph per line and are
      separated by an empty line.
    - "vocab_file" (str): Vocabulary file, to build tokenization from
    - "encoder_file (str): Encoder file, map from word-pieces to
       token IDs for tokenization
    - "max_sequence_length (int): Maximum length of the sequence to generate
    - "short_sequence_prob (int): Probability of a short sequence. Defaults to 0.
    - "overlap_size (int): Size of overlap when forming sequences from
      buffered token ids in a sliding window fashion.
      Defaults to None, which sets the overlap of max_sequence_length/4.
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of samples loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    - "add_special_tokens" (bool): Flag to add BOS and EOS tokens.
    - "eos_token" (str): EOS token.
    - "pad_token" (str): PAD token.
    """

    def __init__(self, params):
        super(GptTextDataProcessor, self).__init__()

        self.metadata_files = params["metadata_files"]
        self.vocab_file = params["vocab_file"]
        self.encoder_file = params["encoder_file"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]

        self.short_sequence_prob = params.get("short_sequence_prob", 0)
        self.overlap_size = params.get("overlap_size", None)

        self.shuffle = params["shuffle"]
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)

        self.num_workers = params.get("num_workers", 8)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

        self.num_tasks = num_tasks()
        self.task_id = task_id()

        self.add_special_tokens = params.get("add_special_tokens", True)
        self.eos_token = params.get("eos_tokens", "<|endoftext|>")
        self.pad_token = params.get("pad_tokens", "<|endoftext|>")

        assert self.batch_size > 0, "Batch size should be positive."

        # get all text files by reading metadata files
        if isinstance(self.metadata_files, str):
            self.metadata_files = [self.metadata_files]

        input_files = []
        for _file in self.metadata_files:
            with open(_file, 'r') as _fin:
                input_files.extend(_fin.readlines())

        input_files_list = [x.strip() for x in input_files if x]

        random.seed(self.shuffle_seed)
        random.shuffle(input_files_list)

        self.input_files_in_this_task = shard_list_contiguous(
            input_files_list, self.task_id, self.num_tasks
        )

    def __iter__(self):
        """
        Iterating over the data to construct input features.
        """
        for example, _ in training_data_generator(
            self.input_files_in_this_worker,
            self.vocab_file,
            self.encoder_file,
            self.max_sequence_length,
            buffer_size=1e6,
            overlap_size=self.overlap_size,
            short_seq_prob=self.short_sequence_prob,
            inverted_mask=False,
            add_special_tokens=self.add_special_tokens,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            input_ids_dtype="int32",
            input_mask_dtype="int32",
            labels_dtype="int32",
        ):
            yield example

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single-process
            worker_id = 0
            num_workers = 1

        if self.shuffle_seed:
            # Use a unique seed for each worker.
            random.seed(self.shuffle_seed + worker_id)

        # Shard the data files between workers
        self.input_files_in_this_worker = shard_list_contiguous(
            input_files_in_this_task, worker_id, num_workers
        )

    def create_dataloader(self, is_training=True):
        """
        Classmethod to create the dataloader object.
        """
        data_loader = torch.utils.data.DataLoader(
            BufferedShuffleDataset(
                dataset=self, buffer_size=self.shuffle_buffer
            )
            if self.shuffle
            else self,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            worker_init_fn=self._worker_init_fn,
        )
        # set self.data_partitions in case self.num_workers == 0
        if self.num_workers == 0:
            self._worker_init_fn(0)
        return data_loader
