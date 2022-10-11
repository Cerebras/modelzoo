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

import os
from functools import partial

import tensorflow as tf

from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset
from modelzoo.transformers.tf.t5.input.data_processing.Tokenization import (
    T5BaseTokenizer,
)
from modelzoo.transformers.tf.t5.input.data_processing.utils import (
    concatenate_documents,
    construct_denoising_objective,
    convert_text_to_token_ids,
    create_batched_dataset,
    create_masks_map_fn,
    get_raw_sequence_lengths,
    get_special_tokens,
    scale_loss,
    select_random_chunk,
    split_sequences,
)


class T5DynamicDataProcessor:
    def __init__(self, params):
        """
        Dataset generator for T5 model.
        :param str vocab_file: Path to vocabulary file.
        :param str data_dir: Path to files containing tokenized data.
        :param int batch_size: Number of sequences to process at once.
        :param int src_max_sequence_length: Largest possible sequence length
        for inputs. If larger, sequence will be truncated. Other sequences
        padded to max.
        :param int tgt_max_sequence_length: Largest possible sequence length
        for labels. If larger, sequence will be truncated. Other sequences
        padded to max.
        :param bool shuffle, optional: Change the order of data each epoch.
        :param int shuffle_buffer, optional: Size of buffer for shuffling. 
        :param bool shuffle_seed, optional: If true allows for reproducibility
        while retaining shuffling.
        :param bool repeat, optional: Allows TF dataset to continue sending
        data for arbitrary epochs.
        :param bool use_multiple_workers, optional: Specifies whether to use 
        multiple workers with the Cerebras System.
        :param int n_parallel_reads, optional: The number of elements 
        processed concurrently for dataset.
        :param bool pack_sequences, optional: Combine sentences that are 
        shorter than max_sequence_length to reduce computation on padding. 
        :param int num_documents_to_concatenate, optional: Number of docs to 
        pack together.
        :param int vocab_size, optional: Number of tokens in vocabulary.
        :param bool do_lower_case, optional: Set all tokens to lowercase.
        :param int or list buckets, optional: Used for bucketing similar-sized 
        sequences to pack together. If int, will split max sequence length 
        to evenly spaced intervals based on the number of buckets specified.
        If list, the list elements are the boundaries of the sequence lengths.
        :param bool use_vsl, optional: Use variable-sequence-length.
        :param int input_pad_id, optional: Can specify the index of token to 
        use for padding.
        :params bool mixed_precision, optional: If set, will do calculations 
        in float16 rather than float32 when possible.
        
        """
        self.batch_size = params["batch_size"]
        self.shuffle = params.get("shuffle", True)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.repeat = params.get("repeat", True)
        self.use_multiple_workers = params.get("use_multiple_workers", False)
        self.n_parallel_reads = params.get("n_parallel_reads", 4)
        self.pack_sequences = params.get("pack_sequences", False)
        self.num_documents_to_concatenate = params.get(
            "num_documents_to_concatenate", 128
        )

        # Data directory and files.
        self.data_dir = params["data_dir"]

        # Special tokens for padding, unknown tokens etc.
        self.special_tokens = get_special_tokens()

        # Get vocab size and file.
        self.vocab_size = params["vocab_size"]
        self.vocab_file = params["vocab_file"]

        # Init tokenizer.
        self.tokenizer = T5BaseTokenizer(
            self.vocab_file,
            unk_token=self.special_tokens["UNK"],
            do_lower_case=params["do_lower_case"],
        )

        # Get max sequence lengths.
        self.src_max_sequence_length = params["src_max_sequence_length"]
        self.tgt_max_sequence_length = params["tgt_max_sequence_length"]

        # buckets can be passed in as an int or list
        self.buckets = params.get("buckets", 1)
        if isinstance(self.buckets, int):
            self.buckets = [
                int((i + 1) * self.src_max_sequence_length / self.buckets)
                for i in range(self.buckets - 1)
            ]

        # VSL specific.
        self.use_vsl = params["use_vsl"]
        self.input_pad_id = (
            params["input_pad_id"]
            if self.use_vsl
            else self.tokenizer.get_id(self.special_tokens["PAD"])
        )

        self.compute_dtype = (
            tf.float16 if params["mixed_precision"] else tf.float32
        )

    def create_tf_dataset(
        self, mode=tf.estimator.ModeKeys.TRAIN, input_context=None
    ):
        """
        Create tf dataset.

        :param mode: tf.estimator.ModeKeys.TRAIN (default) or
            tf.estimator.ModeKeys.EVAL or tf.estimator.ModeKeys.PREDICT.
        :param dict input_context: Given by distributed strategy for training.
        :return: tf dataset.
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        map_args = {
            "num_parallel_calls": tf.data.experimental.AUTOTUNE,
            "deterministic": not (self.shuffle and self.shuffle_seed is None),
        }
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

        dataset = tf.data.Dataset.list_files(
            os.path.join(self.data_dir, "*.txt"),
            shuffle=self.shuffle,
            seed=self.shuffle_seed,
        )
        dataset = shard_dataset(
            dataset, self.use_multiple_workers, input_context
        )
        # Read dataset of sentencepiece string tokens.
        dataset = dataset.interleave(
            tf.data.TextLineDataset,
            cycle_length=self.n_parallel_reads,
            **map_args,
        )

        # parse, tokenize, and select chunks from the input documents
        dataset = dataset.map(
            lambda *x: convert_text_to_token_ids(x[0], self.tokenizer),
            **map_args,
        )
        dataset = dataset.map(
            partial(select_random_chunk, seed=self.shuffle_seed), **map_args
        )

        # pack sequences to reduce padding
        if self.pack_sequences:
            if is_training and self.shuffle:
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer // self.batch_size,
                    seed=self.shuffle_seed,
                )
            dataset = concatenate_documents(
                dataset,
                num_to_concatenate=self.num_documents_to_concatenate,
                pad_id=self.tokenizer.get_id(self.special_tokens["PAD"]),
                **map_args,
            )

        # split documents into sequences and format for input to T5 model
        dataset = split_sequences(
            dataset, length=max_raw_sequence_len, **map_args
        )
        dataset = dataset.filter(lambda x: tf.greater(tf.size(x), 0))
        dataset = dataset.map(
            partial(
                construct_denoising_objective,
                vocab_size=self.vocab_size,
                sos_token=self.tokenizer.get_id(self.special_tokens["SOS"]),
                eos_token=self.tokenizer.get_id(self.special_tokens["EOS"]),
            ),
            **map_args,
        )

        if is_training and self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer, seed=self.shuffle_seed
            )

        # Creates a padded batched dataset.
        dataset = create_batched_dataset(
            dataset=dataset,
            batch_size=self.batch_size,
            buckets=self.buckets,
            src_max_sequence_length=self.src_max_sequence_length,
            tgt_max_sequence_length=self.tgt_max_sequence_length,
            pad_id=self.input_pad_id,
        )

        # Creates masks with `1` in padded locations and `0` in non-padded.
        mask_map_fn = create_masks_map_fn(
            pad_id=self.input_pad_id, pad_id_label=-1 if self.use_vsl else None,
        )
        dataset = dataset.map(mask_map_fn, **map_args)

        dataset = dataset.map(
            lambda x, y: scale_loss(
                x,
                y,
                self.batch_size,
                self.tgt_max_sequence_length,
                self.compute_dtype,
            ),
            **map_args,
        )

        if is_training and self.repeat:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
