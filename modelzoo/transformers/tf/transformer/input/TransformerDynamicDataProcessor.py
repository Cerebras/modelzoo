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

import tensorflow as tf

from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset
from modelzoo.transformers.tf.transformer.input.data_processing.Tokenization import (
    BaseTokenizer,
)
from modelzoo.transformers.tf.transformer.input.data_processing.utils import (
    check_special_tokens_index,
    create_batched_dataset,
    create_filter_fn,
    create_masks_map_fn,
    get_special_tokens,
    get_special_tokens_index,
    scale_loss,
)


class TransformerDynamicDataProcessor:
    def __init__(self, params):
        """
        Dataset generator for Transformer model.
        :param dict params: List of training input parameters for creating dataset.
        
        :param int batch_size: Number of sequences to process at once.
        :param str src_data_dir: Path to input sequence (source) files 
        containing tokenized data.
        :param str tgt_data_dir: Path to output sequence (target) files 
        containing tokenized data.
        :param str src_vocab_file: Path to input sequence (source) vocabulary file.
        :param str tgt_vocab_file: Path to output sequence (target) vocabulary file.
        :param int src_max_sequence_length: Maximum sequence length allowed
        for inputs (source sequence). Sequences longer than this will be truncated 
        while shorter ones will be padded to max.
        :param int tgt_max_sequence_length: Maximum sequence length allowed
        for output (target sequnce). Sequences longer than this will be truncated 
        while shorter ones will be padded to max.
        :param bool shuffle, optional: Change the order of data each epoch.
        :param int shuffle_buffer, optional: Size of buffer for shuffling. 
        :param bool shuffle_seed, optional: If true allows for reproducibility
        while retaining shuffling.
        :param bool repeat, optional: Allows TF dataset to continue sending
        data for arbitrary epochs.
        :param bool source_reverse, optional: Whether to reverse the target sequence 
        as source and vice-versa.
        :param bool use_multiple_workers, optional: Specifies whether to use 
        multiple workers with the Cerebras System.
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

        self._source_reverse = params.get("source_reverse", False)
        self.use_multiple_workers = params.get("use_multiple_workers", False)

        # data directory/files.
        self.src_data_dir = params["src_data_dir"]
        self.tgt_data_dir = params["tgt_data_dir"]

        # vocab files.
        self.src_vocab_file = params["src_vocab_file"]
        self.tgt_vocab_file = params["tgt_vocab_file"]

        # Get max_sequence_length.
        self.src_max_sequence_length = params["src_max_sequence_length"]
        self.tgt_max_sequence_length = params["tgt_max_sequence_length"]

        # buckets can be passed in as an int or a list
        self.buckets = params.get("buckets", None)
        if isinstance(self.buckets, int):
            max_len = max(
                self.src_max_sequence_length, self.tgt_max_sequence_length
            )
            self.buckets = [
                int((i + 1) * max_len / self.buckets)
                for i in range(self.buckets - 1)
            ]

        # Get index of specical tokens.
        self.special_tokens = get_special_tokens()
        self.src_tokenizer = BaseTokenizer(
            self.src_vocab_file, unk_token=self.special_tokens["UNK"]
        )
        self.tgt_tokenizer = BaseTokenizer(
            self.tgt_vocab_file, unk_token=self.special_tokens["UNK"]
        )
        self.special_tokens_index = get_special_tokens_index(
            self.special_tokens, self.src_tokenizer, self.tgt_tokenizer,
        )
        # VSL specific.
        self.use_vsl = params["use_vsl"]
        self.input_pad_id = (
            params["input_pad_id"]
            if self.use_vsl
            else self.special_tokens_index["PAD"]
        )

        self.compute_dtype = (
            tf.float16 if params["mixed_precision"] else tf.float32
        )

    def create_tf_dataset(
        self, mode=tf.estimator.ModeKeys.TRAIN, input_context=None
    ):
        """
        Create tf dataset.

        :param bool is_training: Specifies whether the data is for training
        :param dict input_context: Given by distributed strategy for training
        :returns: tf dataset
            dataset (tf.Dataset): contain 'feature, label' with the following
                keys and tensors.
                feature:
                    "encoder_input_ids": [tf.int32, ...]
                    "encoder_input_length": tf.int32
                    "decoder_input_ids": [tf.int32, ...]
                    "decoder_input_length": tf.int32
                    "encoder_mask": [tf.int32, ...]
                    "decoder_mask": [tf.int32, ...]
                    "loss_scale": [tf.float16/float32, ...]
                label: [tf.int32, ...]  # dec_tgt
        """
        check_special_tokens_index(
            self.special_tokens, self.src_tokenizer, self.tgt_tokenizer
        )

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        # Read from BPE tokenized text files for source and target translations.
        src_dataset = tf.data.TextLineDataset(
            sorted(tf.io.gfile.glob(self.src_data_dir))
        )
        tgt_dataset = tf.data.TextLineDataset(
            sorted(tf.io.gfile.glob(self.tgt_data_dir))
        )

        # Group dataset.
        src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        src_tgt_dataset = shard_dataset(
            src_tgt_dataset, self.use_multiple_workers, input_context
        )

        # Add feature and label mapping.
        src_tgt_dataset = src_tgt_dataset.map(
            self.map_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=not (self.shuffle and self.shuffle_seed is None),
        )

        if is_training and self.shuffle:
            src_tgt_dataset = src_tgt_dataset.shuffle(
                buffer_size=self.shuffle_buffer, seed=self.shuffle_seed,
            )

        # Filter by length.
        # Removes all examples where the length of either.
        # src/tgt sentence is = 0 or greater than src/tgt_max_sequence_length.
        filter_fn = create_filter_fn(
            self.src_max_sequence_length, self.tgt_max_sequence_length
        )
        src_tgt_dataset = src_tgt_dataset.filter(filter_fn)

        # Creates a padded batch.
        # If `num_buckets` > 1, bins/buckets examples according to
        # the length and batches them.

        # VSL specific changes:
        # Set the pad_id in input_ids passed to embedding layer as:
        # a. If `use_vsl` is `False`, use the pad token id from vocab
        #   and pass -1 to `EmbeddingLayer`.
        # b. If `use_vsl` is `True`, user specified `input_pad_id` is used for
        #   pad tokens for `encoder_input_ids` and `decoder_input_ids` and
        #   pass user specified `input_pad_id` to `EmbeddingLayer`.
        # Note: `input_pad_id` defaults to `0` if not specified by user
        # and `use_vsl` is `True`. Defaults to `-1` if `use_vsl` is `False`.
        src_tgt_dataset = create_batched_dataset(
            src_tgt_dataset,
            self.batch_size,
            buckets=self.buckets,
            src_max_sequence_length=self.src_max_sequence_length,
            tgt_max_sequence_length=self.tgt_max_sequence_length,
            pad_id=self.input_pad_id,
        )

        # Create masks with `1` in padded locations and `0` in non-padded locations.
        # For labels, i.e decoder target outputs,
        # set pad token to `-1` if `use_vsl` is `True`.
        mask_map_fn = create_masks_map_fn(
            pad_id=self.input_pad_id, pad_id_label=-1 if self.use_vsl else None,
        )
        src_tgt_dataset = src_tgt_dataset.map(
            mask_map_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=not (self.shuffle and self.shuffle_seed is None),
        )

        src_tgt_dataset = src_tgt_dataset.map(
            lambda x, y: scale_loss(
                x,
                y,
                self.batch_size,
                self.tgt_max_sequence_length,
                self.compute_dtype,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=not (self.shuffle and self.shuffle_seed is None),
        )

        if is_training and self.shuffle and self.buckets:
            src_tgt_dataset = src_tgt_dataset.shuffle(
                buffer_size=int(self.shuffle_buffer / self.batch_size),
                seed=self.shuffle_seed,
            )

        if is_training and self.repeat:
            src_tgt_dataset = src_tgt_dataset.repeat()

        src_tgt_dataset = src_tgt_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        return src_tgt_dataset

    def _convert_to_ids(self, text, tokenizer):
        """
        Helper function to convert a string of text to integer ids
        :param tf.string text sentence to split
        :param BaseTokenizer tokenizer instance of BaseTokenizer
            to convert strings to ids based on vocab file
        returns tensor containing tf.int32 ids
        """
        text = tf.compat.v1.string_split([text]).values
        return tokenizer.convert_tokens_tensor_to_ids(text)

    def map_fn(self, *raw_record):
        """
        Map function for tf.data.Dataset
        """
        src, tgt = raw_record
        src_ids = self._convert_to_ids(src, self.src_tokenizer)
        tgt_ids = self._convert_to_ids(tgt, self.tgt_tokenizer)

        tgt_in = tf.concat(
            (
                tf.constant([self.special_tokens_index["SOS"]], dtype=tf.int32),
                tgt_ids,
            ),
            0,
        )
        tgt_out = tf.concat(
            (
                tgt_ids,
                tf.constant([self.special_tokens_index["EOS"]], dtype=tf.int32),
            ),
            0,
        )

        feature = {}
        feature["encoder_input_ids"] = src_ids
        feature["encoder_input_length"] = tf.size(src_ids, out_type=tf.int32)
        feature["decoder_input_ids"] = tgt_in
        feature["decoder_input_length"] = tf.size(tgt_in, out_type=tf.int32)

        return feature, tgt_out
