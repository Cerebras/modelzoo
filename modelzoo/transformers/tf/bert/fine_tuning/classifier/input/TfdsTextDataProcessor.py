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
Processors for GLUE tasks.
"""
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from modelzoo.transformers.data_processing.Tokenization import FullTokenizer


class TfdsTextDataProcessor(ABC):
    """
    Base class for processors that load their raw data from TFDS.
    Child classes must provide map_fn, name.

    :param dict params: List of training input parameters for creating dataset.
    Expects the following fields:
        - "vocab_file" (str): Path to the vocab file. 
        - "data_dir" (str): Path to directory containing the TF Records. 
        - "batch_size" (int): Batch size.
        - "max_sequence_length" (int): Maximum length of the sequence.
        - "repeat" (bool): Flag to enable data repeat.
        - "shuffle" (bool): Flag to enable data shuffling.
        - "shuffle_seed" (int): Shuffle seed.
        - "shuffle_buffer" (int): Shuffle buffer size.
        - "cache" (bool): Caches the elements in this dataset.
        - "do_lower" (bool): Flag to lower case the texts.
        - "add_special_tokens" (bool): Flag to add special tokens `[SEP]` and `[CLS]`.
        - "feature_rename_map" (dict): optional dictionary to rename the feature keys.
    :param String name: name of dataset to be passed into tfds.load, 
        for example, `glue/sst2` or `glue/mnli`.
    """

    def __init__(self, params, name):
        self.name = name
        self.vocab_file = params["vocab_file"]
        self.data_dir = params["data_dir"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]
        self.add_special_tokens = params.get("add_special_tokens", True)
        self.shuffle = params.get("shuffle", True)
        self.repeat = params.get("repeat", True)
        self.shuffle_buffer = params.get("shuffle_buffer", 1500)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.cache = params.get("cache", False)
        self.do_lower = params.get("do_lower", True)
        self.feature_rename_map = params.get("feature_rename_map", None)
        self.invert_mask = params.get("invert_mask", True)

        # only allow nondeterminism when shuffling unseeded
        self.deterministic = not (self.shuffle and self.shuffle_seed is None)

        assert self.batch_size > 0, "Batch size should be positive."
        assert (
            self.max_sequence_length > 0
        ), "Max sequence length should be positive."

        self.tokenizer = FullTokenizer(
            self.vocab_file, do_lower_case=self.do_lower
        )
        if self.add_special_tokens:
            self.cls_id, self.sep_id = self.tokenizer.convert_tokens_to_ids(
                ["[CLS]", "[SEP]"]
            )
        self.pad_id = (
            params["pad_id"]
            if "pad_id" in params
            else self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        )

    def select_partition(self, dataset, is_training):
        """
        Selects the correct partition of dataset loaded from TFDS.
        Override if dataset has unusual structure (e.g. MNLI)

        :param tf.data.Dataset dataset: dataset loaded from TFDS
        :param bool is_training: Specifies whether the data is for training
        :returns: unmodified tfds dataset
        """
        return dataset["train" if is_training else "validation"]

    def create_tf_dataset(self, is_training=True):
        """
        Create tf dataset.

        :param bool is_training: Specifies whether the data is for training
        :returns: tf dataset
        """
        as_dataset_kwargs = {
            "shuffle_files": is_training and not self.deterministic
        }

        dataset = tfds.load(
            self.name,
            data_dir=self.data_dir,
            as_dataset_kwargs=as_dataset_kwargs,
        )

        dataset = self.select_partition(dataset, is_training)

        if is_training and self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer, seed=self.shuffle_seed
            )

        dataset = dataset.map(
            self.map_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=self.deterministic,
        )

        if is_training and self.cache:
            dataset = dataset.cache()

        dataset = dataset.batch(
            batch_size=self.batch_size, drop_remainder=True,
        )

        if is_training and self.repeat:
            dataset = dataset.repeat()

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def _encode_sequence(self, text_a, text_b=None):
        """
        Tokenizes a single text (if text_b is None) or a pair of texts.
        Truncates and adds special tokens as needed.

        :param String text_a: first text to encode
        :param String text_b: second text to encode or None
        :returns list:
            - np array of ids
            - np array of segment_ids
        """
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b) if text_b else None
        # During fine-tuning, we truncate only at the end of each sequence,
        # rather than at both ends as we do during pre-training.
        max_num_tokens = self.max_sequence_length
        if self.add_special_tokens:
            max_num_tokens -= 3 if tokens_b else 2
        if tokens_b:
            total_length = len(tokens_a) + len(tokens_b)
            while total_length > max_num_tokens:
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
                total_length -= 1
        else:
            tokens_a = tokens_a[:max_num_tokens]

        ids = self.tokenizer.convert_tokens_to_ids(tokens_a)
        if self.add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]
        segment_ids = np.zeros(self.max_sequence_length, dtype=np.int32)

        if tokens_b:
            ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
            if self.add_special_tokens:
                ids_b = ids_b + [self.sep_id]
            segment_ids[len(ids) : len(ids) + len(ids_b)] = 1
            ids += ids_b

        ids = ids + [self.pad_id] * (self.max_sequence_length - len(ids))
        ids = np.array(ids, dtype=np.int32)
        return ids, segment_ids

    def _create_input_features(self, text_a, text_b):
        """
        Encodes one or two texts, creates input features dict.

        :param str text_a: first text to be encoded.
        :param str text_b: second text to be encoded or None.
        :returns dict: input features
        """
        inp = [text_a] if text_b is None else [text_a, text_b]
        input_ids, segment_ids = tf.numpy_function(
            self._encode_sequence, inp, [tf.int32, tf.int32]
        )
        input_ids.set_shape(self.max_sequence_length)
        segment_ids.set_shape(self.max_sequence_length)
        if self.invert_mask:
            input_mask = tf.math.equal(input_ids, self.pad_id)
        else:
            input_mask = tf.math.not_equal(input_ids, self.pad_id)
        input_mask = tf.cast(input_mask, tf.int32)
        input_features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
        }
        if self.feature_rename_map:
            for old_name, new_name in self.feature_rename_map.items():
                input_features[new_name] = input_features.pop(old_name)
        return input_features

    def map_fn_helper(
        self, features, text_key_a, text_key_b=None, dtype=tf.int32
    ):
        """
        Encodes one or two texts, packs into (features_dict, label)
        format expected by estimator.

        :param dict features: features dict from the tfds dataset.
            Must contain keys "label", text_key_a, text_key_b.
        :param str text_key_a: key (in features) of first text to be encoded.
        :param str text_key_b: key of second text to be encoded or None.
        :param str dtype: dtype for labels
        :returns:
            - input features dict
            - label tensor
        """
        labels = tf.cast(features["label"], dtype)
        input_features = self._create_input_features(
            features[text_key_a], features[text_key_b] if text_key_b else None,
        )
        return input_features, labels

    @abstractmethod
    def map_fn(self, features):
        """
        Preprocessing function passed into tf.data.Dataset.map
        Must return (features dict, label) as expected by estimator
        """
        raise NotImplementedError("map_fn must be implemented in child class!")


class Sst2DataProcessor(TfdsTextDataProcessor):
    """
    SST2 dataset processor for sentiment analysis.

    :param dict params: List of training input parameters for creating dataset
    """

    def __init__(self, params):
        super(Sst2DataProcessor, self).__init__(params, "glue/sst2")

    def map_fn(self, features):
        return self.map_fn_helper(features, "sentence")


class MnliDataProcessor(TfdsTextDataProcessor):
    """
    MNLI dataset processor for textual entailment prediction.

    :param dict params: List of training input parameters for creating dataset
    """

    def __init__(self, params):
        super(MnliDataProcessor, self).__init__(params, "glue/mnli")

    def select_partition(self, dataset, is_training):
        """
        MNLI has two validation sets:
            - the matched set comes from the same domains as the training set
            - the mismatched set comes from different domains

        Since we can only pass a single dataset to Estimator, during eval we
        concatenate the two validation sets. Before doing so, we give each
        example an "is_matched" label so that during eval we can measure
        matched and mismatched accuracies separately.
        """
        if is_training:
            return dataset["train"]

        matched = dataset["validation_matched"]
        mismatched = dataset["validation_mismatched"]

        def _add_matched_label(features, is_matched):
            """
            Adds feature "is_matched" = is_matched to one element
            """
            features["is_matched"] = is_matched
            return features

        def _add_matched_labels(ds, is_matched):
            """
            Adds feature "is_matched" = is_matched to all elements of ds
            """
            return ds.map(
                lambda features: _add_matched_label(features, is_matched),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=self.deterministic,
            )

        matched = _add_matched_labels(matched, 1)
        mismatched = _add_matched_labels(mismatched, 0)
        return matched.concatenate(mismatched)

    def map_fn(self, features):
        input_features, labels = self.map_fn_helper(
            features, "premise", "hypothesis"
        )
        if "is_matched" in features:
            input_features["is_matched"] = features["is_matched"]
        return input_features, labels
