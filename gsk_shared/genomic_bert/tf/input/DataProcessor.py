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
Processors for BERT and Genome Pre-Training
"""

import collections
import json
import random
from functools import partial
from os import path

import numpy as np
import tensorflow as tf
from gsk_shared.genomic_bert.common.tokenization import GenomeTokenizer

_PAD_TOKEN = "[PAD]"
_CLS_TOKEN = "[CLS]"
_SEP_TOKEN = "[SEP]"
_MASK_TOKEN = "[MASK]"


class GenomeDataProcessor:
    """
    Genome dataset processor for BERT pre-training.
    Example:
        >>> from modelzoo.bert.tf.utils import get_params
        >>> from modelzoo.bert.tf.input.DataProcessor import GenomeDataProcessor
        >>> params = get_params("configs/params_bert_4f_mini.yaml")
        >>> dataset = GenomeDataProcessor(params).create_tf_dataset()
    """

    def __init__(self, params):
        metadata_files = params["metadata_files"]
        self._prepare_file_processor(metadata_files)
        self._prepare_tokenization(params)

        self.batch_size = params["batch_size"]
        self.add_special_tokens = False
        self.masked_lm_prob = params.get("masked_lm_prob", 0.15)
        self.max_sequence_length = params["max_sequence_length"]
        self.max_predictions_per_seq = params["max_predictions_per_seq"]
        self.scale_mlm_weights = params.get("scale_mlm_weights", False)
        self.shuffle_buffer = params.get("shuffle_buffer", 1500)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.rng = random.Random(self.shuffle_seed)
        self.mp_type = (
            "float16" if params.get("mixed_precision", False) else "float32"
        )
        self.num_special_ids = 4
        self.repeat = params.get("repeat", False)

        assert self.batch_size > 0, "Batch size should be positive."

        self.output_type_shape = {
            "input_ids_dna": {
                "output_type": "int32",
                "shape": [self.max_sequence_length],
                "pad": params.get("input_pad_id", self.pad_id[0]),
            },
            "input_ids_ideas": {
                "output_type": "int32",
                "shape": [self.max_sequence_length],
                "pad": params.get("input_pad_id", self.pad_id[0]),
            },
            "input_mask": {
                "output_type": "int32",
                "shape": [self.max_sequence_length],
                "pad": 1,
            },
            "masked_lm_positions": {
                "output_type": "int32",
                "shape": [self.max_predictions_per_seq],
                "pad": 0,
            },
            "masked_lm_ids_dna": {
                "output_type": "int32",
                "shape": [self.max_predictions_per_seq],
                "pad": params.get("mlm_pad_id", 0),
            },
            "masked_lm_ids_ideas": {
                "output_type": "int32",
                "shape": [self.max_predictions_per_seq],
                "pad": params.get("mlm_pad_id", 0),
            },
            "masked_lm_weights_dna": {
                "output_type": self.mp_type,
                "shape": [self.max_predictions_per_seq],
                "pad": 0,
            },
            "masked_lm_weights_ideas": {
                "output_type": self.mp_type,
                "shape": [self.max_predictions_per_seq],
                "pad": 0,
            },
        }

        self.MaskedLmInstance = collections.namedtuple(
            "MaskedLmInstance", ["index", "dna_label", "ideas_label"]
        )

    def _prepare_file_processor(self, metadata_files):
        all_metadata_files = metadata_files

        # get all text files to process by reading metadata files
        actual_txt_files = []
        if isinstance(all_metadata_files, str):
            with open(all_metadata_files, 'r') as _fin:
                actual_txt_files.extend(_fin.readlines())
        elif isinstance(all_metadata_files, list):
            for _file in all_metadata_files:
                with open(_file, 'r') as _fin:
                    actual_txt_files.extend(_fin.readlines())
        else:
            raise TypeError(
                f"must be a string or a list, not {type(all_metadata_files)}"
            )
        actual_txt_files = [x.strip() for x in actual_txt_files if x]
        actual_txt_files = [x for x in actual_txt_files if path.exists(x)]
        self.all_files = list(set(actual_txt_files))

        np.random.shuffle(self.all_files)
        data_path = self.all_files[0]

        # get column indices
        with open(data_path) as f:
            columns = f.readline().strip().split(",")
        try:
            self.ideas_col = columns.index("ideas")
            self.dna_seq_col = columns.index("dna_seq")
        except ValueError as e:
            print(
                "Provided CSV file must contain columns 'ideas' and 'dna_seq'"
            )
            raise (e)

    def _prepare_tokenization(self, params):
        self.tokenizer_dna = GenomeTokenizer(
            params["vocab_file_dna"], params["ngram"], params["stride"], False
        )
        self.tokenizer_ideas = GenomeTokenizer(
            params["vocab_file_ideas"], params["ngram"], params["stride"], True
        )

        # make sure special token ids
        for t in [_PAD_TOKEN, _CLS_TOKEN, _SEP_TOKEN, _MASK_TOKEN]:
            assert self.tokenizer_dna.convert_tokens_to_ids(
                [t]
            ) == self.tokenizer_ideas.convert_tokens_to_ids([t])

        # vocab ids for dna and ideas sequences
        self.vocab_ids_dna = list(self.tokenizer_dna.vocab.values())
        self.vocab_ids_ideas = list(self.tokenizer_ideas.vocab.values())

        # common tokens
        self.cls_id = self.tokenizer_dna.convert_tokens_to_ids([_CLS_TOKEN])
        self.sep_id = self.tokenizer_dna.convert_tokens_to_ids([_SEP_TOKEN])
        self.pad_id = self.tokenizer_dna.convert_tokens_to_ids([_PAD_TOKEN])
        self.mask_id = self.tokenizer_dna.convert_tokens_to_ids([_MASK_TOKEN])

    def _scale_mlm_weights(self, feature, label):
        for key in ["masked_lm_weights_dna", "masked_lm_weights_ideas"]:
            mlm_weights = feature[key]
            scale = self.batch_size / tf.reduce_sum(mlm_weights)
            feature[key] = tf.cast(mlm_weights * scale, self.mp_type)
        return feature, label

    def create_tf_dataset(self, is_training=True, input_context=None):
        # create tf dataset from generator
        output_types = (
            {
                k: getattr(tf, v["output_type"])
                for k, v in self.output_type_shape.items()
            },
            tf.int32,
        )
        padded_shapes = (
            {k: v["shape"] for k, v in self.output_type_shape.items()},
            [1],
        )
        pad_id = self.pad_id[0]
        padding_values = (
            {
                k: getattr(np, v["output_type"])(v["pad"])
                for k, v in self.output_type_shape.items()
            },
            np.int32(pad_id),
        )

        dataset = tf.data.Dataset.from_generator(
            partial(
                self._data_generator,
                self.all_files,
                self.dna_seq_col,
                self.ideas_col,
            ),
            output_types=output_types,
        )

        # shuffle, batch, repeat, then prefetch
        if is_training:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        if self.scale_mlm_weights:
            dataset = dataset.map(
                self._scale_mlm_weights,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=not (is_training and self.shuffle_seed is None),
            )
        if is_training or self.repeat:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def _data_generator(self, all_csv_files, dna_seq_col, ideas_col):
        # for each file, repeat operations
        for csv_file in all_csv_files:
            # create mlm inputs and labels for each pair of dna/ideas
            with open(csv_file, "r") as f:
                # skip first line (column names)
                _ = f.readline()

                # begin looping through lines and yielding processed data
                for line in f:
                    # Note: Because lists are stored as strings,
                    #       i.e. "[a,b,c,d]",
                    # we need to do the following to each line before and after
                    # splitting:
                    # 1. Before: Remove extra quotations mark
                    # 2. Before: Convert comma with space ", " to double space
                    #    "  " to avoid splitting on elements in the list
                    # 3. After: Convert double space "  " back to single space
                    #    "," to be converted back to list
                    line = (
                        line.strip()
                        .replace('"', "")
                        .replace(", ", "  ")
                        .split(",")
                    )
                    line = [l.replace("  ", ",") for l in line]

                    # Index 0 because self._encode returns a list
                    ideas_tokens = self._encode(
                        text=[json.loads(line[ideas_col])],
                        ideas=True,
                        max_sequence_length=self.max_sequence_length,
                        add_special_tokens=self.add_special_tokens,
                    )[0]
                    dna_tokens = self._encode(
                        text=[line[dna_seq_col]],
                        ideas=False,
                        max_sequence_length=self.max_sequence_length,
                        add_special_tokens=self.add_special_tokens,
                    )[0]

                    (
                        dna_tokens,
                        ideas_tokens,
                        masked_lm_positions,
                        masked_lm_dna_labels,
                        masked_lm_ideas_labels,
                    ) = self._create_masked_lm_predictions(
                        dna_tokens=dna_tokens,
                        ideas_tokens=ideas_tokens,
                        masked_lm_prob=self.masked_lm_prob,
                        max_predictions_per_seq=self.max_predictions_per_seq,
                    )

                    feature = {
                        "input_ids_dna": dna_tokens,
                        "input_ids_ideas": ideas_tokens,
                        "input_mask": [0] * len(dna_tokens),  # inverted mask
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_ids_dna": masked_lm_dna_labels,
                        "masked_lm_ids_ideas": masked_lm_ideas_labels,
                        "masked_lm_weights_dna": [1.0]
                        * len(masked_lm_dna_labels),  # treat all mask equally
                        "masked_lm_weights_ideas": [1.0]
                        * len(masked_lm_ideas_labels),  # treat all mask equally
                    }
                    feature = {
                        k: getattr(
                            np, self.output_type_shape[k]["output_type"]
                        )(v)
                        for k, v in feature.items()
                    }
                    label = np.int32(
                        [0]
                    )  # need to provide labels to fit signature of BERT

                    yield feature, label

    def _create_masked_lm_predictions(
        self, dna_tokens, ideas_tokens, masked_lm_prob, max_predictions_per_seq
    ):
        """
        Create mlm predictions

        Adapted from function with same name in
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py
        """

        cand_indexes = [
            [i]
            for i, token in enumerate(dna_tokens)
            if token not in [_CLS_TOKEN, _SEP_TOKEN]
        ]
        np.random.shuffle(cand_indexes)

        dna_output_tokens = list(dna_tokens)
        ideas_output_tokens = list(ideas_tokens)

        num_to_predict = min(
            max_predictions_per_seq,
            max(1, int(round(len(dna_tokens) * masked_lm_prob))),
        )

        masked_lms = []
        covered_indexes = set()

        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                # 80% of the time, replace with [MASK]
                if self.rng.random() < 0.8:
                    masked_id_dna = self.mask_id[0]
                    masked_id_ideas = self.mask_id[0]
                else:
                    # 10% of the time, keep original
                    if self.rng.random() < 0.5:
                        masked_id_dna = dna_tokens[index]
                        masked_id_ideas = ideas_tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        # skip the first 4 because they are special ids
                        masked_id_dna = self.vocab_ids_dna[
                            self.rng.randint(
                                self.num_special_ids,
                                len(self.vocab_ids_dna) - 1,
                            )
                        ]
                        masked_id_ideas = self.vocab_ids_ideas[
                            self.rng.randint(
                                self.num_special_ids,
                                len(self.vocab_ids_ideas) - 1,
                            )
                        ]

                dna_output_tokens[index] = masked_id_dna
                ideas_output_tokens[index] = masked_id_ideas

                masked_lms.append(
                    self.MaskedLmInstance(
                        index=index,
                        dna_label=dna_tokens[index],
                        ideas_label=ideas_tokens[index],
                    )
                )

        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = [p.index for p in masked_lms]
        masked_lm_dna_labels = [p.dna_label for p in masked_lms]
        masked_lm_ideas_labels = [p.ideas_label for p in masked_lms]

        # TODO: Need to add mask for ideas state as well
        # TODO: rename variable for consistnency between id and token
        return (
            dna_output_tokens,
            ideas_output_tokens,
            masked_lm_positions,
            masked_lm_dna_labels,
            masked_lm_ideas_labels,
        )

    def _encode(
        self, text, ideas, max_sequence_length, add_special_tokens=False
    ):
        tokenizer = self.tokenizer_ideas if ideas else self.tokenizer_dna

        tokenized_text = []
        for line in text:
            tokens = tokenizer.tokenize(line)
            ids = tokenizer.convert_tokens_to_ids(tokens)

            if add_special_tokens:
                ids = self.cls_id + ids
                if len(ids) < max_sequence_length:
                    ids += self.sep_id
                else:
                    ids[max_sequence_length - 1] = self.sep_id[0]

            # truncate the ids sequence to have max_sequence_length
            if max_sequence_length - len(ids) <= 0:
                ids = ids[0:max_sequence_length]

            tokenized_text.append(ids)

        return tokenized_text


class GenomeDNAOnlyDataProcessor(GenomeDataProcessor):
    """
    Genome dataset processor for BERT pre-training
    that only streams DNA data. Used for testing
    a single-MLM model version.
    Read in genome_pretrain.csv stored in _GENOME_DATA_DIR.
    Example:
        >>> from modelzoo.bert.tf.utils import get_params
        >>> from modelzoo.bert.tf.input.DataProcessor import GenomeDataProcessor
        >>> params = get_params("configs/params_bert_gsk-1.yaml")
        >>> dataset = GenomeDataProcessor(params).create_tf_dataset()
    """

    def __init__(self, params):

        super(GenomeDNAOnlyDataProcessor, self).__init__(params)
        # Override output shapes
        self.output_type_shape = {
            "input_ids": {
                "output_type": "int32",
                "shape": [self.max_sequence_length],
                "pad": params.get("input_pad_id", self.pad_id[0]),
            },
            "input_mask": {
                "output_type": "int32",
                "shape": [self.max_sequence_length],
                "pad": 1,
            },
            "masked_lm_positions": {
                "output_type": "int32",
                "shape": [self.max_predictions_per_seq],
                "pad": 0,
            },
            "masked_lm_ids": {
                "output_type": "int32",
                "shape": [self.max_predictions_per_seq],
                "pad": params.get("mlm_pad_id", 0),
            },
            "masked_lm_weights": {
                "output_type": self.mp_type,
                "shape": [self.max_predictions_per_seq],
                "pad": 0,
            },
        }

    def create_tf_dataset(self, is_training=True, input_context=None):
        # create tf dataset from generator
        output_types = (
            {
                k: getattr(tf, v["output_type"])
                for k, v in self.output_type_shape.items()
            },
            tf.int32,
        )
        padded_shapes = (
            {k: v["shape"] for k, v in self.output_type_shape.items()},
            [1],
        )
        pad_id = self.pad_id[0]
        padding_values = (
            {
                k: getattr(np, v["output_type"])(v["pad"])
                for k, v in self.output_type_shape.items()
            },
            np.int32(pad_id),
        )

        dataset = tf.data.Dataset.from_generator(
            partial(
                self._data_generator,
                self.all_files,
                self.dna_seq_col,
                self.ideas_col,
            ),
            output_types=output_types,
        )

        # shuffle, batch, repeat, then prefetch
        if is_training:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        if self.scale_mlm_weights:
            dataset = dataset.map(
                self._scale_mlm_weights,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=not (is_training and self.shuffle_seed is None),
            )
        if is_training or self.repeat:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def _scale_mlm_weights(self, feature, label):
        mlm_weights = feature["masked_lm_weights"]
        scale = self.batch_size / tf.reduce_sum(mlm_weights)
        feature["masked_lm_weights"] = tf.cast(
            mlm_weights * scale, self.mp_type
        )
        return feature, label

    def _data_generator(self, all_csv_files, dna_seq_col, ideas_col):
        # for each file, repeat operations
        for csv_file in all_csv_files:
            # create mlm inputs and labels for each pair of dna/ideas
            with open(csv_file, "r") as f:
                # skip first line (column names)
                _ = f.readline()

                # begin looping through lines and yielding processed data
                for line in f:
                    # Note: Because lists are stored as strings,
                    #       i.e. "[a,b,c,d]",
                    # we need to do the following to each line before and after
                    # splitting:
                    # 1. Before: Remove extra quotations mark
                    # 2. Before: Convert comma with space ", " to double space
                    #    "  " to avoid splitting on elements in the list
                    # 3. After: Convert double space "  " back to single space
                    #    "," to be converted back to list
                    line = (
                        line.strip()
                        .replace('"', "")
                        .replace(", ", "  ")
                        .split(",")
                    )
                    line = [l.replace("  ", ",") for l in line]

                    # Index 0 because self._encode returns a list
                    ideas_tokens = self._encode(
                        text=[json.loads(line[ideas_col])],
                        ideas=True,
                        max_sequence_length=self.max_sequence_length,
                        add_special_tokens=self.add_special_tokens,
                    )[0]
                    dna_tokens = self._encode(
                        text=[line[dna_seq_col]],
                        ideas=False,
                        max_sequence_length=self.max_sequence_length,
                        add_special_tokens=self.add_special_tokens,
                    )[0]

                    (
                        dna_tokens,
                        ideas_tokens,
                        masked_lm_positions,
                        masked_lm_dna_labels,
                        masked_lm_ideas_labels,
                    ) = self._create_masked_lm_predictions(
                        dna_tokens=dna_tokens,
                        ideas_tokens=ideas_tokens,
                        masked_lm_prob=self.masked_lm_prob,
                        max_predictions_per_seq=self.max_predictions_per_seq,
                    )

                    feature = {
                        "input_ids": dna_tokens,
                        "input_mask": [0] * len(dna_tokens),  # inverted mask
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_ids": masked_lm_dna_labels,
                        "masked_lm_weights": [1.0]
                        * len(masked_lm_dna_labels),  # treat all mask equally
                    }
                    feature = {
                        k: getattr(
                            np, self.output_type_shape[k]["output_type"]
                        )(v)
                        for k, v in feature.items()
                    }
                    label = np.int32(
                        [0]
                    )  # need to provide labels to fit signature of BERT

                    yield feature, label
