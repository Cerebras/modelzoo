# This code is adapted from
# https://github.com/google-research/bert/blob/master/run_squad.py
#
# coding=utf-8
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 The Google AI Language Team Authors.
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
"""Generate TFRecords for SQuAD 1.1 and SQuAD 2.0."""

import os
import random
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
from modelzoo.transformers.data_processing.qa_utils import (
    convert_examples_to_features,
    read_squad_examples,
)
from modelzoo.transformers.data_processing.Tokenization import FullTokenizer
from modelzoo.transformers.tf.bert.fine_tuning.qa.input.utils import (
    FeatureWriter,
)

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "vocab_file",
    None,
    "The vocabulary file that the BERT model was trained on.",
)

flags.DEFINE_string("input_file", None, "SQuAD json. E.g., train-v1.1.json")

flags.DEFINE_string(
    "output_dir", None, "The directory for the TFRecords to be written in",
)

## Other parameters
flags.DEFINE_string(
    "output_file", "train.tf_record", "The file name for the TFRecords.",
)

flags.DEFINE_bool(
    "is_training", True, "Whether the data being generated is for training."
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.",
)

flags.DEFINE_integer(
    "max_seq_length",
    384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_integer(
    "doc_stride",
    128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.",
)

flags.DEFINE_integer(
    "max_query_length",
    64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.",
)

flags.DEFINE_bool(
    "verbose_logging",
    False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.",
)

flags.DEFINE_bool(
    "version_2_with_negative",
    False,
    "If true, the SQuAD examples contain some that do not have an answer.",
)


def validate_flags_or_throw():
    """Validate the input FLAGS or throw an exception."""

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length)
        )


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    validate_flags_or_throw()

    tf.compat.v1.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case
    )
    tokenize_fn = tokenizer.tokenize
    convert_tokens_to_ids_fn = tokenizer.convert_tokens_to_ids
    tokenizer_scheme = "bert"

    examples = read_squad_examples(
        input_file=FLAGS.input_file,
        is_training=FLAGS.is_training,
        version_2_with_negative=FLAGS.version_2_with_negative,
    )

    rng = random.Random(12345)
    rng.shuffle(examples)

    writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, FLAGS.output_file),
        is_training=FLAGS.is_training,
    )
    convert_examples_to_features(
        examples=examples,
        tokenize_fn=tokenize_fn,
        convert_tokens_to_ids_fn=convert_tokens_to_ids_fn,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        tokenizer_scheme=tokenizer_scheme,
        is_training=FLAGS.is_training,
        output_fn=writer.process_feature,
    )
    writer.close()

    tf.compat.v1.logging.info("***** Examples written *****")
    tf.compat.v1.logging.info("  Num orig examples = %d", len(examples))
    tf.compat.v1.logging.info("  Num split examples = %d", writer.num_features)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
