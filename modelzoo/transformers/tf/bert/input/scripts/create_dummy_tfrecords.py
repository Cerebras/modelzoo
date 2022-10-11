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
Script that generates TFRecords for a single text example
to be used for pre-training a BERT model
"""
import argparse
import random

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mlm_only",
        action="store_true",
        help="If False, TFRecords will contain two extra features for NSP which are "
        "`segment_ids` and `next_sentence_labels`.",
    )
    parser.add_argument(
        "--disable_masking",
        action="store_true",
        help="If False, TFRecords will be stored with static masks. If True, "
        "masking will happen dynamically during training.",
    )

    return parser.parse_args()


def create_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_dummy_tfrecords():
    """
    Generate TFRecords for a single example text.
    Note: The TFRecords can be used for pre-training a BERT model
    :output TFRecords file with the features (when masking is not disabled):
        :feature tf.int64 input_ids: Input token IDs. 
                 shape: [`max_seq_length`]
        :feature tf.int64 input_mask: Mask for padded positions 
                 (has values `1` on the padded positions, and `0` elsewhere).
                 shape: [`max_seq_length`]
        :feature tf.int64 masked_lm_positions: Positions of masked tokens in the sequence.
                 shape: [`max_predictions_per_seq`]
        :feature tf.int64 masked_lm_ids: IDs of masked tokens.
                 shape: [`max_predictions_per_seq`]
        :feature tf.float32 masked_lm_weights: Mask for `masked_lm_ids` and `masked_lm_positions` 
                 (has values `1.0` on the positions corresponding to masked tokens, and `0.0` elsewhere). 
                 shape: [`max_predictions_per_seq`]
        :feature tf.int64 segment_ids: Segment IDs (has values `0` on the positions corresponding to 
                 the first segment, and `1` on the positions corresponding to the second segment).
                 shape: [`max_seq_length`]
        :feature tf.int64 next_sentence_labels: Next Sentence Prediction (NSP) label 
                 (has value `1` if second segment is next sentence, and `0` otherwise). 
                 shape: [`1`]
    Note: when the flag `mlm_only` is set, the last two features `segment_ids` and
          `next_sentence_labels` are omitted.

    :output TFRecords file that contains sequence of tokens encoded into bytes using UTF-8
            (when masking is disabled)
    """
    args = parse_args()

    # parameters to generate TFRecords
    max_seq_length = 128
    max_predictions_per_seq = 20
    output_file = "sample.tfrecords"
    # percentage of input tokens to be masked
    masked_lm_prob = 0.15

    # dummy example input that will be converted into TFRecords
    input_text = "The quick brown fox jumps over the lazy dog. \
    This sentence contains all letters in the English alphabet"
    # There are five special tokens in the vocabulary:
    # `[PAD]`: used to pad the input text to a fixed length `max_seq_length`
    # `[UNK]`: assigned to unknown tokens in the input text
    # `[CLS]`: added to the beginning of the sequence
    # `[SEP]`: added to the end of each sentence
    # `[MASK]`: used to mask tokens in the input text
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    ### STEP 0: Process the raw text ###
    # Note: this step varies based on dataset and tokenization method

    # filter out whitespaces
    processed_text = "".join(input_text.split())

    # lower case the text and split the sentences
    sentences = processed_text.lower().split(".")

    ### STEP 1: Create a list of tokens for the training sample ###
    tokens = ["[CLS]"] + list(sentences[0]) + ["[SEP]"]
    if not args.mlm_only:
        tokens.extend(list(sentences[1]))
        tokens.append("[SEP]")
        segment_ids = (
            [0] + [0] * len(sentences[0]) + [0] + [1] * len(sentences[0]) + [1]
        )
        # Note: `next_sentence_label` should be randomly set for each sample to either
        # `1` (2nd sentence is next sentencee) or `0` (2nd sentence is random sentence)
        # but here it is set trivially to `1` since we have a single sample example
        next_sentence_label = [1]
    elif args.disable_masking:
        # write TFRecords containing variable length sequence of tokens
        # for MLM only pre-training with dynamic masking (masking is done in dataloader)
        array = [create_bytes_feature(token.encode()) for token in tokens]
        features = {"tokens": tf.train.FeatureList(feature=array)}
        features = tf.train.FeatureLists(feature_list=features)
        tf_example = tf.train.SequenceExample(feature_lists=features)
        with tf.io.TFRecordWriter(output_file) as writer:
            writer.write(tf_example.SerializeToString())
        return

    ### STEP 2: Map sample tokens to its indices based on the vocabulary ###
    vocab_dict = {token: i for (i, token) in enumerate(vocab)}
    input_ids = [vocab_dict[token] for token in tokens]

    ### STEP 3: Create masked predictions (`[MASK] tokens`) based on `masked_lm_prob` ###
    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(tokens) * masked_lm_prob))),
    )
    pad_length = max_predictions_per_seq - num_to_predict

    cand_idxs = [
        i
        for (i, token) in enumerate(tokens)
        if token != "[SEP]" and token != "[CLS]"
    ]
    random.shuffle(cand_idxs)
    masked_lm_positions = cand_idxs[:num_to_predict]
    masked_lm_positions.sort()
    masked_lm_positions.extend([0] * pad_length)

    masked_lm_ids = []
    for i in range(num_to_predict):
        masked_lm_ids.append(input_ids[masked_lm_positions[i]])
        input_ids[masked_lm_positions[i]] = vocab_dict["[MASK]"]

    masked_lm_ids.extend([0] * pad_length)
    masked_lm_weights = [1.0] * num_to_predict + [0] * pad_length

    ### STEP 4: Pad the sequence to reach `max_seq_length` ###
    pad_input_length = max_seq_length - len(tokens)
    input_mask = [1] * len(tokens) + [0] * pad_input_length
    input_ids.extend([vocab_dict["[PAD]"]] * pad_input_length)
    if not args.mlm_only:
        segment_ids.extend([0] * pad_input_length)

    ### STEP 5: Prepare the feature dictionary to be serialized into TFRecords ###
    features = dict()
    features["input_ids"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=input_ids)
    )
    features["input_mask"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=input_mask)
    )
    features["masked_lm_positions"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=masked_lm_positions)
    )
    features["masked_lm_ids"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=masked_lm_ids)
    )
    features["masked_lm_weights"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=masked_lm_weights)
    )

    if not args.mlm_only:
        features["segment_ids"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=segment_ids)
        )
        features["next_sentence_labels"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=next_sentence_label)
        )

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    with tf.io.TFRecordWriter(output_file) as writer:
        writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    create_dummy_tfrecords()
