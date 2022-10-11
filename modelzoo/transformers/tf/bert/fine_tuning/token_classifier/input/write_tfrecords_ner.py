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
Based on https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/run_ner.py 
with minor modifications 

Example Usage:

python write_tfrecords_ner.py \
    --data_dir=./blurb/data_generation/data/NCBI-disease \
    --vocab_file=../../../../../vocab/uncased_pubmed_abstracts_and_fulltext_vocab.txt \
    --output_dir=./blurb/ner/ncbi-disease-tfrecords \
    --do_lower_case

"""
from __future__ import absolute_import, division, print_function

import collections
import os
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
from modelzoo.common.tf.run_utils import save_params
from modelzoo.transformers.data_processing.ner_data_processor import (
    NERProcessor,
    create_parser,
    get_tokens_and_labels,
    write_label_map_files,
)
from modelzoo.transformers.data_processing.Tokenization import FullTokenizer
from modelzoo.transformers.data_processing.utils import convert_to_unicode


def update_parser(parser):
    """
    Add required command-line arguments.
    """
    parser.add_argument(
        "--output_dir",
        required=False,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tfrecords_dir"
        ),
        help="Directory to store tfrecords",
    )
    parser.add_argument(
        "--num_training_shards",
        required=False,
        type=int,
        default=3,
        help="Number of training tf records",
    )


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, input_mask, segment_ids, label_ids,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def convert_single_example(
    ex_index, example, label_list, max_seq_length, tokenizer, out_dir
):

    label_map = write_label_map_files(label_list, out_dir)

    tokens, labels = get_tokens_and_labels(example, tokenizer, max_seq_length)

    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)

    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(tokenizer.convert_tokens_to_ids(['[PAD]'])[0])
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(label_map["[PAD]"])
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("guid: %s" % (example.guid))
        tf.compat.v1.logging.info(
            "tokens: %s"
            % " ".join(
                [
                    convert_to_unicode(x) + "__" + label_list[label_ids[i + 1]]
                    for i, x in enumerate(tokens)
                ]
            )
        )
        tf.compat.v1.logging.info(
            "input_ids: %s" % " ".join([str(x) for x in input_ids])
        )
        tf.compat.v1.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask])
        )
        tf.compat.v1.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids])
        )
        tf.compat.v1.logging.info(
            "label_ids: %s" % " ".join([str(x) for x in label_ids])
        )

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )

    return feature


def filed_based_convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    nshards,
    output_dir,
    file_prefix,
):

    output_files = [
        os.path.join(output_dir, file_prefix + "_{}.tfrecord".format(i))
        for i in range(nshards)
    ]

    writers = []
    tot_writers = nshards
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_id = 0

    num_examples_written = 0

    for (ex_index, example) in enumerate(examples):
        try:
            if ex_index % 5000 == 0:
                tf.compat.v1.logging.info(
                    "Writing example %d of %d" % (ex_index, len(examples))
                )

            feature = convert_single_example(
                ex_index,
                example,
                label_list,
                max_seq_length,
                tokenizer,
                output_dir,
            )

            def create_int_feature(values):
                f = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(values))
                )
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            # features["label_mask"] = create_int_feature(feature.label_mask)
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features)
            )
            writers[writer_id].write(tf_example.SerializeToString())

            writer_id = (writer_id + 1) % tot_writers
            num_examples_written += 1

        except:
            print(f"Skipping {ex_index} example")

    for writer in writers:
        writer.close()

    return output_files, num_examples_written


def write_tfrecords(args):

    task_name = os.path.basename(args.data_dir.lower())
    output_dir = os.path.join(os.path.abspath(args.output_dir), task_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processor = NERProcessor()

    tokenizer = FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case
    )

    nshards = {}
    nexamples = {}

    if args.data_split_type == "all":
        to_write = ["train", "test", "dev"]
    else:
        to_write = [args.data_split_type]

    for data_split_type in to_write:
        data_split_type_dir = os.path.join(output_dir, data_split_type)
        if not os.path.exists(data_split_type_dir):
            os.makedirs(data_split_type_dir)

        file_prefix = task_name + "_{}".format(data_split_type)

        if data_split_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
            label_list = processor.get_labels(data_split_type)
            nshards[data_split_type] = args.num_training_shards

        elif data_split_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
            label_list = processor.get_labels(data_split_type)
            nshards[data_split_type] = 1

        elif data_split_type == 'test':
            examples = processor.get_test_examples(args.data_dir)
            label_list = processor.get_labels(data_split_type)
            nshards[data_split_type] = 1

        _, num_examples_written = filed_based_convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            nshards[data_split_type],
            data_split_type_dir,
            file_prefix,
        )

        nexamples[data_split_type] = num_examples_written

    # Write params passed and number of examples
    params_dict = vars(args)
    params_dict["num_examples"] = nexamples
    save_params(vars(args), model_dir=args.output_dir)


def main():
    parser = create_parser()
    update_parser(parser)
    args = parser.parse_args()

    tf.compat.v1.logging.info("***** Configuration *****")
    for key, val in vars(args).items():
        tf.compat.v1.logging.info(' {}: {}'.format(key, val))
    tf.compat.v1.logging.info("**************************")

    write_tfrecords(args)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
