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
File: write_csv_ner.py

Use to create pre-processed CSV files for the Data Processor from the NER raw dataset CSV files.

Based on https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/run_ner.py 
with minor modifications 

Example Usage:

python write_csv_ner.py \
    --data_dir /cb/ml/language/datasets/blurb/data_generation/data/BC5CDR-chem/ \
    --vocab_file /cb/ml/language/datasets/pubmed_abstracts_baseline_fulltext_vocab/Pubmed_fulltext_vocab.txt \
    --output_dir /cb/ml/language/datasets/ner-pt/bc5cdr-chem-csv \
    --do_lower_case

"""

import csv
import os

# isort: off
import sys

# isort: on

from collections import defaultdict, namedtuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from cerebras.modelzoo.common.utils.utils import save_params
from cerebras.modelzoo.data_preparation.nlp.bert.ner_data_processor import (
    NERProcessor,
    create_parser,
    get_tokens_and_labels,
    write_label_map_files,
)
from cerebras.modelzoo.data_preparation.nlp.tokenizers.Tokenization import (
    FullTokenizer,
)
from cerebras.modelzoo.data_preparation.utils import convert_to_unicode


def update_parser(parser):
    """
    Add required command-line arguments.
    """
    parser.add_argument(
        "--output_dir",
        required=False,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "preprocessed_csv_dir"
        ),
        help="Directory to store pre-processed CSV files.",
    )
    parser.add_argument(
        "--num_output_files",
        type=int,
        default=4,
        help="number of files on disk to separate csv files into. "
        "Defaults to 4.",
    )


InputFeatures = namedtuple("InputFeatures", ["tokens", "labels"])


def convert_single_example(
    ex_idx, example, label_list, max_seq_len, tokenizer, out_dir
):
    label_map = write_label_map_files(label_list, out_dir)

    tokens, labels = get_tokens_and_labels(example, tokenizer, max_seq_len)
    # add special token for input separation
    tokens.append("[SEP]")
    labels.append("[SEP]")

    # add special token for input start
    tokens.insert(0, "[CLS]")
    labels.insert(0, "[CLS]")

    if ex_idx < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print(
            "tokens: %s"
            % " ".join(
                [
                    convert_to_unicode(t) + "__" + l
                    for t, l in zip(tokens, labels)
                ]
            )
        )

    tokens = " ".join(tokens)
    labels = " ".join(labels)

    feature = InputFeatures(tokens, labels)
    return feature


def convert_examples_to_features_and_write(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_dir,
    file_prefix,
    num_output_files,
):
    num_output_files = max(num_output_files, 1)
    output_files = [
        os.path.join(output_dir, f"{file_prefix}-{fidx + 1}.csv")
        for fidx in range(num_output_files)
    ]

    # create csv writers
    meta_data = defaultdict(int)
    writers = []
    for output_file in output_files:
        csvfile = open(output_file, "w", newline="")
        writer = csv.DictWriter(
            csvfile,
            fieldnames=InputFeatures._fields,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writers.append((writer, csvfile, output_file))

    total_written = 0
    writer_idx = 0
    for ex_idx, example in enumerate(examples):
        if ex_idx % 5000 == 0:
            print(f"Writing example {ex_idx} of {len(examples)}...")
        features = convert_single_example(
            ex_idx, example, label_list, max_seq_length, tokenizer, output_dir
        )

        # Write the dict into the csv file
        features_dict = features._asdict()
        writer, _, output_file = writers[writer_idx]
        writer.writerow(features_dict)
        writer_idx = (writer_idx + 1) % len(writers)
        total_written += 1

        meta_data[os.path.basename(output_file)] += 1

    for _, csvfile, _ in writers:
        csvfile.close()

    return total_written, meta_data


def write_csv_files(args):
    task_name = os.path.basename(args.data_dir.lower())
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processor = NERProcessor()

    tokenizer = FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case
    )

    to_write = [args.data_split_type]

    if args.data_split_type == "all":
        to_write = ["train", "test", "dev"]

    nexamples = {}

    for data_split_type in to_write:
        data_split_type_dir = os.path.join(output_dir, data_split_type)
        if not os.path.exists(data_split_type_dir):
            os.makedirs(data_split_type_dir)

        file_prefix = task_name + f"{data_split_type}"

        if data_split_type == 'train':
            examples = processor.get_train_examples(args.data_dir)

        elif data_split_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)

        elif data_split_type == 'test':
            examples = processor.get_test_examples(args.data_dir)
        label_list = processor.get_labels()

        num_output_files = 1
        if data_split_type == 'train':
            num_output_files = args.num_output_files
        (
            num_examples_written,
            meta_data,
        ) = convert_examples_to_features_and_write(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            data_split_type_dir,
            file_prefix,
            num_output_files,
        )

        nexamples[data_split_type] = num_examples_written

        meta_file = os.path.join(data_split_type_dir, "meta.dat")
        with open(meta_file, "w") as fout:
            for output_file, num_lines in meta_data.items():
                fout.write(f"{output_file} {num_lines}\n")

    # Write params passed and number of examples
    params_dict = vars(args)
    params_dict["num_examples"] = nexamples
    save_params(vars(args), model_dir=args.output_dir)


if __name__ == "__main__":
    parser = create_parser()
    update_parser(parser)
    args = parser.parse_args()

    print("***** Configuration *****")
    for key, val in vars(args).items():
        print(' {}: {}'.format(key, val))
    print("**************************")

    write_csv_files(args)
