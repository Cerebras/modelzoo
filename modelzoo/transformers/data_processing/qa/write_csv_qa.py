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
File: write_csv_qa.py

Use to create pre-processed CSV files of SQuAD for various models. Called by {model}/fine_tuning/qa/write_csv_qa.sh with the correct command-line arguments to adjust processing for each model. 
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from modelzoo.common.input.utils import save_params
from modelzoo.transformers.data_processing.qa.qa_utils import (
    convert_examples_to_features_and_write,
    read_squad_examples,
)
from modelzoo.transformers.data_processing.tokenizers.Tokenization import (
    FullTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing train-v1.1.json",
    )
    # Note I'm re-using this arg as the path for the sentencepiece pretrained tokenizer '.model' file, but in the BERT usage it is a .txt. Not sure if there should be a separate arg for this or if its fine to preserve more compatibility
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="The vocabulary file that the T5 Pretrained model was trained on.",
    )
    parser.add_argument(
        "--data_split_type",
        choices=["train", "dev", "all"],
        default="all",
        help="Dataset split, choose from 'train', 'dev' or 'all'.",
    )
    parser.add_argument(
        "--do_lower_case",
        required=False,
        action="store_true",
        help="Whether to convert tokens to lowercase",
    )
    parser.add_argument(
        "--max_seq_length",
        required=False,
        type=int,
        default=384,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--doc_stride",
        required=False,
        type=int,
        default=128,
        help="When splitting up a long document into chunks, how much stride to "
        "take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        required=False,
        type=int,
        default=64,
        help="The maximum number of tokens for the question. Questions longer than "
        "this will be truncated to this length.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        required=False,
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
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
        default=16,
        help="number of files on disk to separate csv files into. "
        "Defaults to 16.",
    )
    parser.add_argument(
        "--tokenizer_scheme",
        required=True,
        type=str,
        help="Specify which tokenization scheme should be used based on the desired model. Currently supports BERT and T5.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print("***** Configuration *****")
    for key, val in vars(args).items():
        print(' {}: {}'.format(key, val))
    print("**************************")
    print("")

    write_csv_files(args)


def get_tokenizer_fns(args):
    if args.tokenizer_scheme == 'bert':
        tokenizer = FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=args.do_lower_case
        )
        tokenize_fn = tokenizer.tokenize
        convert_tokens_to_ids_fn = tokenizer.convert_tokens_to_ids

    elif args.tokenizer_scheme == 't5':
        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(args.vocab_file)
        tokenize_fn = tokenizer.encode_as_pieces
        convert_tokens_to_ids_fn = tokenizer.piece_to_id

    else:
        raise ValueError("Tokenization scheme for this model not supported")

    return tokenize_fn, convert_tokens_to_ids_fn


def write_csv_files(args):
    task_name = os.path.basename(args.data_dir.lower())
    output_dir = os.path.abspath(args.output_dir)
    rng = random.Random(12345)

    tokenize_fn, convert_tokens_to_ids_fn = get_tokenizer_fns(args)

    to_write = [args.data_split_type]

    if args.data_split_type == "all":
        to_write = ["train", "dev"]

    num_examples_dict = dict()

    for data_split_type in to_write:
        data_split_type_dir = os.path.join(output_dir, data_split_type)
        if not os.path.exists(data_split_type_dir):
            os.makedirs(data_split_type_dir)

        if data_split_type == "train":
            input_fn = "train-v1.1.json"
            file_prefix = "train-v1.1"
        elif data_split_type == "dev":
            input_fn = "dev-v1.1.json"
            file_prefix = "dev-v1.1"
        else:
            assert False, "Unknown data_split_type: %s" % args.data_split_type

        input_file = os.path.join(args.data_dir, input_fn)

        examples = read_squad_examples(
            input_file=input_file,
            is_training=True,
            version_2_with_negative=args.version_2_with_negative,
        )

        rng.shuffle(examples)

        (
            num_examples_written,
            meta_data,
        ) = convert_examples_to_features_and_write(
            examples=examples,
            tokenize_fn=tokenize_fn,
            convert_tokens_to_ids_fn=convert_tokens_to_ids_fn,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            output_dir=data_split_type_dir,
            file_prefix=file_prefix,
            num_output_files=args.num_output_files,
            tokenizer_scheme=args.tokenizer_scheme,
            is_training=True,
        )

        num_examples_dict[data_split_type] = num_examples_written

        meta_file = os.path.join(data_split_type_dir, "meta.dat")
        with open(meta_file, "w") as fout:
            for output_file, num_lines in meta_data.items():
                fout.write("%s %s\n" % (output_file, num_lines))

    # Write args passed and number of examples
    args_dict = vars(args)
    args_dict["num_examples"] = num_examples_dict
    save_params(args_dict, model_dir=args.output_dir)


if __name__ == "__main__":
    main()
