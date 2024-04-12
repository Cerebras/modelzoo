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

import argparse


def add_common_parser_args(parser):
    parser.add_argument(
        "--metadata_files",
        type=str,
        required=True,
        nargs='+',
        help="path to text file containing a list of file names "
        "corresponding to the raw input documents to be "
        "processed and stored; can handle multiple metadata files "
        "separated by space",
    )
    parser.add_argument(
        "--multiple_docs_in_single_file",
        action="store_true",
        help="Pass this flag when a single text file contains multiple"
        " documents separated by <multiple_docs_separator>",
    )
    parser.add_argument(
        "--multiple_docs_separator",
        type=str,
        default="\n",
        help="String which separates multiple documents in a single text file. "
        "If newline character, pass \\n"
        "There can only be one separator string for all the documents.",
    )
    parser.add_argument(
        "--single_sentence_per_line",
        action="store_true",
        help="Pass this flag when the document is already split into sentences with"
        "one sentence in each line and there is no requirement for "
        "further sentence segmentation of a document ",
    )
    parser.add_argument(
        '--input_files_prefix',
        type=str,
        default="",
        help='prefix to be added to paths of the input files. '
        'For example, can be a directory where raw data is stored '
        'if the paths are relative',
    )
    parser.add_argument(
        "--vocab_file", type=str, required=True, help="path to vocabulary"
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=1000,
        help="number of input files to read at a given time for processing. "
        "Defaults to 1000.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="pass this flag to lower case the input text; should be "
        "True for uncased models and False for cased models",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.1,
        help="probability of creating sequences which are shorter "
        "than the maximum sequence length",
    )
    parser.add_argument(
        "--min_short_seq_length",
        type=int,
        default=None,
        help="The minimum number of tokens to be present in an example"
        "if short sequence probability > 0."
        "If None, defaults to 2 "
        "Allowed values are [2, max_seq_length - 3)",
    )
    parser.add_argument(
        "--masked_lm_prob",
        type=float,
        default=0.15,
        help="masked LM probability",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        type=int,
        default=20,
        help="maximum number of masked LM predictions per sequence",
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to load, i.e. shortcut link, package name or path.",
    )
    parser.add_argument(
        "--mask_whole_word",
        action="store_true",
        help="whether to use whole word masking rather than per-WordPiece "
        "masking.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory where HDF5 files will be stored.",
    )
    parser.add_argument(
        "--num_output_files",
        type=int,
        default=10,
        help="number of output files in total i.e each process writes num_output_files//num_processes number of files"
        "Defaults to 10.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="preprocessed_data",
        help="name of the dataset; i.e. prefix to use for hdf5 file names. "
        "Defaults to 'preprocessed_data'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed. Defaults to 0.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=0,
        help="Number of parallel processes to use, defaults to cpu count",
    )
    return parser


def add_mlm_only_specific_args(parser):
    parser = add_common_parser_args(parser)
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=None,
        help="overlap size for generating sequences from buffered data for mlm only sequences"
        "Defaults to None, which sets the overlap to max_seq_len/4.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1e6,
        help="buffer_size number of elements to be processed at a time",
    )
    parser.add_argument(
        "--allow_cross_document_examples",
        action="store_true",
        help="Pass this flag when examples can cross document boundaries",
    )
    parser.add_argument(
        "--document_separator_token",
        type=str,
        default="[SEP]",
        help="If examples can span documents, "
        "use this separator to indicate separate tokens of current and next document",
    )
    # This is a suppressed argument that will not show in --help.
    # Users MUST NOT specify this arg. It is used to switch between the two modes
    # defined to generate HDF5 files.
    parser.add_argument(
        '--__mode', default="mlm_only", help=argparse.SUPPRESS, required=False
    )
    return parser


def add_mlm_nsp_specific_args(parser):
    parser = add_common_parser_args(parser)
    # This is a suppressed argument that will not show in --help.
    # Users MUST NOT specify this arg. It is used to switch between the two modes
    # defined to generate HDF5 files.
    parser.add_argument(
        '--__mode', default="mlm_nsp", help=argparse.SUPPRESS, required=False
    )
    return parser


def create_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers()

    mlm_subparser = subparsers.add_parser("mlm_only")
    mlm_subparser = add_mlm_only_specific_args(mlm_subparser)

    mlm_nsp_subparser = subparsers.add_parser("mlm_nsp")
    mlm_nsp_subparser = add_mlm_nsp_specific_args(mlm_nsp_subparser)

    return parser


def get_parser_args():
    parser = create_arg_parser()
    args = parser.parse_args()
    return args
