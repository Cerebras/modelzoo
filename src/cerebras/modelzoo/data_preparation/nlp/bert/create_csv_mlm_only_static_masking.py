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
Preprocessed CSV data generator for BERT pretraining from
raw text documents.
"""

import argparse
import csv
import json
import logging
import os
import subprocess as sp

# isort: off
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# isort: on

from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs
from cerebras.modelzoo.data_preparation.nlp.bert.mlm_only_processor import (
    data_generator,
)
from cerebras.modelzoo.data_preparation.utils import (
    count_total_documents,
    get_output_type_shapes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument(
        '--input_files_prefix',
        type=str,
        default="",
        help='prefix to be added to paths of the input files. '
        'For example, can be a directory where raw data is stored '
        'if the paths are relative',
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="path to vocabulary",
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
        "--dupe_factor",
        type=int,
        default=10,
        help="number of times to duplicate the input data (with "
        "different masks) if disable_masking is False",
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
        default="./csvfiles/",
        help="directory where CSV files will be stored. "
        "Defaults to ./csvfiles/.",
    )
    parser.add_argument(
        "--num_output_files",
        type=int,
        default=10,
        help="number of files on disk to separate csv files into. "
        "Defaults to 10.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="preprocessed_data",
        help="name of the dataset; i.e. prefix to use for csv file names. "
        "Defaults to 'preprocessed_data'.",
    )
    parser.add_argument(
        "--init_findex",
        type=int,
        default=1,
        help="Index used in first output file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed. Defaults to 0.",
    )

    return parser.parse_args()


def create_csv(
    metadata_files,
    vocab_file,
    do_lower_case,
    max_seq_length,
    short_seq_prob,
    mask_whole_word,
    max_predictions_per_seq,
    masked_lm_prob,
    dupe_factor,
    overlap_size,
    buffer_size,
    filename_prefix,
    filename_initial_index,
    output_dir,
    num_output_files,
    min_short_seq_length=None,
    multiple_docs_in_single_file=False,
    multiple_docs_separator="\n",
    single_sentence_per_line=False,
    allow_cross_document_examples=False,
    document_separator_token="[SEP]",
    inverted_mask=False,
    seed=None,
    spacy_model="en",
    input_files_prefix="",
):

    num_output_files = max(num_output_files, 1)

    output_files = [
        os.path.join(output_dir, "%s-%04i.csv" % (filename_prefix, fidx))
        for fidx in range(
            filename_initial_index, filename_initial_index + num_output_files
        )
    ]

    output_type_shapes = get_output_type_shapes(
        max_seq_length, max_predictions_per_seq, mlm_only=False
    )

    ## Names of keys of instance dictionary
    fieldnames = [
        "input_ids",
        "masked_lm_weights",
        "masked_lm_positions",
        "attention_mask",
        "labels",
    ]

    ## Create csv writers for each csv file
    writers = []
    for output_file in output_files:
        csvfile = open(output_file, 'w', newline='')
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writers.append((writer, csvfile))

    def _data_generator():
        return data_generator(
            metadata_files=metadata_files,
            vocab_file=vocab_file,
            do_lower=do_lower_case,
            disable_masking=False,
            mask_whole_word=mask_whole_word,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            masked_lm_prob=masked_lm_prob,
            dupe_factor=dupe_factor,
            output_type_shapes=output_type_shapes,
            multiple_docs_in_single_file=multiple_docs_in_single_file,
            multiple_docs_separator=multiple_docs_separator,
            single_sentence_per_line=single_sentence_per_line,
            overlap_size=overlap_size,
            min_short_seq_length=min_short_seq_length,
            buffer_size=buffer_size,
            short_seq_prob=short_seq_prob,
            spacy_model=spacy_model,
            inverted_mask=inverted_mask,
            allow_cross_document_examples=allow_cross_document_examples,
            document_separator_token=document_separator_token,
            seed=seed,
            input_files_prefix=input_files_prefix,
        )

    writer_index = 0
    total_written = 0

    for data in _data_generator():
        ## write dictionary into csv
        features, label = data
        features_dict = {
            "input_ids": list(features["input_ids"]),
            "masked_lm_weights": list(features["masked_lm_weights"]),
            "masked_lm_positions": list(features["masked_lm_positions"]),
            "attention_mask": list(features["input_mask"]),
            "labels": list(features["masked_lm_ids"]),
        }
        writers[writer_index][0].writerow(features_dict)
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

    for writer, csvfile in writers:
        csvfile.close()
    return total_written


def main():
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    check_and_create_output_dirs(args.output_dir, filetype="csv")

    total_written = create_csv(
        metadata_files=args.metadata_files,
        vocab_file=args.vocab_file,
        do_lower_case=args.do_lower_case,
        max_seq_length=args.max_seq_length,
        short_seq_prob=args.short_seq_prob,
        mask_whole_word=args.mask_whole_word,
        max_predictions_per_seq=args.max_predictions_per_seq,
        masked_lm_prob=args.masked_lm_prob,
        dupe_factor=args.dupe_factor,
        inverted_mask=False,
        seed=args.seed,
        spacy_model=args.spacy_model,
        overlap_size=args.overlap_size,
        min_short_seq_length=args.min_short_seq_length,
        buffer_size=args.buffer_size,
        allow_cross_document_examples=args.allow_cross_document_examples,
        document_separator_token=args.document_separator_token,
        multiple_docs_in_single_file=args.multiple_docs_in_single_file,
        multiple_docs_separator=args.multiple_docs_separator,
        single_sentence_per_line=args.single_sentence_per_line,
        input_files_prefix=args.input_files_prefix,
        filename_prefix=args.name,
        filename_initial_index=args.init_findex,
        output_dir=args.output_dir,
        num_output_files=args.num_output_files,
    )

    # Store arguments used for csv generation into a json file.
    params = vars(args)
    params["n_examples"] = total_written
    params["n_docs"] = count_total_documents(args.metadata_files)
    json_params_file = os.path.join(args.output_dir, "data_params.json")
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout)

    # Create meta file.
    with open(f"{args.output_dir}/meta.dat", "w") as fout:
        for file_name in os.listdir(args.output_dir):

            if file_name.startswith(args.name):
                # Calculate number of lines in the input data.
                num_lines = sp.run(
                    f"wc -l {args.output_dir}/{file_name}".split(" "),
                    stdout=sp.PIPE,
                )
                num_lines = int(num_lines.stdout.decode("utf-8").split(" ")[0])

                # The first line is dedicated to the header.
                fout.write(f"{file_name} {num_lines - 1}\n")


if __name__ == "__main__":
    main()
