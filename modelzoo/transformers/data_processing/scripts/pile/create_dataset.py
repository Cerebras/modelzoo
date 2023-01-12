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

# This code is adapated from
# https://github.com/EleutherAI/gpt-neo/blob/master/data/create_tfrecords.py
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
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
Script that generates a dataset in tfrecords or HDF5 format for GPT Models.
"""

import argparse
import json
import logging
import os
import sys
from itertools import repeat
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from modelzoo.common.input.utils import check_and_create_output_dirs
from modelzoo.transformers.data_processing.scripts.pile.tokenizer import (
    get_tokenizer,
)
from modelzoo.transformers.data_processing.scripts.pile.utils import (
    archive_to_tokens,
    get_files,
    read_checkpoint,
    seed_runs,
    split_list,
    write_files,
)


def parse_args():
    """Argparser definition for command line arguments from user.

    Returns:
        Argparse namespace object with command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process the raw Pile dataset as TfRecords/HDF5."
    )
    parser.add_argument(
        "--file_format",
        type=str,
        required=True,
        choices=["tfrecords", "HDF5"],
        help=(
            f"output file format for the dataset. Can be one of"
            + f" `tfrecords` or `HDF5`"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="directory where raw data is stored.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        required=True,
        choices=["GPT2Tokenizer", "NeoXTokenizer"],
        help=(
            f"type of tokenizer to use for tfrecord/HDF5 dataset generation. Can be one of"
            + f" `GPT2Tokenizer` or `NeoXTokenizer`"
        ),
    )
    parser.add_argument(
        "--vocab_file", type=str, required=True, help="path to vocabulary"
    )
    parser.add_argument(
        "--encoder_file", type=str, default=None, help="path to BPE encoder"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="maximum sequence length. Defaults to 2048.",
    )
    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.0,
        help=(
            "probability of creating sequences which are shorter than the"
            + " maximum sequence length. Defaults to 0.0"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data_dir/",
        help="directory where TFRecords/HDF5 files will be stored. Defaults to `./data_dir/`.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="examples",
        help=(
            "name of the dataset; i.e. prefix to use for TFRecord/HDF5 file names."
            + "Defaults to `examples`."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed. Defaults to `0`.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="Number of processes to use. Default to cpu count.",
    )
    parser.add_argument(
        "--ftfy", action="store_true", help="Fix text with ftfy"
    )
    parser.add_argument(
        "--ftfy_normalizer",
        type=str,
        default="NFC",
        choices=["NFC", "NFKC", "None"],
        help=(
            "choose what kind of unicode normalization is applied. Usually, we"
            + " apply `NFC` normalization, so that letters followed by combining"
            + " characters become single combined characters. Changing this to"
            + " `NFKC` applies more compatibility conversions. Using `None`"
            + " applies no normalization while fixing text."
        ),
    )
    parser.add_argument(
        "--wikitext-detokenize",
        action="store_true",
        help="use wikitext detokenizer to fix text",
    )
    parser.add_argument(
        "--write_remainder",
        action="store_true",
        help="write the remainder files when data is left over from processing",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="resume record writing from a given checkpoint",
    )
    parser.add_argument(
        "--display_pbar",
        action="store_true",
        help="display progress while runs",
    )
    parser.add_argument(
        "--eos_id",
        type=int,
        default=50256,
        help=(
            "id for padding out shorter sequnces. Defaults to 50256, which"
            + " is `<|endoftext|>` in tokens."
        ),
    )
    parser.add_argument(
        "--pad_id",
        type=int,
        default=50256,
        help=(
            "id for padding out shorter sequnces. Defaults to 50256, which"
            + " is `<|endoftext|>` in tokens."
        ),
    )
    parser.add_argument(
        "--files_per_record",
        type=int,
        default=50000,
        help="Text files to write per tfrecord/HDF5 file",
    )
    parser.add_argument(
        "--write_in_batch",
        action="store_true",
        help="Whether to write the samples in batch for the HDF5 format, setting to false will save memory but a bit slower",
    )
    return parser.parse_args()


def create_dataset(params):
    """Creates TfRecords/HDF5 dataset from given parameters.

    Args:
        params (tuple): Tuple containing the files, arguments and process number
            for current execution.

    Returns:
        Dictionary containing results of execution, specifically as number of
            processed, discarded, and successful files as well as number of examples.
    """
    files, args, process_no = params
    rng = seed_runs(args.seed, rank=process_no)

    write_remainder = args.write_remainder
    resume_from_checkpoint = args.resume_from_checkpoint
    display_pbar = args.display_pbar

    tokenizer, eos_id, pad_id = get_tokenizer(args)

    # re-assign eos_id, pos_id to the args value since it is used for generation
    if not isinstance(eos_id, list):
        eos_id = [eos_id]
    args.eos_id = eos_id
    args.pad_id = pad_id

    discarded_files = 0
    files_processed = 0
    pbar = tqdm(
        desc=f"Parsed 0 input files. Files written ", disable=not display_pbar,
    )
    checkpoint_path = f"{args.output_dir}/checkpoint.txt"
    resume_files_processed, df_count = read_checkpoint(
        checkpoint_path, resume_from_checkpoint
    )

    data_to_prepend = []
    tokenized_files_array = []

    for _file in files:
        for tokenized_files in archive_to_tokens(
            _file, tokenizer, args, prefix=data_to_prepend
        ):
            files_processed += 1
            if files_processed < resume_files_processed:
                continue  # enable resuming from checkpoint

            # if the last chunk < chunk size, but > minimum_size, take it
            # and append it to the beginning of the next file
            if data_to_prepend:
                data_to_prepend.clear()
            n_tokens = len(tokenized_files[-1])
            if n_tokens < args.max_seq_length:
                data = tokenized_files.pop(-1)
                if n_tokens > 0:
                    data_to_prepend.extend(data)
                else:
                    discarded_files += 1

            # add tokenized files > chunk size to main array
            tokenized_files_array.extend(tokenized_files)

            if len(tokenized_files_array) >= args.files_per_record:
                _df_count, remainder = write_files(
                    tokenized_files_array,
                    args,
                    start_number=df_count,
                    process_number=process_no,
                    rng=rng,
                )
                pbar.update(_df_count - df_count)
                pbar.set_description(
                    f"Parsed {files_processed} input files. Files written "
                )

                df_count = _df_count
                tokenized_files_array = (
                    remainder if remainder is not None else []
                )  # add remaining files to next chunk
                with open(checkpoint_path, "w") as checkpoint_file:
                    checkpoint_file.write(f"{files_processed}, {df_count}")

    if len(tokenized_files_array) >= args.files_per_record:
        _df_count, remainder = write_files(
            tokenized_files_array,
            args,
            start_number=df_count,
            process_number=process_no,
            rng=rng,
        )
        pbar.update(_df_count - df_count)
        pbar.set_description(
            f"Parsed {files_processed} input files. Files written "
        )
        df_count = _df_count
        with open(checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(f"{files_processed}, {df_count}")
    else:
        remainder = tokenized_files_array

    n_examples = df_count * args.files_per_record + len(remainder)
    if write_remainder:
        write_files(
            remainder,
            args,
            start_number=df_count,
            write_remainder=True,
            process_number=process_no,
            rng=rng,
        )

    successful_files = files_processed - discarded_files
    return {
        "discarded": discarded_files,
        "processed": files_processed,
        "successful": successful_files,
        "examples": n_examples,
    }


def create_dataset_mp(files, args):
    """Create TFRecords/HDF5 dataset using multiple processes.

    Args:
        files (list): List of files to process.
        args (argparse namespace): Arguments for writing out tfrecords/HDF5 dataset.

    Returns:
        Dictionary containing results of execution, specifically as number of
            processed, discarded, and successful files as well as number of examples 
            from all processes.
    """
    try:
        files = split_list(files, len(files) // args.processes)
    except ValueError as e:
        # We hit errors in two potential scenarios,
        # 1) Files is an empty list, in which case there is nothing to split
        # 2) There are more processes than files, in which case we cannot split
        #    the files to processes correctly, as there will be many ideal
        #    processes which are not doing anything.
        print(e)
        raise

    with Pool(processes=args.processes) as pool:
        pbar = tqdm(
            pool.imap(
                create_dataset, zip(files, repeat(args), range(len(files)))
            )
        )
        meta = {"discarded": 0, "processed": 0, "successful": 0, "examples": 0}
        for results in pbar:
            pbar.update()
            for k, v in results.items():
                meta[k] += v

        return meta


def main():
    """Main function for execution."""

    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    output_dir = args.output_dir

    if args.file_format == "tfrecords":
        check_and_create_output_dirs(output_dir, filetype="tfrecord")
    elif args.file_format == "HDF5":
        check_and_create_output_dirs(output_dir, filetype="h5")
    else:
        raise Exception(
            "Only supports `tfrecords` or `HDF5` file formats for now."
        )
    input_files = get_files(args.input_dir)

    json_params_file = os.path.join(args.output_dir, "data_params.json")
    print(
        f"\nStarting writing data to {args.output_dir}."
        + f" User arguments can be found at {json_params_file}."
    )

    # write initial params to file
    params = vars(args)
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout, indent=4, sort_keys=True)

    # fix args for ftfy normalization
    if args.ftfy_normalizer == "None":
        args.ftfy_normalizer = None

    if args.processes == 0:
        # if nothing is specified, then set number of processes to CPU count.
        args.processes = cpu_count()

    if args.processes > 1:
        results = create_dataset_mp(input_files, args)
    else:
        # Run only single process run, with process number set as 0.
        results = create_dataset((input_files, args, 0))

    # Write outputs of execution
    with open(json_params_file, 'r') as _fin:
        data = json.load(_fin)

    data["discarded_files"] = results["discarded"]
    data["processed_files"] = results["processed"]
    data["successful_files"] = results["successful"]
    data["n_examples"] = results["examples"]
    data["eos_id"] = args.eos_id[0]
    data["pad_id"] = args.pad_id

    with open(json_params_file, 'w') as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)

    print(
        f"\nFinished writing data to {args.output_dir}."
        + f" Runtime arguments and outputs can be found at {json_params_file}."
    )


if __name__ == "__main__":
    main()
