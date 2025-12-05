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
Preprocess a dataste saved in the Eleuther `lm_dataformat` format such as Pile
for use in a data processor such as the `GptG5MapDataProcessor` which is backed
by a `H5Reader`.

The basic logic in this script is to convert each input file to a single
H5 output file by appying unicode normalization, tokenizing, and concatenating
documents with an end of document token in between.

This script is meant to be run in parallel across several nodes using a tool
such as `sbatch`. For example, to preprocess Pile from the raw artifacts
downloaded from https://the-eye.eu/public/AI/pile/, run the following slurm
script using `sbatch --array 0-29`:
```
#!/bin/bash
python preprocess_pile.py \
       --input_path /path/to/raw/pile/train/*.jsonl.zst \
       --output_dir /path/to/output/dir \
       --tokenizer /path/to/gpt2/tokenizer.json \
       --eos_id 50256 \
       --normalizer NFC \
       --rank $SLURM_ARRAY_TASK_ID \
       --world_size $SLURM_ARRAY_TASK_COUNT
```
The files provided are automatically sharded beween workers based on the
provided rank and world size which results in each worker processing a single
file. The script is also functional although less parallel if you reduce the
worker pool (potentially to only a single worker) and let each worker process
multiple files. The only change needed would be in the `--array` sbatch
argument.

This script assumes that the documents in the source dataset are already
shuffled, which is the case for the typical Pile download.
"""

import argparse
import json
import logging
import os

import git
import h5py
import numpy as np
from tokenizers import Tokenizer

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        nargs="+",
        required=True,
        help="a set of paths to raw data files to convert",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="the base directory for saving outputs",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="a path to a HuggingFace Tokenizers configuration file",
    )
    parser.add_argument(
        "--eos_id",
        type=int,
        required=True,
        help="an integer corresponding to the end of document token",
    )
    parser.add_argument(
        "--normalizer",
        choices=["NFC", "NFKC"],
        help="the type of unicode normalization to use, defaults to no normalization",
    )
    parser.add_argument(
        "--no_ftfy",
        action="store_true",
        help="don't normalize the input documents using the `ftfy` package",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="overwrite previous output if it exists",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="the index of the current task if running in a distributed setting",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="the total number of simultaneous tasks if running in a distributed setting",
    )
    parser.add_argument(
        "--no_artifacts",
        action="store_true",
        help=(
            "disable storing the git state and arguments used to run this script"
            " which can be useful if running outside the modelzoo repo"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        raise ValueError(
            f"tokenizer must be a valid file, got {args.tokenizer}"
        )
    if args.world_size < 1:
        raise ValueError(
            f"world_size must be greater than or equal to 1, got {args.world_size}"
        )
    if not (0 <= args.rank < args.world_size):
        raise ValueError(
            f"You must have 0 <= rank < world_size, got rank = {args.rank}"
        )
    for f in args.input_path:
        if not os.path.isfile(f):
            raise ValueError(
                f"found {f}, which is not a valid file, in input path"
            )
    return args


def save_run_info(args):
    if not args.rank:
        # only save run information for one worker process
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        state = {"args": vars(args), "git_state": sha}
        out_path = os.path.join(args.output_path, "preprocessing_args.json")
        with open(out_path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    if not args.no_ftfy:
        try:
            import ftfy
        except Exception:
            raise ValueError(
                "Preprocessing requires the ftfy package, which is not included in "
                "the default python environment. Please install ftfy manually and "
                "rerun."
            )
    try:
        from lm_dataformat import Reader
    except Exception:
        raise ValueError(
            "Preprocessing requires the lm_dataformat package, which is not "
            "included in the default python environment. Please install "
            "lm_dataformat manually and rerun."
        )

    input_files = sorted(args.input_path)
    input_files = input_files[args.rank :: args.world_size]
    logger.info(f"processing files {input_files}")

    if os.path.isfile(args.output_dir):
        if args.overwirte_output:
            os.remove(args.output_dir)
        else:
            raise ValueError(
                f"Output directory {args.output_dir} already exists as a file. "
                f"To overwrite this file, rerun with `--overwrite_output`"
            )
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not args.no_artifacts:
        save_run_info(args)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size < 2**8:
        dtype = np.uint8
        dtype_str = "uint8"
    elif vocab_size < 2**16:
        dtype = np.uint16
        dtype_str = "uint16"
    elif vocab_size < 2**32:
        dtype = np.uint32
        dtype_str = "uint32"
    else:
        raise ValueError(
            f"only vocab sizes up to 2^32 - 1 are currently supported, got "
            f"{vocab_size}"
        )

    for i, filename in enumerate(input_files):
        output_path = os.path.join(
            args.output_dir, f"{args.rank:02d}-{i:02d}.h5"
        )
        logger.info(
            f"processing input file {filename} into output file {output_path}"
        )
        if os.path.exists(output_path) and not args.overwrite_output:
            raise ValueError(
                f"file {output_path} already exists. To overwrite, rerun with "
                "--overwrite_output"
            )
        reader = Reader(filename)
        corpus = []
        for doc in reader.stream_data():
            if not args.no_ftfy:
                doc = ftfy.fix_text(doc, normalization=args.normalizer)
            doc = tokenizer.encode(doc).ids + [args.eos_id]
            corpus.extend(doc)
        corpus = np.array(corpus, dtype=dtype)
        with h5py.File(output_path, "w") as f:
            dset = f.create_dataset("data", (len(corpus),), dtype=dtype_str)
            dset[:] = corpus


if __name__ == "__main__":
    main()
