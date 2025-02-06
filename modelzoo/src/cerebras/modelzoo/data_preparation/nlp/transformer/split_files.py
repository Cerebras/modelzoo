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
Split input src/tgt file into multiple files.
Number of examples per file is controlled by `buffer_len` param.
"""

import argparse
import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--src_file",
        type=str,
        required=True,
        help="Path to the original source language dataset stored as one file.",
    )
    parser.add_argument(
        "--tgt_file",
        type=str,
        required=True,
        help="Path to the translated target language dataset stored as one file.",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Path to output directory with source language dataset files.",
    )
    parser.add_argument(
        "--tgt_dir",
        type=str,
        required=True,
        help="Path to output directory with target language dataset files.",
    )
    parser.add_argument(
        "--buffer_len",
        type=int,
        required=True,
        default=17581,  # in order to have 256 output files.
        help="Number of examples to store in one file.",
    )
    args = parser.parse_args()

    src_postfix = args.src_file.split("/")[-1]
    tgt_postfix = args.tgt_file.split("/")[-1]

    if not os.path.isdir(args.src_dir):
        os.mkdir(args.src_dir)
    if not os.path.isdir(args.tgt_dir):
        os.mkdir(args.tgt_dir)

    with open(args.src_file, "r") as src_fin:
        with open(args.tgt_file, "r") as tgt_fin:
            src_lines = src_fin.readlines()[::-1]
            tgt_lines = tgt_fin.readlines()[::-1]
            count = 0
            while src_lines and tgt_lines:
                result = []
                while len(result) < args.buffer_len and src_lines:
                    result.append((src_lines.pop(), tgt_lines.pop()))

                with open(
                    f"{args.src_dir}/{src_postfix}-{count}", "w"
                ) as src_fout:
                    with open(
                        f"{args.tgt_dir}/{tgt_postfix}-{count}", "w"
                    ) as tgt_fout:
                        for i, line in tqdm(
                            enumerate(result), total=len(result)
                        ):
                            src_fout.write(line[0])
                            tgt_fout.write(line[1])
                count += 1


if __name__ == "__main__":
    main()
