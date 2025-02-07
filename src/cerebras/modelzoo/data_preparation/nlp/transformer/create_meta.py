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
Create meta file for transformer in pytorch.
Stores meta file in source directory (`src_dir`).
"""

import argparse
import os
from subprocess import run


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Path to the original source language dataset.",
    )
    parser.add_argument(
        "--tgt_dir",
        type=str,
        required=True,
        help="Path to the translated target language dataset.",
    )
    args = parser.parse_args()
    result = []
    for file_name in sorted(os.listdir(args.src_dir)):
        # Counting number of lines in the files with subprocess in bash.
        cmd = f"wc -l {args.src_dir}/{file_name}"
        with open("foo.txt", "w") as fout:
            run(cmd.split(), stdout=fout)
        with open("foo.txt", "r") as fin:
            num_examples = int(fin.read().split()[0])
        result.append((file_name, num_examples))

    total_num_examples = 0
    with open(f"{args.src_dir}/meta.dat", "w") as fout:
        for i, (file_name, num_examples) in enumerate(result):
            total_num_examples += num_examples
            fout.write(
                f"{args.src_dir}/{file_name} {args.tgt_dir}/{file_name.split('en')[0]}de{file_name.split('en')[1]} {num_examples}"
            )
            if i != len(result) - 1:
                fout.write("\n")

    print(f"Number of examples: {total_num_examples}.")


if __name__ == "__main__":
    main()
