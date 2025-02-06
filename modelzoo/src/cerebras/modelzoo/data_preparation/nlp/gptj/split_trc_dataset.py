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
Split input src file into multiple files.
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
        "--input_file",
        type=str,
        required=True,
        help="Path to the original source language dataset stored as one file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory with source language dataset files.",
    )
    parser.add_argument(
        "--buffer_len",
        type=int,
        default=10000,
        help="Number of examples to store in one file.",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of the output number of files to be considered as "
        "validation dataset.",
    )
    args = parser.parse_args()

    src_postfix = args.input_file.split("/")[-1]
    split_dir = os.path.join(args.out_dir, "split_files")

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
        os.mkdir(split_dir)
    out_files_list = []

    with open(args.input_file, "r") as src_fin:
        src_lines = src_fin.readlines()
    count = 0
    start_idx = 0
    end_idx = min(args.buffer_len, len(src_lines))
    num_out_files = (len(src_lines) // args.buffer_len) + 1
    pbar = tqdm(range(num_out_files))
    while end_idx < len(src_lines):
        split_lines = src_lines[start_idx:end_idx]
        shard_out_file_name = os.path.join(split_dir, f"{src_postfix}-{count}")
        out_files_list.append(shard_out_file_name)
        with open(shard_out_file_name, "w") as src_fout:
            src_fout.writelines(split_lines)
        count += 1
        start_idx += args.buffer_len
        end_idx = min(end_idx + args.buffer_len, len(src_lines))
        pbar.update()

    val_count = int(args.val_split_ratio * len(out_files_list))
    train_count = len(out_files_list) - val_count
    train_files = out_files_list[:train_count]
    val_files = out_files_list[train_count:]
    with open(os.path.join(args.out_dir, "train_meta.txt"), "w") as fid:
        fid.writelines([file + "\n" for file in train_files])
    with open(os.path.join(args.out_dir, "val_meta.txt"), "w") as fid:
        fid.writelines([file + "\n" for file in val_files])


if __name__ == "__main__":
    main()
