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
Script that generates a dataset in HDF5 format for GPT2Model.
"""
import argparse
import json
import multiprocessing
import os
import queue
import random
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from modelzoo.common.input.utils import check_and_create_output_dirs
from modelzoo.transformers.data_processing.utils import count_total_documents
from modelzoo.transformers.pytorch.gpt2.input.data_processor_utils import (
    training_data_generator,
)
from modelzoo.transformers.pytorch.gpt2.input.scripts.utils import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_files",
        type=str,
        required=True,
        help="path to text file containing a list of file names "
        "corresponding to the raw input documents to be "
        "processed and stored; can handle multiple metadata files "
        "separated by comma",
    )
    parser.add_argument(
        "--vocab_file", type=str, required=True, help="path to vocabulary"
    )
    parser.add_argument(
        "--encoder_file", type=str, required=True, help="path to BPE encoder"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum sequence length. Defaults to 128.",
    )
    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.0,
        help="probability of creating sequences which are shorter "
        "than the maximum sequence length. Defaults to 0.0.",
    )
    parser.add_argument(
        "--add_special_tokens",
        type=bool,
        default=True,
        help="Add '<endoftext>' token at the end of each document. "
        "Defaults to True.",
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=None,
        help="overlap size for generating sequences from buffered data. "
        "Defaults to None, which sets the overlap to max_seq_len/4.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hdf5_dataset/",
        help="directory where HDF5 will be stored. "
        "Defaults to ./hdf5_dataset/'.'",
    )
    parser.add_argument(
        "--num_output_files",
        type=int,
        default=64,
        help="number of files on disk to separate HDF5 dataset into. "
        "Defaults to 64.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset-partition",
        help="name of the dataset; i.e. prefix to use for HDF5 file names. "
        "Defaults to 'dataset-partition'.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed. Defaults to 0.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="number of Python processes to use for generating data. "
        "Defaults to 8.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    check_and_create_output_dirs(output_dir, filetype="h5")

    metadata_files = args.metadata_files.split(',')
    # get all text files by reading metadata files
    if isinstance(metadata_files, str):
        metadata_files = [metadata_files]

    input_files = []
    for _file in metadata_files:
        with open(_file, 'r') as _fin:
            input_files.extend(_fin.readlines())

    input_files_list = [x.strip() for x in input_files if x]

    random.seed(args.seed)
    random.shuffle(input_files_list)

    num_output_files = max(args.num_output_files, 1)
    output_files = [
        os.path.join(output_dir, f"{args.name}-{fidx+1}.h5")
        for fidx in range(num_output_files)
    ]

    writers = []
    for output_file in output_files:
        writer = h5py.File(output_file, mode='w')
        writer.attrs["n_examples"] = 0
        writers.append(writer)

    features_list = ["input_ids", "attention_mask", "labels"]

    def _data_generator_worker(
        worker_id, input_files, examples_queue, file_counter
    ):
        random.seed(args.seed + worker_id)
        prev_file_num = 0
        for example, file_num in training_data_generator(
            input_files,
            args.vocab_file,
            args.encoder_file,
            args.max_seq_length,
            buffer_size=1e6,
            overlap_size=args.overlap_size,
            short_seq_prob=args.short_seq_prob,
            inverted_mask=False,
            add_special_tokens=args.add_special_tokens,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
        ):
            example_np_array = np.concatenate(
                [
                    np.expand_dims(example[feature], axis=0)
                    for feature in features_list
                ],
                axis=0,
            )
            examples_queue.put(example_np_array)
            if file_num > prev_file_num:
                file_counter.increment(file_num - prev_file_num)
                prev_file_num = file_num
        file_counter.increment(len(input_files) - prev_file_num)

    num_processes = args.num_processes
    examples_queue = multiprocessing.Queue()
    processes = []
    print(f"Running data generator with {num_processes} processes!")
    file_counter = Counter(0)

    # Shard the input files between processes and start
    per_worker_num_files = len(input_files_list) // num_processes
    for process_id in range(num_processes):
        if process_id < num_processes - 1:
            input_files = input_files_list[
                (process_id * per_worker_num_files) : (
                    (process_id + 1) * per_worker_num_files
                )
            ]
        else:
            input_files = input_files_list[
                (process_id * per_worker_num_files) :
            ]
        p = multiprocessing.Process(
            target=_data_generator_worker,
            args=(process_id, input_files, examples_queue, file_counter),
        )
        processes.append(p)
        p.start()

    writer_index = 0
    total_written = 0

    with tqdm(total=len(input_files_list)) as pbar:
        while True:
            try:
                # Wait for examples to be prepared by data workers
                # but if the queue is empty, wait for `timeout` seconds
                # to ensure there are no leftover data from the workers
                # Specifically at the begining, it takes some time for workers
                # to ramp up and load their buffers and return examples
                example = examples_queue.get(timeout=30)
                writers[writer_index].create_dataset(
                    f"example_{total_written // num_output_files}", data=example
                )
                writers[writer_index].attrs["n_examples"] += 1
                writer_index = (writer_index + 1) % num_output_files
                total_written += 1
                if not total_written % 100:
                    pbar.update(file_counter.value() - pbar.n)
            except queue.Empty:
                # If queue has been empty for `timeout` seconds,
                # there is no data left to be processed, so break
                # the infinite loop and exit
                break

    for p in processes:
        p.join()

    for writer in writers:
        writer.close()

    params = vars(args)
    params["n_examples"] = total_written
    params["n_docs"] = count_total_documents(args.metadata_files.split(','))
    json_params_file = os.path.join(output_dir, "data_params.json")
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout)

    print(f"Done! Wrote total of {total_written} examples.")


if __name__ == "__main__":
    main()
