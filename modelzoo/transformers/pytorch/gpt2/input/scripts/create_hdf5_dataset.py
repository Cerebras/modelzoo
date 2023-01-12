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
        "--samples_per_file",
        type=int,
        default=20000,
        help="number of samples written to each HDF5 file "
        "(last file can have less samples if the dataset isn't divisible). "
        "Defaults to 20000.",
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


def write_hdf5_file(
    file_path, data, n_examples, chunks, dtype="i4", compression="gzip"
):
    """Write data to HDF5 file.

    Args:
        file_path (string): HDF5 file path.
        data (numpy array): Input features and labels that will be written to HDF5.
        n_examples (int): Number of examples that will be written in the file.
        chunks (tuple or bool): Chunk shape, or True to enable auto-chunking.
        dtype (string): Data type for the HDF5 dataset.
        compression (string): Compression strategy.
    """
    with h5py.File(file_path, mode='w') as h5_file:
        h5_file.attrs["n_examples"] = n_examples
        h5_file.create_dataset(
            "data",
            data=data,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )


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

    writer_index = 1
    total_written = 0
    example_index = 0
    data_buffer = []

    with tqdm(total=len(input_files_list)) as pbar:
        while True:
            try:
                # Wait for examples to be prepared by data workers
                # but if the queue is empty, wait for `timeout` seconds
                # to ensure there are no leftover data from the workers
                # Specifically at the begining, it takes some time for workers
                # to ramp up and load their buffers and return examples
                data_buffer.append(examples_queue.get(timeout=30))
                example_index += 1
                if example_index == args.samples_per_file:
                    file_samples = np.stack(data_buffer, axis=0)
                    fp = os.path.join(
                        output_dir, f"{args.name}-{writer_index}.h5"
                    )
                    write_hdf5_file(
                        file_path=fp,
                        data=file_samples,
                        n_examples=example_index,
                        chunks=(1, 3, args.max_seq_length),
                    )
                    example_index = 0
                    data_buffer = []
                    writer_index += 1
                total_written += 1
                if not total_written % 100:
                    pbar.update(file_counter.value() - pbar.n)
            except queue.Empty:
                # If queue has been empty for `timeout` seconds,
                # write the last file if there are examples in data_buffer,
                # then break the infinite loop and exit
                if example_index > 0:
                    file_samples = np.stack(data_buffer, axis=0)
                    fp = os.path.join(
                        output_dir, f"{args.name}-{writer_index}.h5"
                    )
                    write_hdf5_file(
                        file_path=fp,
                        data=file_samples,
                        n_examples=example_index,
                        chunks=(1, 3, args.max_seq_length),
                    )
                break

    for p in processes:
        p.join()

    params = vars(args)
    params["n_examples"] = total_written
    params["n_docs"] = count_total_documents(args.metadata_files.split(','))
    json_params_file = os.path.join(output_dir, "data_params.json")
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout)

    print(f"Done! Wrote total of {total_written} examples.")


if __name__ == "__main__":
    main()
