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
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import jsonlines
import tqdm
import zstandard as zstd

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    Reader,
)

max_file_size = 16 * 1024 * 1024  # make output files of 16 MB


def _process_zst(jsonl_file, output_dir):
    input_file_path = os.path.join(output_dir, jsonl_file)
    output_file_path = os.path.join(output_dir, f'{jsonl_file}.zst')

    with (
        open(input_file_path, 'rb') as input_file,
        open(output_file_path, 'wb') as output_file,
    ):
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(output_file) as writer:
            for chunk in input_file:
                writer.write(chunk)

    os.remove(input_file_path)


def _write_to_zst(output_dir):
    jsonl_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(_process_zst, [(file, output_dir) for file in jsonl_files])


def _process_non_duplicates(
    non_duplicate_file, input_dir, output_dir, jsonl_key
):
    current_records = []
    total_size = 0

    if non_duplicate_file:
        tokenizable_columns = {"jsonl_key": jsonl_key}
        reader = Reader(
            f"{input_dir}/{non_duplicate_file.split('/')[1]}",
            tokenizable_columns,
        )
        for doc_id, doc in enumerate(reader.stream_data()):
            jsonl_record = {jsonl_key: doc}
            record_size = sys.getsizeof(json.dumps(jsonl_record))
            total_size += record_size
            current_records.append(jsonl_record)

            if total_size >= max_file_size:
                output_file = os.path.join(output_dir, f'output_{doc_id}.jsonl')

                with jsonlines.open(output_file, mode='a') as writer:
                    writer.write(current_records)

                current_records = []
                total_size = 0


def _process_duplicates(
    duplicate_file, duplicates, input_dir, output_dir, jsonl_key
):
    current_records = []
    total_size = 0

    tokenizable_columns = {"jsonl_key": jsonl_key}
    reader = Reader(
        f"{input_dir}/{duplicate_file.split('/')[1]}", tokenizable_columns
    )
    duplicates_set = set(duplicates[duplicate_file])

    for doc_id, doc in enumerate(reader.stream_data()):
        if doc_id not in duplicates_set:
            jsonl_record = {jsonl_key: doc}
            record_size = sys.getsizeof(json.dumps(jsonl_record))
            total_size += record_size
            current_records.append(jsonl_record)

            if total_size >= max_file_size:
                output_file = os.path.join(output_dir, f'output_{doc_id}.jsonl')

                with jsonlines.open(output_file, mode='a') as writer:
                    writer.write(current_records)

                current_records = []
                total_size = 0


def generate_duplicates(args):
    logging.info("Processing duplicates...")

    # load pickled components and other artifacts
    with open(args.input_file, "rb") as fin:
        components, n_components, reversed_mapper = pickle.load(fin)

    duplicates = defaultdict(set)
    n_duplicate_docs = 0
    for component in tqdm.tqdm(components):
        for j in range(1, len(component)):
            doc = reversed_mapper[component[j]]
            file_name, doc_idx = doc.split("@")
            duplicates[file_name].add(int(doc_idx))
            n_duplicate_docs += 1

    logging.info(
        f"Number of duplicate documents that will be removed is: {n_duplicate_docs}"
    )

    logging.info("Generating final dataset...")

    duplicate_files = set()
    for key in duplicates.keys():
        file_name = key.split('/')[1]
        duplicate_files.add(file_name)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Process every non-duplicate file as is
    non_duplicate_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file not in duplicate_files and file.endswith(args.format):
                non_duplicate_files.append(os.path.join(root, file))

    with Pool(processes=cpu_count()) as pool:
        pool.starmap(
            _process_non_duplicates,
            [
                (file, args.input_dir, args.output_dir, args.jsonl_key)
                for file in non_duplicate_files
            ],
        )

    # Process duplicate files separately.
    duplicate_files = []
    for key, value in duplicates.items():
        duplicate_files.append(key)

    with Pool(processes=cpu_count()) as pool:
        pool.starmap(
            _process_duplicates,
            [
                (
                    file,
                    duplicates,
                    args.input_dir,
                    args.output_dir,
                    args.jsonl_key,
                )
                for file in duplicate_files
            ],
        )

    _write_to_zst(args.output_dir)

    # Used for local testing.
    output_file = os.path.join(args.output_dir, "n_duplicate_docs.txt")

    with open(output_file, "w") as txt_file:
        txt_file.write(str(n_duplicate_docs))


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The connected_components.pickle file from previous stage.",
        required=True,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory, which contains the documents on which deduplication pipeline will be run.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The final output directory, where the deduplicated files would be present.",
        required=True,
    )
    parser.add_argument(
        "--format", type=str, help="File format of the dataset.", required=True
    )
    parser.add_argument(
        "--jsonl_key",
        type=str,
        help="JSONL key for the dataset.",
        required=True,
    )
    args = parser.parse_args()
    generate_duplicates(args)


if __name__ == "__main__":
    main()
