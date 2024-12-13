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
import logging
import os
from multiprocessing import cpu_count

from cerebras.modelzoo.data_preparation.data_preprocessing.data_dedup import (
    deduplicate_dataset,
    generate_connected_components,
    generate_duplicate_pairs,
    to_hash,
)


def deduplication_pipeline(args):
    logging.info("Starting deduplication pipeline...")

    # Generate MinHash
    to_hash_args = argparse.Namespace(
        dataset_name=args.dataset_name,
        input_dir=args.input_dir,
        output_dir=args.input_dir,
        job_id=0,
        jsonl_key=args.jsonl_key,
        format=args.format,
        threshold=0,
        window_size=6,
        batch_size=100,
        n_jobs=1,
    )
    to_hash.generate_hashes(to_hash_args)
    logging.info("Finished generating MinHash!")

    # Generate duplicate pairs
    duplicates_dir = os.path.join(args.input_dir, "duplicates")
    os.makedirs(duplicates_dir, exist_ok=True)

    generate_duplicate_pairs_args = argparse.Namespace(
        input_dir=to_hash_args.output_dir,
        out_file=os.path.join(
            to_hash_args.output_dir, "duplicates", "duplicate_pairs.txt"
        ),
        jaccard_threshold=0.8,
        processes=cpu_count(),
    )
    generate_duplicate_pairs.generate_pairs(generate_duplicate_pairs_args)
    logging.info("Finished generating duplicate pairs!")

    generate_connected_components_args = argparse.Namespace(
        input_dir=duplicates_dir,
        out_file=os.path.join(
            args.input_dir, "duplicates", "connected_components.pickle"
        ),
    )
    generate_connected_components.generate_connected_components_mp(
        generate_connected_components_args
    )
    logging.info("Finished generating graph of connected components!")

    # Deduplicate dataset
    os.makedirs(args.output_dir, exist_ok=True)

    deduplicate_dataset_args = argparse.Namespace(
        input_file=generate_connected_components_args.out_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        format=args.format,
        jsonl_key=args.jsonl_key,
    )

    deduplicate_dataset.generate_duplicates(deduplicate_dataset_args)
    logging.info("Finished deduplicating dataset!")

    if args.clean_up:
        logging.info("Cleaning up...")
        os.system(
            f"cd {args.input_dir} && rm -rf minhash_nfc/ duplicates/ && rm duplicate_pairs.pickle"
        )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, help="Name of the dataset", required=True
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory which contains documents, on which the deduplication pipeline needs to be run.",
        required=True,
    )
    parser.add_argument(
        "--jsonl_key",
        type=str,
        help="JSONL key for the dataset.",
        required=True,
    )
    parser.add_argument(
        "--format", type=str, help="Format of the dataset.", required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for final, deduplicated dataset.",
        required=True,
    )
    parser.add_argument(
        "--clean_up",
        type=bool,
        help="Delete intermediate, created files. By default, this is set to False, retaining hashes, the graph of duplicates etc.",
        default=False,
        required=False,
    )

    args = parser.parse_args()

    deduplication_pipeline(args)
