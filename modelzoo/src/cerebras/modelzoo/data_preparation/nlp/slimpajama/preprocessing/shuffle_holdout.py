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
import gc
import os

# 2 pass shuffling algorithm: https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
import queue
import random
import time
from multiprocessing import Process, Queue

import lm_dataformat as lmd
from more_itertools import chunked
from tqdm import tqdm

# isort: off
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
# isort: on
from cerebras.modelzoo.data_preparation.nlp.slimpajama.preprocessing.datasets import (
    RedPajamaReplication,
    redpj_datasets,
)
from cerebras.modelzoo.data_preparation.nlp.slimpajama.utils import (
    rm_if_exists,
    write_lmd_dataset,
)


def write_docs(q, ar, archive_name):
    i = 0
    start_time = time.time()
    while True:
        try:
            doc = q.get(timeout=60)
            ar.add_data(doc["doc"], doc["meta"])
            i += 1
            if i % 10000 == 0:
                ar.commit(archive_name=archive_name + str(ar.i))
                print(
                    f"Total number of processed documents: {i} ",
                    f"Total time: {time.time() - start_time}",
                )
        except queue.Empty:
            ar.commit(archive_name="redpajama" + str(ar.i))
            break
    print("Finished writing documents.")


def pass_1_shuffle(
    redpajama_dataset, output_dir_path="./", archive_name="redpajama"
):
    # We create piles of the dataset and store them as lmd;
    rm_if_exists(output_dir_path)
    os.mkdir(output_dir_path)
    total_bytes = redpajama_dataset.size()

    n_process = 20
    ars = [
        lmd.Archive(f"{output_dir_path}/chunk{i}", threads=10)
        for i in range(n_process)
    ]

    # queue to collect documents from reading processes
    docs_queue = [Queue(64 * 10000) for _ in range(n_process)]
    # returning a list of reading processes
    r_procs, manager = redpajama_dataset.documents(docs_queue)
    w_procs = []
    for process_id in range(n_process):
        p = Process(
            target=write_docs,
            args=(docs_queue[process_id], ars[process_id], archive_name),
        )
        w_procs.append(p)

    # run read and write processes in parallel
    prs = r_procs + w_procs

    for p in prs:
        p.start()

    for p in prs:
        p.join()

    print("Pass 1 finished...")


def pass_2_shuffle_holdout(
    input_dir_path,
    output_dir_path,
    output_holdout_dir_path,
    start_index,
    end_index,
    chunk_id,
):
    # both eval and test set contain 0.17% of the data.
    holdout_ratio = 0.0017

    # We shuffle each pile of documents in memory here
    random.seed(42)

    print("Pass 2 started, going through pile documents...")

    chunks = os.listdir(input_dir_path)[start_index:end_index]
    print(chunks)

    start_time = time.time()
    for chunk in tqdm(chunks, total=len(chunks)):
        print(f"Started processing chunk {chunk_id} in pass 2...")

        reader = lmd.Reader(f"{input_dir_path}/{chunk}")
        lines = []
        for doc_id, doc in enumerate(reader._stream_data(get_meta=True)):
            text, meta = doc
            lines.append((text, meta))
            if doc_id % 10000 == 0:
                print(
                    f"Processed doc {doc_id} after {time.time() - start_time}"
                )

        # shuffling each output file.
        random.shuffle(lines)

        # selecting a subset for holdout
        pivot = int(len(lines) * holdout_ratio)

        n = len(os.listdir(f"{input_dir_path}/{chunk}"))
        buckets_train = list(
            chunked(range(pivot, len(lines)), (len(lines) - pivot) // n)
        )
        buckets_holdout = list(chunked(range(0, pivot), pivot // n))
        train_output_chunk = f"{output_dir_path}/chunk{chunk_id}"
        holdout_output_chunk = f"{output_holdout_dir_path}/chunk{chunk_id}"
        os.makedirs(output_dir_path, exist_ok=True)
        os.makedirs(output_holdout_dir_path, exist_ok=True)
        rm_if_exists(train_output_chunk)
        os.mkdir(train_output_chunk)
        rm_if_exists(holdout_output_chunk)
        os.mkdir(holdout_output_chunk)

        for j in range(n):
            output_file_name = (
                f"{train_output_chunk}/example_train_{j}.jsonl.zst"
            )
            output_holdout_file_name = (
                f"{holdout_output_chunk}/example_holdout_{j}.jsonl.zst"
            )

            with (
                open(output_file_name, "wb") as fout,
                open(output_holdout_file_name, "wb") as holdout_fout,
            ):
                # train output set
                write_lmd_dataset(fout, lines, buckets_train[j])

                # holdout output set
                write_lmd_dataset(holdout_fout, lines, buckets_holdout[j])

        for j in range(n, len(buckets_train)):
            output_file_name = (
                f"{train_output_chunk}/example_train_{j}.jsonl.zst"
            )
            with open(output_file_name, "wb") as fout:
                write_lmd_dataset(fout, lines, buckets_train[j])

        for j in range(n, len(buckets_holdout)):
            output_holdout_file_name = (
                f"{holdout_output_chunk}/example_holdout_{j}.jsonl.zst"
            )
            with open(output_holdout_file_name, "wb") as holdout_fout:
                write_lmd_dataset(holdout_fout, lines, buckets_holdout[j])

        del lines
        gc.collect()
        print("Pass 2 is finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="stage", required=True)
    pass1_parser = subparser.add_parser("pass1")
    pass1_parser.add_argument("--input_dir", type=str)
    pass1_parser.add_argument("--duplicates", type=str)
    pass1_parser.add_argument("--short_docs", type=str)
    pass1_parser.add_argument("--out_dir", type=str)

    pass2_parser = subparser.add_parser("pass2")
    pass2_parser.add_argument("start_index", type=int)
    pass2_parser.add_argument("end_index", type=int)
    pass2_parser.add_argument("chunk_id", type=int)
    pass2_parser.add_argument("--input_dir", type=str)
    pass2_parser.add_argument("--train_dir", type=str)
    pass2_parser.add_argument("--holdout_dir", type=str)

    args = parser.parse_args()

    if args.stage == "pass1":
        inputdir = args.input_dir
        if inputdir[-1] != '/':
            inputdir += '/'
        pass_1_shuffle(
            RedPajamaReplication(
                redpj_datasets(inputdir), args.duplicates, args.short_docs
            ),
            output_dir_path=args.out_dir,
        )
    elif args.stage == "pass2":
        pass_2_shuffle_holdout(
            input_dir_path=args.input_dir,
            output_dir_path=args.train_dir,
            output_holdout_dir_path=args.holdout_dir,
            start_index=args.start_index,
            end_index=args.end_index,
            chunk_id=args.chunk_id,
        )
    else:
        print("Please specify either pass1 or pass2")
