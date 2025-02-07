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
import pickle
import queue
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Process, Queue

from datasketch.lean_minhash import LeanMinHash
from more_itertools import divide


def _H(hs):
    return bytes(hs.byteswap().data)


def split_files(input_dir, n_proc):
    files = []
    for dataset in [
        "arxiv",
        "stackexchange",
        "book",
        "wikipedia",
        "github",
        "c4",
        "common_crawl",
    ]:
        if dataset == "common_crawl":
            files.extend(glob(f"{input_dir}/{dataset}/*/minhash_nfc/*"))
        else:
            files.extend(glob(f"{input_dir}/{dataset}/minhash_nfc/*"))
    files = sorted(files)
    parts = divide(n_proc, files)
    return [list(p) for p in parts]


def get_hashes(files, doc_queues, r):
    for fp in files:
        with open(fp, "rb") as fin:
            for item in pickle.load(fin):
                key = f"{item['file_name']}@{item['doc_id']}"
                minhash = LeanMinHash(item["hash"])
                for i, doc_queue in enumerate(doc_queues):
                    H = _H(minhash.hashvalues[i * r : (i + 1) * r])
                    doc_queue.put((key, H))


def lsh(out_file, doc_queue, idx):
    lsh_dict = defaultdict(str)
    i = 0
    start_time = time.time()
    f = open(out_file.replace(".txt", f"-{idx}.txt"), "w")
    while True:
        try:
            key, H = doc_queue.get(timeout=30)
            cand = lsh_dict.get(H, "None")
            if cand != "None":
                f.write(f'{key} :: {cand}\n')
            else:
                lsh_dict[H] = key

            if i % 100000 == 0:
                print(
                    f"{idx}: Processed {i / 931361530 * 100}%.",
                    time.time() - start_time,
                )
            i += 1
        except queue.Empty:
            break

    print(f"Total number of documents: {i}")
    f.close()


def generate_pairs(args):
    # size of the queue was tuned for optimal perf and memory constraints.
    doc_queues = [Queue(1000000) for _ in range(args.bands)]
    files = split_files(args.input_dir, args.processes)

    processes = []
    for process_id in range(args.processes):
        p = Process(
            target=get_hashes,
            args=(
                files[process_id],
                doc_queues,
                args.range,
            ),
        )
        processes.append(p)
        p.start()

    for process_id in range(args.bands):
        p = Process(
            target=lsh,
            args=(
                args.out_file,
                doc_queues[process_id],
                process_id,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--out_file")
    parser.add_argument(
        "--range",
        type=int,
    )
    parser.add_argument(
        "--bands",
        type=int,
    )
    parser.add_argument(
        "--processes",
        type=int,
    )
    args = parser.parse_args()

    generate_pairs(args)
