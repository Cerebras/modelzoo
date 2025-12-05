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
This script is used for duplicate pairs generation.

It includes some functions from the datasketch library for calculation of 
range and bands - namely, _false_positive_probability, _false_negative_probability 
and optimal_param. The original source code can be found at:
https://github.com/ekzhu/datasketch/blob/master/datasketch/lsh.py#L24
"""

import argparse
import logging
import pickle
import queue
import sys
import threading
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Process, Queue, cpu_count

from datasketch.lean_minhash import LeanMinHash
from more_itertools import divide
from scipy.integrate import quad as integrate


def custom_progress_bar(length=30, animation_delay=0.5):
    chars = ['|', '/', '-', '\\']
    progress = 0

    while True:
        sys.stdout.write(f'\rProcessing: [{chars[progress % len(chars)]}]')
        sys.stdout.flush()
        progress += 1
        time.sleep(animation_delay)


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def optimal_param(
    threshold, num_perm, false_positive_weight, false_negative_weight
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def _H(hs):
    return bytes(hs.byteswap().data)


def split_files(input_dir, n_proc):
    files = []
    files.extend(glob(f"{input_dir}/minhash_nfc/*"))
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
            i += 1
        except queue.Empty:
            break

    logging.info(f"Total number of documents: {i}")
    f.close()


def generate_pairs(args):
    progress_thread = threading.Thread(target=custom_progress_bar)
    progress_thread.daemon = True
    progress_thread.start()

    # Generating range and bands using threshold value
    num_perm = 128
    false_positive_weight = 0.5
    false_negative_weight = 0.5
    b, r = optimal_param(
        args.jaccard_threshold,
        num_perm,
        false_positive_weight,
        false_negative_weight,
    )
    # size of the queue was tuned for optimal perf and memory constraints.
    doc_queues = [Queue(1000000) for _ in range(b)]
    files = split_files(args.input_dir, args.processes)

    processes = []
    for process_id in range(args.processes):
        p = Process(
            target=get_hashes,
            args=(
                files[process_id],
                doc_queues,
                r,
            ),
        )
        processes.append(p)
        p.start()

    for process_id in range(b):
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

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory, which contains documents.",
        required=True,
    )
    parser.add_argument(
        "--out_file",
        type=str,
        help="Output file, where the list of duplicate pairs will be stored.",
        required=True,
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        help="Threshold for Jaccard similarity, above which documents will be considered 'similar'. By default, this is set to 0.8.",
        default=0.8,
        required=False,
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="Number of processes to parallelise on. By default, this is set to the number of cores in the machine.",
        default=cpu_count(),
        required=False,
    )
    args = parser.parse_args()

    generate_pairs(args)
