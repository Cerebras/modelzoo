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
import os
import time
from glob import glob
from multiprocessing import Process, Queue

from lm_dataformat import Archive, Reader
from more_itertools import divide


def generate_samples(files, queues, process_id):
    for fp in files:
        reader = Reader(fp)
        for doc, meta in reader._stream_data(get_meta=True, jsonl_key="text"):
            queues[meta["redpajama_set_name"]].put({"text": doc, "meta": meta})

    print(f"process {process_id} is done!")


def write_samples(q, dataset, out_dir):
    output_dir = os.path.join(out_dir, dataset.replace("RedPajama", ""))
    os.makedirs(output_dir, exist_ok=True)
    ar = Archive(output_dir, threads=10)
    i = 0
    start_time = time.time()
    while True:
        try:
            doc = q.get(timeout=30)
            ar.add_data(doc["text"], doc["meta"])
        except TypeError:
            assert doc == "Done!"
            ar.commit(archive_name="slimpajama" + str(ar.i))
            break
        i += 1
        if i % 100000 == 0:
            ar.commit(archive_name="slimpajama" + str(ar.i))
            print(
                f"Total number of processed documents: {i} ",
                f"Total time: {time.time() - start_time}",
            )
    print(f"Finished writing documents for {dataset}.")


def main(args):
    files = glob(os.path.join(args.input_dir, "**/*.jsonl.zst"), recursive=True)
    files = sorted(files)
    n_process = args.processes
    files = divide(n_process, files)
    chunks = [list(f) for f in files]

    datasets = [
        "RedPajamaCommonCrawl",
        "RedPajamaC4",
        "RedPajamaGithub",
        "RedPajamaBook",
        "RedPajamaArXiv",
        "RedPajamaWikipedia",
        "RedPajamaStackExchange",
    ]
    producers = []
    queues = {dataset: Queue(64 * 10000) for dataset in datasets}
    for process_id in range(n_process):
        p = Process(
            target=generate_samples,
            args=(
                chunks[process_id],
                queues,
                process_id,
            ),
        )
        producers.append(p)

    consumers = []
    for dataset in datasets:
        p = Process(
            target=write_samples,
            args=(
                queues[dataset],
                dataset,
                args.output_dir,
            ),
        )
        consumers.append(p)

    for p in producers:
        p.start()
    for p in consumers:
        p.start()

    for p in producers:
        p.join()
    for q in queues.values():
        q.put("Done!")
    for p in consumers:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--processes", type=int)
    args = parser.parse_args()
    main(args)
