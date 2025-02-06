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
import pickle
import re
import string
from itertools import repeat
from multiprocessing import Pool, cpu_count

import jsonlines
from datasketch import MinHash
from lm_dataformat import Reader
from more_itertools import chunked
from nltk import ngrams
from tqdm import tqdm


def get_features(s, width):
    # lower cased
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return map(lambda x: "".join(x), ngrams(s, width))


def get_documents(input_dir, index_start, index_end, output_dir, dataset_name):
    gc.collect()
    files = sorted(os.listdir(input_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    for i, input_file in tqdm(enumerate(files[index_start:index_end])):
        file_path = f"{input_dir}/{input_file}"
        file_name = file_path.split("/")[-1]

        if dataset_name == "common_crawl":
            dir_2 = file_path.split("/")[-2]
            output_name = f"{dataset_name}/{dir_2}/{file_name}"
        else:
            output_name = f"{dataset_name}/{file_name}"

        if dataset_name == "common_crawl":
            reader = Reader(file_path)
            for doc_id, doc in enumerate(reader._stream_data(jsonl_key="text")):
                yield doc, file_path, doc_id
        else:
            with jsonlines.open(file_path) as rdr:
                for doc_id, doc in enumerate(rdr):
                    yield doc["text"], file_path, doc_id


def to_minhash(chunks):
    gc.collect()
    buckets = []
    documents, output_dir, width, dataset_name, n_docs = chunks
    for doc in tqdm(documents, total=n_docs):
        text, file_path, doc_id = doc[0], doc[1], doc[2]
        file_name = file_path.split("/")[-1]
        if dataset_name == "common_crawl":
            dir_2 = file_path.split("/")[-2]
            output_name = f"{dataset_name}/{dir_2}/{file_name}"
        else:
            output_name = f"{dataset_name}/{file_name}"

        m = MinHash(num_perm=128)
        [m.update(x.encode('utf8')) for x in get_features(text, width)]
        buckets.append(
            {
                "file_name": output_name,
                "doc_id": doc_id,
                "hash": m,
            }
        )
    return buckets


def output_results(output_dir, results, chunk_id, iter):
    with open(
        f"{output_dir}/minhash_nfc/{iter}-{chunk_id}.pickle", "wb"
    ) as fout:
        pickle.dump(results, fout)


def generate_hashes(args):
    if not os.path.exists(f"{args.output_dir}/minhash_nfc"):
        os.mkdir(f"{args.output_dir}/minhash_nfc")

    documents = get_documents(
        args.input_dir,
        args.index_start,
        args.index_end,
        args.output_dir,
        args.dataset_name,
    )
    results = []
    chunk_id = 0
    gc.collect()
    with Pool(processes=cpu_count()) as pool:
        for i, chunks in enumerate(
            tqdm(
                pool.imap(
                    to_minhash,
                    zip(
                        chunked(documents, args.n_docs // cpu_count()),
                        repeat(args.output_dir),
                        repeat(args.w),
                        repeat(args.dataset_name),
                        repeat(args.n_docs // cpu_count()),
                    ),
                ),
                total=cpu_count(),
            )
        ):

            for chunk in chunks:
                if len(results) == args.k:
                    output_results(
                        args.output_dir, results, chunk_id, args.iter
                    )
                    del results
                    gc.collect()
                    results = []
                    chunk_id += 1
                results.append(chunk)

    if results:
        output_results(args.output_dir, results, chunk_id, args.iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    parser.add_argument(
        "n_docs", type=int, help="Number of documents located in the dataset."
    )
    parser.add_argument("iter", help="Job id")
    parser.add_argument(
        "index_start",
        type=int,
        help="Start indexing documents from input directory after ls.",
    )
    parser.add_argument(
        "index_end",
        type=int,
        help="End indexing documents from input directory after ls.",
    )
    parser.add_argument(
        "-w", type=int, default=6, help="The window size", required=False
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10000,
        help="Number of batches to output with.",
        required=False,
    )
    args = parser.parse_args()
    generate_hashes(args)
