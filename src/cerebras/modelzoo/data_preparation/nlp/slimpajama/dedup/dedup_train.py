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
from glob import glob

from lm_dataformat import Reader
from tqdm import tqdm

# isort: off
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
# isort: on
from cerebras.modelzoo.data_preparation.nlp.slimpajama.utils import (
    rm_if_exists,
    sha256str,
    write_lmd_dataset,
)


def deduplicate_train_holdout_sets(
    train_path, holdout_path, deduped_train_path, chunk_id
):
    # Calculate hashes on holdout set.
    seen = set()
    if os.path.exists("hashes.txt"):
        with open("hashes.txt") as fh:
            for line in tqdm(fh):
                seen.add(line.strip())
    else:
        hashf = open("hashes.txt", "w")
        for f in tqdm(glob(f"{holdout_path}/*/*.zst")):
            reader = Reader(f)
            for doc_id, text in enumerate(
                reader._stream_data(jsonl_key="text")
            ):
                hash = sha256str(text)
                hashf.write(hash + "\n")
                seen.add(hash)
        hashf.close()

    print("Finished collecting hashes for eval", len(seen))

    rm_if_exists(f"{deduped_train_path}/chunk{chunk_id}")
    os.makedirs(f"{deduped_train_path}/chunk{chunk_id}")

    total_written = 0
    # Remove elements from train set with hashes seen in eval set.
    for f in tqdm(glob(f"{train_path}/chunk{chunk_id}/*.zst")):

        def filtered_docs():
            reader = Reader(f)
            for doc_id, doc in enumerate(reader._stream_data(get_meta=True)):
                text, meta = doc
                hash = sha256str(text)
                if hash not in seen:
                    yield text, meta
                else:
                    print("Found an intersection!!!")

        with open(
            f"{deduped_train_path}/chunk{chunk_id}/" + f.split("/")[-1], "wb"
        ) as fout_dedup_train:
            total_written += write_lmd_dataset(
                fout_dedup_train,
                filtered_docs(),
                indices=None,
                return_total_written=True,
            )

    print(f"Total written: {total_written}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_id", type=int)
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--tgt_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()
    deduplicate_train_holdout_sets(
        args.src_dir,
        args.tgt_dir,
        args.out_dir,
        args.chunk_id,
    )
