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
Script that normalizes text
"""

import argparse
import io
import json
from itertools import repeat
from math import ceil
from multiprocessing import Pool, cpu_count
from os import listdir, makedirs, path

import ftfy
import jsonlines
import zstandard
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing data files.",
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        type=str,
        help="Path to directory where normlaized data files will be stored.",
    )
    parser.add_argument(
        "--zst",
        action="store_true",
        help="files with zst compression",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=-1,
        help="index for spltting files in a directory for multiple parallel runs",
    )

    return parser.parse_args()


def recreate_dataset(params):
    files, args, process_no = params
    pbar = tqdm(
        desc=f"Parsed 0 input files. Files written ",
        disable=False,
    )
    for _file in files:
        file_path = path.join(args.data_dir, _file)
        target_path = path.join(args.target_dir, _file)
        if args.zst:
            with open(file_path, 'rb') as fh:
                dcctx = zstandard.ZstdDecompressor()
                reader = io.BufferedReader(dcctx.stream_reader(fh))
                rdr = jsonlines.Reader(reader)
                with open(target_path, "wb") as f:
                    cctx = zstandard.ZstdCompressor()
                    wrt = cctx.stream_writer(f)
                    writer = io.BufferedWriter(wrt)
                    for ob in rdr:
                        doc = ob["text"]
                        doc = ftfy.fix_text(doc, normalization="NFC")
                        record = {
                            "text": doc,
                            "pred_label": ob["pred_label"],
                            "pred_label_prob": ob["pred_label_prob"],
                            "wiki_prob": ob["wiki_prob"],
                            "source": ob["source"],
                        }
                        s = json.dumps(record) + "\n"
                        writer.write(s.encode("utf-8"))
                    writer.flush()
                    wrt.flush(zstandard.FLUSH_FRAME)
        else:
            with jsonlines.open(file_path) as rdr:
                with open(target_path, "w") as f:
                    for ob in rdr:
                        doc = ob["text"]
                        doc = ftfy.fix_text(doc, normalization="NFC")
                        record = {"text": doc, "meta": ob["meta"]}
                        f.write(json.dumps(record) + "\n")

    return True


def normalize_text(args):
    makedirs(args.target_dir, exist_ok=True)
    files = sorted(listdir(args.data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))

    if args.idx != -1:
        files = files[args.idx * 64 : (args.idx + 1) * 64]

    n_proc = cpu_count()
    n_chunks = ceil(len(files) / n_proc)
    remain = len(files) % n_proc
    if n_chunks == 1 and remain:
        n_proc = remain
    print(f"resetting to {n_proc} for number of processes")
    files = [files[i : i + n_chunks] for i in range(0, len(files), n_chunks)]

    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(
                recreate_dataset,
                zip(
                    files,
                    repeat(args),
                    range(len(files)),
                ),
            ),
            total=len(files),
        )
        for test in pbar:
            pbar.update()
            if test:
                continue


if __name__ == "__main__":
    args = parse_args()
    normalize_text(args)
