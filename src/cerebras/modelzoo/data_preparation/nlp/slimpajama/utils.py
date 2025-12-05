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

import hashlib
import os
import shutil

import ujson as json
import zstandard


def utf8len(s):
    return len(s.encode('utf-8'))


def cycle_documents(dataset, process_id, n_process, dup_sh, short_sh):
    while True:
        # https://github.com/EleutherAI/the-pile/blob/df97f8651ae3da658b19659b3ceaa6a34b0fc014/the_pile/utils.py#L104
        yield from filter(
            lambda x: x,
            dataset.documents(process_id, n_process, dup_sh, short_sh),
        )


def sha256str(s):
    h = hashlib.sha256()
    try:
        h.update(s.encode("utf-8"))
    except UnicodeEncodeError:
        # to avoid things like \ud809\udc50\ud808\udefc\ud808\udedb
        h.update(s.encode("utf-8", "replace"))
    return h.hexdigest()


def rm_if_exists(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except NotADirectoryError:
        os.remove(path)


def write_lmd_dataset(fh, lines, indices=None, return_total_written=False):
    cctx = zstandard.ZstdCompressor(level=3, threads=10)
    compressor = cctx.stream_writer(fh)
    # to not store large lists into memory, use index
    total_written = 0
    if indices is not None:
        for index in indices:
            text, meta = lines[index]
            compressor.write(
                json.dumps({"text": text, "meta": meta}).encode("UTF-8") + b"\n"
            )
            total_written += 1
    else:
        for line in lines:
            text, meta = line
            compressor.write(
                json.dumps({"text": text, "meta": meta}).encode("UTF-8") + b"\n"
            )
            total_written += 1

    compressor.flush(zstandard.FLUSH_FRAME)
    if return_total_written:
        return total_written
