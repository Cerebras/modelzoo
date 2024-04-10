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

import os

import numpy as np


class DPREmbeddingSaver:
    def __init__(self, dir, batches_per_file) -> None:
        self.dir = dir
        self.batches_per_file = batches_per_file
        self.embed_writer_buffer = []
        self.id_writer_buffer = []
        self.written_files = 0

        os.makedirs(self.dir)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()

    def add_embeddings(self, embeddings, ids):
        self.embed_writer_buffer.append(embeddings)
        self.id_writer_buffer.append(ids)

        if len(self.embed_writer_buffer) >= self.batches_per_file:
            self.flush()

    def flush(self):
        if len(self.embed_writer_buffer) > 0:
            file = os.path.join(
                self.dir,
                f"embeddings-{str(self.written_files).zfill(4)}",
            )

            save_data = {
                "ids": np.concatenate(self.id_writer_buffer, axis=0).flatten(),
                "embds": np.concatenate(self.embed_writer_buffer, axis=0),
            }
            np.savez(file, **save_data)

            self.written_files += 1
            self.embed_writer_buffer = []
            self.id_writer_buffer = []
