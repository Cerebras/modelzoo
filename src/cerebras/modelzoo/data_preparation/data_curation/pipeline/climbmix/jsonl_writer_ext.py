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

from datatrove.data import Document
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import StatHints


class JsonlWriterExt(JsonlWriter):
    """
    A JsonlWriter that supports writing a maximum number of documents per file.
    If the limit is reached, it switches to a new file.
    If `split_by_cluster` is enabled, it will create a subdirectory for each cluster
    and write the documents for that cluster into a separate file within that subdirectory.
    """

    default_output_filename: str = "${rank}.jsonl"
    name = "🐿 JsonlExt"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        max_docs_per_file: int = 0,
        split_by_cluster: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_docs_per_file = max_docs_per_file
        self.split_by_cluster = split_by_cluster
        self._docs_written = 0

        if (self.max_file_size == -1) != (self.max_docs_per_file == 0):
            raise ValueError(
                "Either max_file_size or max_docs_per_file must be set, but not both."
            )
        if self.split_by_cluster and self.max_docs_per_file != 0:
            raise ValueError(
                "split_by_cluster with max_docs_per_file is not supported."
            )

    def write(self, document: Document, rank: int = 0, **kwargs):
        if self.max_docs_per_file == 0 and not self.split_by_cluster:
            return super().write(document, rank, **kwargs)

        original_name = output_filename = self._get_output_filename(
            document, rank, **kwargs
        )

        if self.split_by_cluster:
            cluster_id = document.metadata.get("cluster_id", None)
            if cluster_id is None:
                raise ValueError(
                    "split_by_cluster is enabled but metadata has no cluster_id"
                )
            cluster_id = str(cluster_id)
            original_name = output_filename = os.path.join(
                cluster_id, output_filename
            )

        def get_rotated_filename(cur):
            self.file_id_counter[original_name] += 1
            new_output_filename = self._get_filename_with_file_id(original_name)
            if not self.split_by_cluster:
                self._on_file_switch(original_name, cur, new_output_filename)
            return new_output_filename

        if self.max_docs_per_file != 0:
            output_filename = self._get_filename_with_file_id(original_name)
            if self._docs_written >= self.max_docs_per_file:
                output_filename = get_rotated_filename(output_filename)
                self._docs_written = 0
            # This will not reflect the num docs per file if we are splitting by cluster
            self._docs_written += 1

        elif self.max_file_size > 0:
            # get size of current file
            output_filename = self._get_filename_with_file_id(original_name)
            # we have to switch file!
            if (
                self.output_mg.get_file(output_filename).tell()
                >= self.max_file_size
            ):
                output_filename = get_rotated_filename(output_filename)

        # actually write
        self._write(
            self.adapter(document),
            self.output_mg.get_file(output_filename),
            original_name,
        )
        self.stat_update(self._get_output_filename(document, "XXXXX", **kwargs))
        self.stat_update(StatHints.total)
        self.update_doc_stats(document)
