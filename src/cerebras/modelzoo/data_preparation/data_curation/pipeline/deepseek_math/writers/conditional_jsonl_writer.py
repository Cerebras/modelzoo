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

from __future__ import annotations

import os
from typing import Dict, Iterator

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.jsonl import JsonlWriter


class ConditionalJsonlWriter(PipelineStep):
    """
    A writer that routes Document objects to different subfolders based on metadata conditions.
    """

    name = "ConditionalJsonlWriter"

    def __init__(
        self,
        base_output_folder: str,
        filename_template: str = "${cc_path}.jsonl.gz",
    ):
        super().__init__()
        self.base_output_folder = base_output_folder
        self.filename_template = filename_template
        self._writers: Dict[str, JsonlWriter] = {}
        self._closed = False

    def _get_writer(self, subfolder: str) -> JsonlWriter:
        """Return a JsonlWriter for the given subfolder, creating if needed."""
        if subfolder not in self._writers:
            output_folder = os.path.join(self.base_output_folder, subfolder)
            os.makedirs(output_folder, exist_ok=True)
            writer = JsonlWriter(
                output_folder=output_folder,
                output_filename=self.filename_template,
            )
            self._writers[subfolder] = writer
        return self._writers[subfolder]

    def run(
        self, docs: Iterator[Document], rank: int, world_size: int
    ) -> Iterator[Document]:
        """
        Route documents to appropriate writers and yield downstream.
        """
        docs_processed = 0
        try:
            for doc in docs:
                meta = doc.metadata or {}
                latex_flag = meta.get("contains_latex_symbols", 0) == 1
                math_flag = meta.get("math_fasttext", 0) == 1

                if latex_flag and math_flag:
                    sub = "true_positives"
                elif latex_flag and not math_flag:
                    sub = "false_negatives"
                elif not latex_flag and math_flag:
                    sub = "false_positives"
                else:
                    sub = "true_negatives"

                writer = self._get_writer(sub)
                writer.write(doc, rank)

                docs_processed += 1
                yield doc

        finally:
            if docs_processed > 0:
                print(
                    f"ConditionalJsonlWriter processed {docs_processed} documents, closing..."
                )
                self.close()

    def close(self) -> None:
        """Close all underlying writers properly."""
        if self._closed:
            return

        if self._writers:
            for subfolder, writer in self._writers.items():
                try:
                    writer.close()
                    print(f"  ✓ Closed {subfolder}")
                except Exception as e:
                    print(f"  Error closing {subfolder}: {e}")

            self._writers.clear()

        self._closed = True
