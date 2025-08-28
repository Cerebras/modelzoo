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
        # cache JsonlWriter instances by subfolder
        self._writers: Dict[str, JsonlWriter] = {}

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
        Iterate incoming documents, route each one to the appropriate writer,
        and yield it downstream unchanged.
        """
        for idx, doc in enumerate(docs):

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
            yield doc

    def close(self) -> None:
        """Close all underlying writers."""
        for writer in self._writers.values():
            writer.close()
