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

from datatrove.data import Document
from datatrove.pipeline.writers.disk_base import DiskWriter


class TextWriter(DiskWriter):
    """
    A writer for text files.
    Each document's text is written to a separate line in the output file.
    """

    default_output_filename: str = "${rank}.txt"
    name = "Text"

    def _default_adapter(self, document: Document) -> str:
        """
        Convert a document to a text string.
        Only writes the document's text content.
        """
        return document.text + "\n"
