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
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.logging import logger


class IDDedupFilter(BaseFilter):
    """Filter that checks Python code for syntax errors."""

    name = "🐍 ID Dedup Filter"

    def __init__(self, text_ids_path, **kwargs):  # Accept kwargs
        super().__init__(**kwargs)  # Pass kwargs to parent
        self.text_ids_path = text_ids_path

    def filter(self, doc: Document) -> bool:
        """Returns True if the document passes the filter (no syntax errors)."""
        if not os.path.exists(self.text_ids_path):
            open(self.text_ids_path, 'w').close()
        with open(self.text_ids_path, 'r', encoding='utf-8') as f:
            text_ids = set(line.strip() for line in f)
        # logger.info(f"Loaded {len(text_ids)} existing IDs from {self.text_ids_path}. Document ID: {doc.id}")
        if doc.id in text_ids:
            logger.info(f"Document {doc.id} is a duplicate.")
            self.stat_update("documents_duplicate_id")
            return False
        else:
            with open(self.text_ids_path, 'a', encoding='utf-8') as f:
                f.write(f"{doc.id}\n")
            self.stat_update("documents_filtered")
            return True
