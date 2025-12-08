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

import json
import os

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.logging import logger


class PyPIPackageFilter(BaseFilter):
    """Filter that checks if the pypi repo has a valid github homepage."""

    name = "🐍 PyPI Package Filter"

    def __init__(self, **kwargs):  # Accept kwargs
        super().__init__(**kwargs)  # Pass kwargs to parent

    def filter(self, doc: Document) -> bool:
        """Returns True if the document passes the filter (has a github homepage)."""
        metadata_file = os.path.join(doc.text, "metadata.json")
        if not os.path.exists(metadata_file):
            logger.info(f"Document {doc.text} is missing metadata.json.")
            self.stat_update("documents_missing_metadata")
            return False
        ## Read the metadata file to get description and project name
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if metadata and metadata.get("info", {}).get("description", "") == "":
            logger.info(f"Document {doc.text} has no description in metadata.")
            self.stat_update("documents_missing_description")
            return False
        if not metadata.get("info", {}).get("project_urls", {}):
            logger.info(f"Document {doc.text} has no homepage URL in metadata.")
            self.stat_update("documents_missing_homepage")
            return False
        self.stat_update("documents_filtered_passed")
        return True
