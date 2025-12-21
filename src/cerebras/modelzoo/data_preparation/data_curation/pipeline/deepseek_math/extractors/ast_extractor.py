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

import ast
from typing import Generator, Iterable

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


class ASTExtractor(PipelineStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Generator[Document, None, None]:
        """Process documents containing PyPI project paths."""
        for doc in data:
            doc.metadata["python_ast_files"] = []
            doc.metadata["python_code"] = doc.text
            doc.text = ""
            with self.track_time():
                python_code_files = doc.metadata.get("python_code", [])
                for python_code_file in python_code_files:
                    try:
                        source_code = python_code_file.get("content", "")
                        filename = python_code_file.get("filename", "")
                        tree = ast.parse(source_code)
                        ast_string = ast.dump(tree, indent=2)
                        doc.metadata["python_ast_files"].append(
                            {"filename": filename, "ast": ast_string}
                        )
                        self.stat_update("python_src_ast_extracted")
                    except SyntaxError as e:
                        logger.error(
                            f"Unable to get ast of file - {filename}. Error - {e}"
                        )
                        self.stat_update("python_src_ast_extraction_failed")

                python_test_files = doc.metadata.get("python_test_files", [])
                for python_test_file in python_test_files:
                    try:
                        source_code = python_test_file.get("content", "")
                        filename = python_test_file.get("filename", "")
                        tree = ast.parse(source_code)
                        ast_string = ast.dump(tree, indent=2)
                        if not doc.metadata.get("python_test_ast_files"):
                            doc.metadata["python_test_ast_files"] = []
                        doc.metadata["python_test_ast_files"].append(
                            {"filename": filename, "ast": ast_string}
                        )
                        self.stat_update("python_test_ast_extracted")
                    except SyntaxError as e:
                        logger.error(
                            f"Unable to get ast of file - {filename}. Error - {e}"
                        )
                        self.stat_update("python_test_ast_extraction_failed")
            yield doc
