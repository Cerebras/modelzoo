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

from typing import Union

from data_curation.pipeline.deepseek_math.utils.latex_parsing import (
    improve_latex_content_parsing,
)
from data_curation.pipeline.deepseek_math.utils.table_parsing import (
    process_tables,
)
from datatrove.pipeline.extractors.base import BaseExtractor
from lxml import html
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str, detect_encoding


class ResiliparseExtractor(BaseExtractor):
    """
    1) Decode raw HTML bytes (or accept str as‑is)
    2) Improve LaTeX rendering
    3) Convert tables to markdown/pre blocks
    4) Extract plain text (preserving formatting)
    """

    name = "Resiliparse"

    def __init__(self, timeout: float = 30.0, **kwargs):
        # BaseExtractor.__init__ only takes timeout
        super().__init__(timeout=timeout, **kwargs)

    def extract(self, html_content: Union[bytes, bytearray, str]) -> str:
        # Decode if needed
        if isinstance(html_content, (bytes, bytearray)):
            html_content = bytes_to_str(
                html_content, detect_encoding(html_content)
            )

        try:
            # 1) Improve LaTeX snippets
            html_content = improve_latex_content_parsing(html_content)

            # 2) Process tables (in‑place transform of the tree)
            try:
                tree = html.fromstring(html_content)
                tree = process_tables(tree)  # or pass format_choice if needed
                html_content = html.tostring(
                    tree, encoding="unicode", pretty_print=True
                )
            except Exception:
                # table conversion failed—keep going with the unmodified HTML
                pass

            # 3) Extract plain text
            text = extract_plain_text(
                html_content,
                alt_texts=False,
                links=False,
                preserve_formatting=True,
            )

            # 4) Record stats
            if not text.strip():
                self.stat_update("doc_extraction_empty")
                # returning empty string drops the doc
                return ""
            else:
                self.stat_update("doc_extraction_success")

                return text

        except Exception:
            # any top‑level failure drops the doc
            self.stat_update("doc_extraction_failed")
            return ""
