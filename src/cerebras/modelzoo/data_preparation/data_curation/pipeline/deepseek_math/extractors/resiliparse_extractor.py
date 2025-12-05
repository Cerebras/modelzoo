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
from datatrove.data import DocumentsPipeline
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
    5) Store original HTML in document metadata
    """

    name = "Resiliparse"

    def __init__(self, timeout: float = 30.0, **kwargs):
        # BaseExtractor.__init__ only takes timeout
        super().__init__(timeout=timeout)

    def extract(self, processed_html: str) -> str:
        """Extract plain text from already processed HTML content.

        Args:
            processed_html: Already processed HTML content

        Returns:
            Extracted plain text
        """
        try:
            # Extract plain text from the processed HTML
            text = extract_plain_text(
                processed_html,
                alt_texts=False,
                links=False,
                preserve_formatting=True,
            )

            # Record stats and return text
            if not text.strip():
                self.stat_update("doc_extraction_empty")
                return ""  # returning empty string drops the doc
            else:
                self.stat_update("doc_extraction_success")
                return text

        except Exception:
            # any top‑level failure drops the doc
            self.stat_update("doc_extraction_failed")
            return ""

    def process_html(self, html_content: Union[bytes, bytearray, str]) -> str:
        """Process HTML content (LaTeX improvement, table processing) in the main process.

        Args:
            html_content: Raw HTML content

        Returns:
            Processed HTML content
        """
        # Decode if needed
        if isinstance(html_content, (bytes, bytearray)):
            html_content = bytes_to_str(
                html_content, detect_encoding(html_content)
            )

        try:
            # 1) Improve LaTeX snippets
            processed_html = improve_latex_content_parsing(html_content)

            # 2) Process tables (in‑place transform of the tree)
            try:
                tree = html.fromstring(processed_html)
                tree = process_tables(tree)  # or pass format_choice if needed
                processed_html = html.tostring(
                    tree, encoding="unicode", pretty_print=True
                )
            except Exception:
                # table conversion failed—keep going with the unmodified HTML
                processed_html = html_content

            return processed_html

        except Exception:
            # Return original HTML if processing fails
            return html_content

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """Override run method to preserve HTML content in metadata"""
        from datatrove.pipeline.extractors.base import ExtractorSandbox
        from datatrove.utils.logging import logger
        from datatrove.utils.typeshelper import StatHints

        with ExtractorSandbox(timeout=self.timeout) as extractor:
            for doc in data:
                self.stat_update(StatHints.total)

                # Process HTML in the main process to preserve it
                original_html = doc.text
                processed_html = self.process_html(original_html)

                with self.track_time():
                    try:
                        # Extract text using the processed HTML
                        doc.text = extractor.process_document(
                            processed_html, self.extract
                        )

                        # Add processed HTML content to metadata for later pipeline steps
                        # Only if extraction was successful
                        if doc.text:
                            doc.metadata["html_content"] = processed_html

                        self.stat_update("extracted")
                    except TimeoutError:
                        self.stat_update("timeout")
                        logger.warning(
                            "Timeout while cleaning record text. Skipping record."
                        )
                        continue
                    except EOFError:
                        # Process died unexpectedly
                        self.stat_update("broken_process")
                        logger.warning(
                            "Process died unexpectedly, will create new process for next document"
                        )
                        continue
                    except Exception as e:
                        self.stat_update("clean_error")
                        if not self._warned_error:
                            logger.warning(
                                f'Error "{e}" while cleaning record text. Skipping record. '
                                f"This message will only appear once."
                            )
                            self._warned_error = True
                        continue

                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
