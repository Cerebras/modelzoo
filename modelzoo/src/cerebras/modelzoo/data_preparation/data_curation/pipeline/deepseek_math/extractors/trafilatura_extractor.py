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

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.extractors.base import BaseExtractor
from datatrove.pipeline.extractors.trafilatura import Trafilatura


class TrafilaturaExtractor(BaseExtractor):
    """
    An extractor that:
    1. Takes HTML content from document metadata
    2. Uses Trafilatura to extract text from that HTML
    3. Makes the extracted text the main text of the document
    """

    name = "TrafilaturaExtractor"

    def __init__(self, timeout: float = 30.0, **kwargs):

        super().__init__(timeout=timeout)
        self._warned_error = False
        self.trafilatura = Trafilatura(**kwargs)

    def extract(self, html_content: str) -> str:
        """
        Extract text from HTML content using Trafilatura.

        Args:
            html_content: The HTML content to extract text from

        Returns:
            Extracted plain text
        """
        # Use Trafilatura to extract text from the HTML content
        extracted_text = self.trafilatura.extract(html_content)

        if not extracted_text:
            self.stat_update("trafilatura_extraction_failed")
            return ""
        else:
            self.stat_update("html_content_extracted")
            return extracted_text

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """Override run method to extract HTML from metadata instead of doc.text"""
        from datatrove.pipeline.extractors.base import ExtractorSandbox
        from datatrove.utils.logging import logger
        from datatrove.utils.typeshelper import StatHints

        with ExtractorSandbox(timeout=self.timeout) as extractor:
            for doc in data:
                self.stat_update(StatHints.total)

                # Check if HTML content exists in metadata
                if not doc.metadata or "html_content" not in doc.metadata:
                    self.stat_update("no_html_content_in_metadata")
                    logger.warning(
                        "No html_content found in document metadata. Skipping document."
                    )
                    continue

                html_content = doc.metadata.pop("html_content")
                if not html_content:
                    self.stat_update("empty_html_content")
                    logger.warning(
                        "Empty html_content in document metadata. Skipping document."
                    )
                    continue

                with self.track_time():
                    try:
                        # Extract text from HTML content in metadata
                        extracted_text = extractor.process_document(
                            html_content, self.extract
                        )

                        if extracted_text:
                            # Replace document text with extracted text
                            doc.text = extracted_text
                            self.stat_update("extracted")
                        else:
                            self.stat_update("extraction_empty")
                            continue

                    except TimeoutError:
                        self.stat_update("timeout")
                        logger.warning(
                            "Timeout while extracting text from HTML. Skipping record."
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
                        self.stat_update("extraction_error")
                        if not self._warned_error:
                            logger.warning(
                                f'Error "{e}" while extracting text from HTML. Skipping record. '
                                f"This message will only appear once."
                            )
                            self._warned_error = True
                        continue

                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
