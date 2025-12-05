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

from urllib.parse import urlparse

from datatrove.data import Document
from datatrove.pipeline.readers.jsonl import JsonlReader


class MathDomainReader(JsonlReader):
    """
    Custom datatrove pipeline step to read URLs from JSONL.gz files
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_count = 0
        self.domains = set()

    def read_file(self, filepath):
        # Process each document from the reader
        for doc in super().read_file(filepath):
            try:
                # Extract URL and domain from metadata
                url = doc.metadata.get('url')
                domain_classification = doc.metadata.get(
                    'domain_classification'
                )
                accepted_reason = doc.metadata.get('acceptance_reason')
                domain = doc.metadata.get('domain')
                if not domain:
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()

                if accepted_reason in [
                    'basic_filter_keyword',
                    'math_environment',
                    "math_domain_with_content",
                ]:
                    self.domain_count += 1
                    self.domains.add(domain)

                    # Create a new document with the URL as content
                    domain_doc = Document(
                        text=domain,
                        id=f"domain_{self.domain_count}",
                        metadata={
                            'original_url': url,
                            'domain_classification': domain_classification,
                        },
                    )

                    yield domain_doc
                else:
                    yield Document(
                        text="", id=f"domain_{self.domain_count}", metadata={}
                    )
            except Exception as e:
                print(f"Error processing document {doc.id}: {e}")
