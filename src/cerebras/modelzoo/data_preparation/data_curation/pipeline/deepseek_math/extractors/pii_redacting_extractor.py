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

import re
from typing import Union

from datatrove.pipeline.extractors.base import BaseExtractor


class PIIRedactingExtractor(BaseExtractor):
    """
    Regex-based PII redaction extractor.

    Replaces:
        email@example.com → <EMAIL_ADDRESS>
        +1-555-123-4567   → <PHONE_NUMBER>
        192.168.0.1       → <IP_ADDRESS>
    """

    name = "PIIRedacting"

    def __init__(self, timeout: float = 10.0, **kwargs):
        super().__init__(timeout=timeout, **kwargs)

        self.EMAIL_REGEX = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        )
        self.PHONE_REGEX = re.compile(r'\b(?:\+?\d[\d\-\s\(\)]{7,}\d)\b')
        self.IP_REGEX = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

    def extract(self, text: Union[str, bytes, bytearray]) -> str:
        if not text.strip():
            self.stat_update("pii_empty_input")
            return ""

        redacted_text = text
        redacted_text = self.EMAIL_REGEX.sub("<EMAIL_ADDRESS>", redacted_text)
        redacted_text = self.PHONE_REGEX.sub("<PHONE_NUMBER>", redacted_text)
        redacted_text = self.IP_REGEX.sub("<IP_ADDRESS>", redacted_text)

        if redacted_text != text:
            self.stat_update("pii_redacted")
        else:
            self.stat_update("pii_unchanged")

        return redacted_text
