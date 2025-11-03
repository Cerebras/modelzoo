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
from typing import List

from datatrove.filters import BaseFilter
from datatrove.typing import Document
from transformers import AutoTokenizer


def split_paragraphs(
    text: str, paragraph_end='\n', remove_empty: bool = True
) -> List[str]:
    r"""
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters,
    followed by a newline character(s), or a sequence of one or more characters, followed by the end
    of the string.

    Args:
        text (str): The text to split.
        paragraph_end (str): The sequence of character that marks a paragraph end. default: '\n', but
                             can be '\n\n' for example.
        remove_empty (bool): Whether to remove empty paragraphs.

    Returns:
        List[str]: A list of strings where each string represents a paragraph from the original text.
    """
    paragraphs = re.split(paragraph_end, text)
    if remove_empty is True:
        paragraphs = [p for par in paragraphs if (p := par.strip())]
    return paragraphs


class AlphanumericCharRatioFilter(BaseFilter):
    name = "AlphanumericCharRatioFilter"

    def __init__(self, max_alnum_ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.max_alnum_ratio = max_alnum_ratio

    def filter(self, doc: Document) -> bool:
        text = doc.text
        if not text:
            return False

        alnum_count = sum(c.isalnum() for c in text)
        alnum_ratio = alnum_count / len(text)
        return alnum_ratio >= self.max_alnum_ratio


class AlphabeticCharactersToTokensFilter(BaseFilter):
    name = "AlphabeticCharactersToTokensFilter"

    def __init__(
        self,
        tokenizer_name: str = "EleutherAI/pythia-6.9b-deduped",
        max_ratio: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_ratio = max_ratio

    def filter(self, doc: Document) -> bool:
        text = doc.text
        if not text:
            return False

        num_token = len(self.tokenizer.tokenize(text))
        if num_token == 0:
            return False

        num_alpha = sum(1 for c in text if c.isalpha())
        alpha_ratio = num_alpha / num_token
        return alpha_ratio >= self.max_ratio


class BulletCountFilter(BaseFilter):
    name = "BulletCountFilter"

    def __init__(self, max_bullet_start_ratio: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.max_bullet_start_ratio = max_bullet_start_ratio
        self.bullets = ['●', '•', '*', '-']

    def filter(self, doc: Document) -> bool:
        lines = split_paragraphs(doc.text, paragraph_end='\n')
        if not lines:
            return False
        max_bullet_count = self.max_bullet_start_ratio * len(lines)
        bullet_lines = sum(
            any(line.startswith(bullet) for bullet in self.bullets)
            for line in lines
        )
        return bullet_lines <= max_bullet_count


class EllipsisCountFilter(BaseFilter):
    name = "EllipsisCountFilter"

    def __init__(self, max_ellipsis_end_ratio: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.max_ellipsis_end_ratio = max_ellipsis_end_ratio
        self.ellipsis_tokens = ['...', '. . .', '\u2026']

    def filter(self, doc: Document) -> bool:
        lines = split_paragraphs(doc.text, paragraph_end='\n')
        if not lines:
            return False
        max_ellipsis_count = self.max_ellipsis_end_ratio * len(lines)
        ellipsis_lines = sum(
            any(line.endswith(ell) for ell in self.ellipsis_tokens)
            for line in lines
        )
        return ellipsis_lines <= max_ellipsis_count


class StopWordFilter(BaseFilter):
    name = "StopWordFilter"

    def __init__(
        self, min_stop_word: int = 2, count_unique: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_stop_word = min_stop_word
        self.count_unique = count_unique
        self.stop_words = {
            'the',
            'be',
            'to',
            'of',
            'and',
            'that',
            'have',
            'with',
        }

    def filter(self, doc: Document) -> bool:
        words = doc.text.split()
        stop_count = 0
        seen = set()
        for word in words:
            w = word.lower()
            if w in self.stop_words:
                if self.count_unique:
                    seen.add(w)
                    if len(seen) >= self.min_stop_word:
                        return True
                else:
                    stop_count += 1
                    if stop_count >= self.min_stop_word:
                        return True
        return False


class WordLengthFilter(BaseFilter):
    name = "WordLengthFilter"

    def __init__(
        self, min_length: int = 0, max_length: int = float('inf'), **kwargs
    ):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def filter(self, doc: Document) -> bool:
        words = doc.text.split()
        if not words:
            return False
        avg_word_length = sum(len(w) for w in words) / len(words)
        return self.min_length <= avg_word_length <= self.max_length


class SymbolRatioFilter(BaseFilter):
    name = "SymbolRatioFilter"

    def __init__(self, max_symbol_to_word_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.max_ratio = max_symbol_to_word_ratio
        self.symbols = ["#", "...", ". . .", "\u2026"]

    def filter(self, doc: Document) -> bool:
        text = doc.text
        num_symbols = sum(text.count(sym) for sym in self.symbols)
        num_words = len(text.split())
        if num_words == 0:
            return False
        return num_symbols / num_words <= self.max_ratio


class AlphabeticWordRatioFilter(BaseFilter):
    name = "AlphabeticWordRatioFilter"

    def __init__(self, max_ratio: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.max_ratio = max_ratio

    def filter(self, doc: Document) -> bool:
        words = doc.text.split()
        total_words = len(words)
        if total_words == 0:
            return False
        non_alpha_words = sum(
            1 for word in words if not any(c.isalpha() for c in word)
        )
        return (non_alpha_words / total_words) <= self.max_ratio
