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

# This code is adapted from
# https://github.com/google-research/bert/https://github.com/google-research/bert/blob/master/tokenization.py
#
# coding=utf-8
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 The Google AI Language Team Authors.
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

"""
Tokenization classes and functions
"""

import unicodedata

import numpy as np
from keras_preprocessing.text import Tokenizer

from cerebras.modelzoo.data_preparation.utils import (
    convert_to_unicode,
    whitespace_tokenize,
)


class BaseTokenizer:
    """
    Class for base tokenization of a piece of text
    Handles grammar operations like removing strip accents, checking for
    chinese characters in text, handling splitting on punctuation and control
    characters. Also handles creating the tokenizer for converting tokens->id
    and id->tokens and storing vocabulary for the dataset
    :param str vocab_file: File containing vocabulary, each token in new line
    :param bool do_lower: Specifies whether to convert to lower case for data
    processing
    """

    def __init__(self, vocab_file, do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file

        # prepare tokenizer with correct camel case handler
        # and filters for vocabulary processing
        self.tokenizer = Tokenizer(filters='', lower=self.do_lower_case)
        self._prepare_tokenizer()

    def _prepare_tokenizer(self):
        """
        Loads the vocabulary for token->id and id->token mapping
        """

        all_tokens = []
        with open(self.vocab_file, 'r') as reader:
            for line in reader:
                token = convert_to_unicode(line)
                if not token:
                    break
                token = token.strip()
                all_tokens.append(token)

        self.tokenizer.fit_on_texts(all_tokens)

    def _is_control_char(self, char):
        """
        Checks where `char` is a control character
        """
        cat = unicodedata.category(char)
        if cat in ("Cc", "Cf"):
            return True
        return False

    def _is_whitespace(self, char):
        """
        Checks whether `char` is a whitespace character
        """
        if char == " ":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_punctuation(self, char):
        """
        Checks whether `char` is a punctuation character
        """
        cp = ord(char)
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _is_chinese_char(self, cp):
        """
        Checks if CP is the codepoint of a CJK character
        """
        # This defines a "chinese character" as anything in the CJK unicode
        # block:
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)

        # CJK does not include all Japanese and Korean characters. The modern
        # Korean Hangul alphabet is a different block, as is Japanese
        # Hiragana and Katakana. These alphabets are used to write
        # space-separated words, so they are not treated specifically and
        # handled like all of the other languages

        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False

    def _tokenize_chinese_chars(self, text):
        """
        Adds whitespace around any CJK characters
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)

        return "".join(output)

    def _run_strip_accents(self, text):
        """
        Strips accents from a piece of text
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punctuation(self, text):
        """
        Splits punctutation on a piece of text
        """

        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _clean_text(self, text):
        """
        Performs cleanup on text and removes invalid characters
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control_char(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)

        return "".join(output)

    def tokenize(self, text):
        """
        Tokenizes a piece of text. Does not convert to ids
        """

        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # handle multilingual and Chinese models
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text, lower=self.do_lower_case)
        split_tokens = []
        for token in orig_tokens:
            token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punctuation(token))

        return whitespace_tokenize(
            " ".join(split_tokens), lower=self.do_lower_case
        )


class WordPieceTokenizer(BaseTokenizer):
    """
    Class for tokenization of a piece of text into its word pieces
    :param str vocab_file: File containing vocabulary, each token in new line
    :param str unknown_token: Token for words not in vocabulary
    :param int max_input_chars_per_word: Max length of word for splitting
    :param bool do_lower: Specifies whether to convert to lower case for data
    processing
    """

    def __init__(
        self,
        vocab_file,
        unknown_token="[UNK]",
        max_input_chars_per_word=200,
        do_lower_case=True,
    ):
        super(WordPieceTokenizer, self).__init__(vocab_file, do_lower_case)

        self.unknown_token = unknown_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenize a piece of text into its word pieces
        This uses a greedy longest-match-first algorithm
        to perfom tokenization using the given vocabulary.
        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]

        Does not convert to ids.
        """

        text = convert_to_unicode(text)
        output_tokens = []

        for token in whitespace_tokenize(text, lower=self.do_lower_case):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unknown_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.tokenizer.word_index:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unknown_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


class FullTokenizer:
    """
    Class for full tokenization of a piece of text
    Calls BaseTokenizer and WordPiece tokenizer to perform basic grammar
    operations and wordpiece splits
    :param str vocab_file: File containing vocabulary, each token in new line
    :param bool do_lower: Specifies whether to convert to lower case for data
    processing
    """

    def __init__(self, vocab_file, do_lower_case=True):
        self.baseTokenizer = BaseTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case
        )

        self.wpTokenizer = WordPieceTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case
        )

    def convert_tokens_to_ids(self, text):
        """
        Converts a list of tokens to a list of ids
        We shift all outputs by 1 because of the dictionary formed by
        keras `Tokenizer` starts with index 1 instead of 0.
        """
        tknzd_seq = self.baseTokenizer.tokenizer.texts_to_sequences(text)
        tknzd_seq = np.concatenate(tknzd_seq).tolist() if tknzd_seq else []
        return list(map(lambda x: x - 1, tknzd_seq))

    def convert_ids_to_tokens(self, text):
        """
        Converts a list of ids to a list of tokens
        We shift all inputs by 1 because of the ids->token dictionary formed by
        keras `Tokenizer` starts with index 1 instead of 0.
        """
        return [
            self.baseTokenizer.tokenizer.index_word[item + 1] for item in text
        ]

    def tokenize(self, text):
        """
        Perform basic tokenization followed by wordpiece tokenization on a
        piece of text. Does not convert to ids.
        """
        split_tokens = []

        for token in self.baseTokenizer.tokenize(text):
            for sub_token in self.wpTokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def get_vocab_words(self):
        """
        Returns a list of the words in the vocab
        """
        return list(self.baseTokenizer.tokenizer.word_index.keys())
