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

"""Tokenization classes."""

import collections

import six


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise TypeError(
                f"must be a string or bytes object, not {type(text)}"
            )
    else:
        raise ValueError("Not running on Python3?")


def load_vocab(vocab_file):
    """
    Loads a vocabulary file into a dictionary.
    """
    vocab = collections.OrderedDict()
    with open(vocab_file, "r") as fin:
        lines = fin.readlines()

    for i, line in enumerate(lines):
        token = convert_to_unicode(line)
        if not token:
            break
        vocab[token.strip()] = i
    return vocab


def convert_by_vocab(vocab, items):
    """
    Converts a sequence of [tokens|ids] using the vocab.
    """
    return [vocab[item] for item in items]


class GenomeTokenizer(object):
    def __init__(self, vocab_file, ngram=5, stride=3, ideas=False):
        self.ngram = ngram
        self.stride = stride
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.ideas = ideas

    def tokenize(self, t):
        if not self.ideas:
            t = t.upper()

        if self.ngram == 1:
            toks = list(t)
        else:
            toks = [
                t[i : i + self.ngram]
                for i in range(0, len(t), self.stride)
                if len(t[i : i + self.ngram]) == self.ngram
            ]

        # remove extra tokens if doesn't fit ngram size
        if len(toks[-1]) < self.ngram:
            toks = toks[:-1]

        # for ideas states: take the mode of ideas state as the token
        if self.ideas:
            toks = [str(max(tok, key=tok.count)) for tok in toks]

        return toks

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)
