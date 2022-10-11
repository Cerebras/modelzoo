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

"""
Tokenization classes and functions.
"""


import re

from modelzoo.transformers.tf.transformer.input.data_processing.Tokenization import (
    BaseTokenizer as TransformerBaseTokenizer,
)


class T5BaseTokenizer(TransformerBaseTokenizer):
    """
    Class for base tokenization of a piece of text.
    This tokenizer inherits from `TransformerBaseTokenizer`
        which contains most of the main methods.

    :param str vocab_file: File containing vocabulary,
        each token in new line.
    :param str unk_token: Token to be used for out of
        vocabulary words.
    :param bool do_lower_case: Whether the tokens should
        be converted to lower case.
    """

    def __init__(self, vocab_file, unk_token="<unk>", do_lower_case=False):
        super(T5BaseTokenizer, self).__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            do_lower_case=do_lower_case,
        )

    def convert_tokens_to_ids(self, text):
        """
        Converts a list of tokens to a list of ids.
        """
        return [self.convert_token_to_id(token) for token in text]

    def convert_token_to_id(self, token):
        """
        Converts a token in an id using the vocab.

        Extra tokens are indexed from the end of the vocabulary up
            to the beginning ("<extra_id_0>" is the last token in the vocabulary).

        We shift all outputs by `1` because of the dictionary formed by
            keras `Tokenizer` starts with index `1` instead of `0`.
        """

        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1

        return self.tokenizer.texts_to_sequences([token])[0][0] - 1

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts a list of ids to a list of tokens.
        """
        return [self.convert_id_to_token(token_id) for token_id in token_ids]

    def convert_id_to_token(self, token_id):
        """
        Extra tokens are indexed from the end of the vocabulary up
            to the beginning ("<extra_id_0>" is the last token in the vocabulary).

        We shift all inputs by `1` because of the ids -> token dictionary formed by
            keras `Tokenizer` starts with index `1` instead of `0`.

        Tokens not present in vocab are assigned `unk_token` id.
        """
        if token_id < len(self.tokenizer.index_word):
            token = self.tokenizer.index_word[token_id + 1]
        else:
            token = f"<extra_id_{self.vocab_size - 1 - token_id}>"
        return token
