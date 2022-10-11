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

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from modelzoo.transformers.data_processing.utils import convert_to_unicode


class BaseTokenizer:
    """
    Class for base tokenization of a piece of text
    handles creating the tokenizer for converting tokens->id
    and id->tokens and storing vocabulary for the dataset.
    :param str vocab_file: File containing vocabulary, each token in new line.
    :param str unk_token: Token to be used for out of vocabulary words.
    :param bool do_lower_case: Whether the tokens should be converted to lower case
        processing.
    """

    def __init__(self, vocab_file, unk_token="<unk>", do_lower_case=False):
        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file

        # prepare tokenizer with correct camel case handler
        # and filters for vocabulary processing.
        self.tokenizer = Tokenizer(filters="", lower=self.do_lower_case)
        self._prepare_tokenizer()
        # A hack to ensure the id for vocab words is the same order as seen in vocab file.
        self.tokenizer.oov_token = unk_token

        self.words = tf.constant(
            list(self.tokenizer.word_index.keys()), dtype=tf.string
        )
        self.word_ids = tf.constant(
            list(self.tokenizer.word_index.values()), dtype=tf.int32
        )

        self.unk_token = unk_token
        self.unk_token_id = self.get_id(self.unk_token)

        self.word_to_id_lookup_table = self._init_word_to_id_lookup_table()
        self.id_to_word_lookup_table = self._init_id_to_word_lookup_table()

    def _prepare_tokenizer(self):
        """
        Loads the vocabulary for token->id and id->token mapping.
        """
        all_tokens = []
        with tf.io.gfile.GFile(self.vocab_file, "r") as reader:
            for line in reader:
                token = convert_to_unicode(line)
                if not token:
                    break
                token = token.strip()
                all_tokens.append(token)

        self.tokenizer.fit_on_texts(all_tokens)

    def _init_word_to_id_lookup_table(self):
        """
        Initialize immutable Hash table for lookup.
        Used to convert string tokens to ids.
        We shift the unknown token id by 1 since our hash table is 1-based.
        """
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=self.words, values=self.word_ids, name="word_id_table_init"
        )

        lookup_table = tf.lookup.StaticHashTable(
            initializer, default_value=self.unk_token_id + 1
        )

        return lookup_table

    def _init_id_to_word_lookup_table(self):
        """
        Initialize immutable Hash table for lookup.
        Used to convert int ids tokens to string tokens.
        """
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=self.word_ids, values=self.words, name="id_word_table_init"
        )

        lookup_table = tf.lookup.StaticHashTable(
            initializer, default_value=self.unk_token
        )

        return lookup_table

    def convert_tokens_to_ids(self, text):
        """
        Converts a list of tokens to a list of ids.
        We shift all outputs by 1 because of the dictionary formed by
        keras `Tokenizer` starts with index 1 instead of 0.
        :param List[str] text: List of tokens strings.
        returns List of integer ids corresponding to token strings.
        """
        tknzd_seq = self.tokenizer.texts_to_sequences(text)
        tknzd_seq = tf.nest.flatten(tknzd_seq)
        return list(map(lambda x: x - 1, tknzd_seq))

    def convert_ids_to_tokens(self, text):
        """
        Converts a list of ids to a list of tokens.
        We shift all inputs by 1 because of the ids->token dictionary formed by
        keras `Tokenizer` starts with index 1 instead of 0.

        Tokens not present in vocab are assigned `unk_token` id
        """
        return [self.tokenizer.index_word[item + 1] for item in text]

    def get_vocab_words(self):
        """
        Returns a list of the words in the vocab.
        """
        return list(self.tokenizer.word_index.keys())

    def get_id(self, word_string):
        """
        Get the index for a particular word string.
        """
        return self.convert_tokens_to_ids([word_string])[0]

    def check_word_and_get_id(self, word):
        """
        Check if word is present in vocab and the index. 
        """
        if word not in self.get_vocab_words():
            raise ValueError(f"{word} not in vocab")

        return self.get_id(word)

    def convert_tokens_tensor_to_ids(self, text):
        """
        Converts a Tensor of tokens to a list of ids.
        :param Tensor[tf.str] tf Tensor containing tokens
        We shift all outputs by 1 because of the dictionary formed by
        keras `Tokenizer` starts with index 1 instead of 0.
        """
        return tf.cast(self.word_to_id_lookup_table.lookup(text) - 1, tf.int32)

    def convert_ids_tensor_to_tokens(self, ids):
        """
        Converts a Tensor of ids to a list of string tokens.
        :param Tensor[tf.int] tf Tensor containing tokens.

        We shift all inputs by 1 because of the ids->token dictionary formed by
        keras `Tokenizer` starts with index 1 instead of 0.

        Tokens not present in vocab are assigned `unk_token` id
        """
        return tf.cast(self.id_to_word_lookup_table.lookup(ids + 1), tf.string,)
