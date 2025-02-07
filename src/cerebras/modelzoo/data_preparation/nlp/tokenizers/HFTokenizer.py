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

import json
import logging
import os

from tokenizers import Tokenizer

logger = logging.getLogger("HFTokenizer")
logger.setLevel(logging.INFO)


class HFTokenizer:
    """Designed to integrate the HF's Tokenizer library
    Args:
        vocab_file (str): A vocabulary file to create the tokenizer from.
        special_tokens (list, str): A list or a string representing the special
            tokens that are to be added to the tokenizer.
    """

    def __init__(self, vocab_file, special_tokens=None):
        self.tokenizer = Tokenizer.from_file(vocab_file)
        vocab_dir_path = os.path.dirname(vocab_file)
        self.tokenizer_config = os.path.join(
            vocab_dir_path, "tokenizer_config.json"
        )
        if not os.path.exists(self.tokenizer_config):
            self.tokenizer_config = None
            logger.warning(
                """Tokenizer config file is not available in the tokenizer encoder file 
            directory. Therefore setting the default_add_bos flag to be false ."""
            )

        if special_tokens:
            self.add_special_tokens(special_tokens)
        self.set_eos_pad_tokens()
        self.add_bos_token = False
        self.bos_token = None
        if self.tokenizer_config:
            with open(self.tokenizer_config, 'r') as json_file:
                data = json.load(json_file)
            self.add_bos_token = data.get("add_bos_token", False)
            self.bos_token = self.get_token_from_tokenizer_config(
                data, "bos_token"
            )
            eos_token = self.get_token_from_tokenizer_config(data, "eos_token")
            pad_token = self.get_token_from_tokenizer_config(data, "pad_token")
            if eos_token:
                self.eos_id = self.get_token_id(eos_token)
            if pad_token:
                self.pad_id = self.get_token_id(pad_token)

            self.bos_token_id = (
                self.get_token_id(self.bos_token) if self.bos_token else None
            )

    def set_eos_pad_tokens(self):
        self.eos_id = self.get_token_id("<|endoftext|>")
        self.pad_id = self.get_token_id("<|padding|>")

    def get_token_from_tokenizer_config(self, json_data, token):
        """
        This api is designed to extract token information from the tokenizer
        config json file. We assume the token data to be in 2 formats either as
        a string or a dictionary.
        """

        if token in json_data and isinstance(json_data[token], str):
            return json_data[token]
        elif token in json_data and isinstance(json_data[token], dict):
            return json_data[token].get("content", None)
        # If token not in the expected formats or missing
        else:
            return None

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def add_special_tokens(self, special_tokens):
        self.tokenizer.add_special_tokens(special_tokens)

    def add_token(self, token):
        self.tokenizer.add_tokens(token)

    def get_token_id(self, token):
        return self.tokenizer.token_to_id(token)

    def get_token(self, id):
        return self.tokenizer.id_to_token(id)

    @property
    def eos(self):
        return self.eos_id

    @property
    def pad(self):
        return self.pad_id
