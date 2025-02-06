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

from tokenizers import Tokenizer


class HFTokenizer:
    """Designed to integrate the HF's Tokenizer library
    Args:
        vocab_file (str): A vocabulary file to create the tokenizer from.
        special_tokens (list, str): A list or a string representing the special
            tokens that are to be added to the tokenizer.
    """

    def __init__(self, vocab_file, special_tokens=None):
        self.tokenizer = Tokenizer.from_file(vocab_file)

        if special_tokens:
            self.add_special_tokens(special_tokens)

        self.set_eos_pad_tokens()

    def set_eos_pad_tokens(self):
        self.eos_id = self.get_token_id("<|endoftext|>")
        self.pad_id = self.get_token_id("<|padding|>")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

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
