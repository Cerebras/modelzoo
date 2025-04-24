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

import os

import transformers


class GenericTokenizer:
    def __init__(self, params, filepath):
        self.processing_params = params['processing']
        self.dataset_params = params['dataset']

        self.filepath = filepath
        self.custom_tokenizer = self.processing_params.pop(
            "custom_tokenizer", None
        )
        if self.processing_params.get('huggingface_tokenizer'):
            self.tokenizer = self.initialize_huggingfacetokenizer()
        elif self.custom_tokenizer == "gpt2tokenizer":
            self.tokenizer = self.initialize_gpt2tokenizer()
        elif self.custom_tokenizer == "neoxtokenizer":
            self.tokenizer = self.initialize_neoxtokenizer()
        elif self.custom_tokenizer:
            # Old tokenizer_type key
            self.tokenizer = self.initialize_customtokenizer()
        else:
            self.tokenizer_type = self.processing_params.pop('tokenizer_type')
            self.tokenizer = self._initialize_customtokenizer_old()

        register_image_token = self.dataset_params.get(
            'register_special_image_token', True
        )
        if register_image_token and self.dataset_params.get(
            'use_single_image_token', False
        ):
            image_token = self.dataset_params.get("image_token")

            if not image_token:
                raise ValueError(
                    "The 'image_token' must be specified in dataset_params."
                )

            self.tokenizer.add_special_tokens(
                {'additional_special_tokens': [image_token]}
            )

    def initialize_gpt2tokenizer(self):
        if self.processing_params.get('vocab_file', False):
            merges_file = os.path.normpath(
                os.path.join(
                    os.path.dirname(self.filepath),
                    self.processing_params.pop('vocab_file'),
                )
            )
            vocab_file = os.path.normpath(
                os.path.join(
                    os.path.dirname(self.filepath),
                    self.processing_params.pop('encoder_file'),
                )
            )
        else:
            merges_path = self.processing_params['tokenizer_params'].pop(
                'vocab_file'
            )
            vocab_path = self.processing_params['tokenizer_params'].pop(
                'encoder_file'
            )
            merges_file = os.path.normpath(
                os.path.join(os.path.dirname(self.filepath), merges_path)
            )
            vocab_file = os.path.normpath(
                os.path.join(os.path.dirname(self.filepath), vocab_path)
            )
        return transformers.GPT2TokenizerFast(
            vocab_file=vocab_file,
            merges_file=merges_file,
            name_or_path="gpt2-tokenizer",
            **self.processing_params.get('tokenizer_params', {}),
        )

    def initialize_neoxtokenizer(self):
        from tokenizers import Tokenizer

        tokenizer_params = self.processing_params.get('tokenizer_params', {})
        return transformers.PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(
                tokenizer_params.pop("encoder_file", None)
            ),
            name_or_path="neox-tokenizer",
            **tokenizer_params,
        )

    def initialize_huggingfacetokenizer(self):
        return transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.processing_params[
                'huggingface_tokenizer'
            ],
            **self.processing_params.get('tokenizer_params', {}),
        )

    def initialize_customtokenizer(self):
        import importlib

        module, class_name = self.custom_tokenizer.rsplit(':', 1)
        module = importlib.import_module(module)
        TokenizerClass = getattr(module, class_name)

        return TokenizerClass(
            **self.processing_params.get('tokenizer_params', {})
        )

    def _initialize_customtokenizer_old(self):
        merges_file = os.path.normpath(
            os.path.join(
                os.path.dirname(self.filepath),
                self.processing_params['vocab_file'],
            )
        )
        vocab_file = os.path.normpath(
            os.path.join(
                os.path.dirname(self.filepath),
                self.processing_params['encoder_file'],
            )
        )
        tokenizer_class = getattr(transformers, self.tokenizer_type)

        return tokenizer_class(vocab_file=vocab_file, merges_file=merges_file)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)
