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

import importlib
import sys
import types
import unittest


def _install_utils_stub():
    utils_name = (
        "cerebras.modelzoo.data_preparation.data_preprocessing.utils"
    )
    utils_module = types.ModuleType(utils_name)

    def _noop(*args, **kwargs):
        return None

    class _Logger:
        def warning(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    utils_module.append_eos_to_multiple_semantic_regions = _noop
    utils_module.clean_text = _noop
    utils_module.default_chat_template = lambda: "stub-chat-template"
    utils_module.find_region_in_formatted_string = _noop
    utils_module.find_token_range = _noop
    utils_module.get_data_stats = _noop
    utils_module.setup_warning_logging = lambda *args, **kwargs: _Logger()
    utils_module.truncate_sequence = _noop

    sys.modules[utils_name] = utils_module


def _reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class _Tokenizer:
    def __init__(self, chat_template=None):
        self.sep_token = None
        self.chat_template = chat_template

    def convert_ids_to_tokens(self, token_id):
        return f"tok-{token_id}"

    def get_vocab(self):
        return {}

    def add_special_tokens(self, tokens):
        return None

    def convert_tokens_to_ids(self, token):
        return 7


def _build_params():
    return {
        "dataset": {},
        "processing": {},
        "setup": {"output_dir": "/tmp/modelzoo-test"},
    }


class FinetuningTokenGeneratorInitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_utils_stub()
        cls.text_module = _reload_module(
            "cerebras.modelzoo.data_preparation.data_preprocessing.finetuning_token_generator"
        )
        cls.mllama_module = _reload_module(
            "cerebras.modelzoo.data_preparation.data_preprocessing.finetuning_token_generator_mllama"
        )

    def test_text_generator_uses_pad_token_when_eos_id_is_none(self):
        generator = self.text_module.FinetuningTokenGenerator(
            _build_params(),
            _Tokenizer(chat_template=None),
            eos_id=None,
            pad_id=99,
        )

        self.assertEqual(generator.pad_id, 99)
        self.assertIsNone(generator.eos_id)
        self.assertEqual(generator.eos_token, "tok-99")
        self.assertEqual(
            generator.tokenizer.chat_template, "stub-chat-template"
        )

    def test_text_generator_uses_explicit_eos_id_when_provided(self):
        generator = self.text_module.FinetuningTokenGenerator(
            _build_params(),
            _Tokenizer(chat_template="existing-template"),
            eos_id=42,
            pad_id=99,
        )

        self.assertEqual(generator.eos_token, "tok-42")
        self.assertEqual(generator.tokenizer.chat_template, "existing-template")

    def test_mllama_generator_uses_pad_token_when_eos_id_is_none(self):
        generator = self.mllama_module.FinetuningTokenGenerator(
            _build_params(),
            _Tokenizer(chat_template="existing-template"),
            eos_id=None,
            pad_id=99,
        )

        self.assertEqual(generator.pad_id, 99)
        self.assertIsNone(generator.eos_id)
        self.assertEqual(generator.eos_token, "tok-99")

    def test_mllama_generator_uses_explicit_eos_id_when_provided(self):
        generator = self.mllama_module.FinetuningTokenGenerator(
            _build_params(),
            _Tokenizer(chat_template="existing-template"),
            eos_id=42,
            pad_id=99,
        )

        self.assertEqual(generator.eos_token, "tok-42")


if __name__ == "__main__":
    unittest.main()
