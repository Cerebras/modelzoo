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
Detokenizer class using the normalizer and detokenizer from sacremoses.

Note that this assumes the use of BPE tokenization with @@ as the connector
symbol right-appended to joined tokens, as is common with Transformer using
the WMT datasets.

See: https://github.com/alvations/sacremoses
"""

import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer

from cerebras.modelzoo.data.nlp.bert.bert_utils import build_vocab


class Detokenizer:
    def __init__(self, params, lang="de"):
        eval_input_params = params["eval_input"]
        self.vocab_file = params["eval_input"]["tgt_vocab_file"]
        do_lower = eval_input_params.get("do_lower", False)

        # Get special tokens and tokens that should not be masked.
        self.special_tokens = {
            "oov_token": eval_input_params.get("oov_token", "<unk>"),
            "sos_token": eval_input_params.get("sos_token", "<s>"),
            "eos_token": eval_input_params.get("eos_token", "</s>"),
            "pad_token": eval_input_params.get("pad_token", "<pad>"),
        }
        if do_lower:
            self.special_tokens = {
                key: value.lower() for key, value in self.special_tokens.items()
            }

        # Get vocab file and size.
        self.vocab, self.vocab_size = build_vocab(
            self.vocab_file, do_lower, self.special_tokens["oov_token"]
        )

        # Updating input and model params to account extra ids
        # for T5 Language Modeling task.
        extra_ids = eval_input_params.get("extra_ids", 0)
        self.vocab_size += extra_ids

        self.special_tokens_indices = {
            key: self.vocab.forward([value])[0]
            for key, value in self.special_tokens.items()
        }

        self._lang = lang
        self._detokenizer = MosesDetokenizer(lang=self._lang)
        self._normalizer = MosesPunctNormalizer(lang=self._lang)

    def _convert_ids_to_tokens(self, token_ids):
        """
        token_ids: list or Tensor of ints.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = self.vocab.backward(token_ids)
        return tokens

    def _remove_special_tokens(self, line):
        """
        line: str
        """
        line = line.replace(self.special_tokens["sos_token"], "")
        line = line.replace(self.special_tokens["eos_token"], "")
        line = line.replace(self.special_tokens["pad_token"], "")
        line = line.replace(" @-@ ", "-")
        line = line.replace("@@", "")
        line = line.strip()
        return line

    def convert_ids_to_str(self, token_ids):
        """
        token_ids: list or Tensor of ints.
        """
        token_strs = self._convert_ids_to_tokens(token_ids)
        line = " ".join(token_strs).replace("@@ ", "")
        line = self._remove_special_tokens(line)
        line = self.detokenize(line)
        return line

    def detokenize(self, line):
        """
        line: str
        """
        line = self._normalizer.normalize(line)
        token_strs = line.split(" ")
        return self._detokenizer.detokenize(token_strs)

    def decode(self, token_ids):
        """
        token_ids: list or Tensor of ints.
        """
        return self.convert_ids_to_str(token_ids)
