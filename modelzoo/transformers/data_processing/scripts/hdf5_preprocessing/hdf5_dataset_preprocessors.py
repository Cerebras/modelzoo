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

import logging

import ftfy

from modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.hdf5_base_preprocessor import (
    HDF5BasePreprocessor,
)
from modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.utils import (
    Reader,
    create_features_auto_lm,
    create_features_summarization,
    wikitext_detokenizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class LMDataPreprocessor(HDF5BasePreprocessor):
    def __init__(self, params):
        super(LMDataPreprocessor, self).__init__(params)
        self.jsonl_key = params["dataset"].pop("jsonl_key", "text")
        self.use_ftfy = params["dataset"].pop("use_ftfy", False)
        self.ftfy_normalizer = params["dataset"].pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = params["dataset"].pop(
            "wikitext_detokenize", False
        )
        self.pack_sequences = params["dataset"].pop("pack_sequences", True)
        self.min_sequence_len = params["dataset"].pop("min_sequence_len", 10)
        self.input_ids_dtype = params["dataset"].pop("input_ids_dtype", "int32")
        self.input_mask_dtype = params["dataset"].pop(
            "input_mask_dtype", "int32"
        )
        self.inverted_mask = params["dataset"].pop("inverted_mask", False)

        if params["dataset"]:
            logger.warning(
                "The following dataset params are unused: "
                + ", ".join(params["dataset"].keys())
            )

        self.prefix = []

    def tokenize_text_auto_lm(self, text):
        if self.use_ftfy:
            text = ftfy.fix_text(text, normalization=self.ftfy_normalizer)
        if self.wikitext_detokenize:
            text = wikitext_detokenizer(text)
        # tokenize text
        tokenized_text = self.tokenizer.encode(text)

        if self.eos_id is not None:
            tokenized_text += (
                self.eos_id if isinstance(self.eos_id, list) else [self.eos_id]
            )

        all_text = self.prefix + tokenized_text
        tokenized_text_chunks = [
            all_text[i : i + self.max_seq_length + 1]
            for i in range(0, len(all_text), self.max_seq_length)
        ]

        # reset prefix
        self.prefix = []

        # update prefix if last chunk is < max_seq_length
        num_tokens_last_chunk = len(tokenized_text_chunks[-1])
        if self.pack_sequences:
            if num_tokens_last_chunk < self.max_seq_length + 1:
                last_chunk = tokenized_text_chunks.pop(-1)
                self.prefix.extend(last_chunk)
        elif num_tokens_last_chunk < 2:
            _ = tokenized_text_chunks.pop(-1)
            self.discarded_files += 1

        return [
            create_features_auto_lm(
                chunk,
                self.max_seq_length,
                short_seq_prob=self.short_seq_prob,
                inverted_mask=self.inverted_mask,
                pad_id=self.pad_id,
                min_len=self.min_sequence_len,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
                rng=self.rng,
            )
            for chunk in tokenized_text_chunks
        ]

    def file_read_generator(self, file):
        reader = Reader(file)
        for doc in reader.stream_data(jsonl_key=self.jsonl_key):
            yield doc

    def preprocessing_generator(self, doc):
        for sample in self.tokenize_text_auto_lm(doc):
            if sample == []:
                self.discarded_files += 1
            yield sample


class SummarizationPreprocessor(HDF5BasePreprocessor):
    def __init__(self, params):
        super(SummarizationPreprocessor, self).__init__(params)
        self.use_ftfy = params["dataset"].pop("use_ftfy", False)
        self.ftfy_normalizer = params["dataset"].pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = params["dataset"].pop(
            "wikitext_detokenize", False
        )
        self.min_sequence_len = params["dataset"].pop("min_sequence_len", 10)
        self.input_ids_dtype = params["dataset"].pop("input_ids_dtype", "int32")
        self.input_mask_dtype = params["dataset"].pop(
            "input_mask_dtype", "int32"
        )
        self.inverted_mask = params["dataset"].pop("inverted_mask", False)

        self.prompt_key = params["dataset"].pop("prompt_key")
        self.completion_key = params["dataset"].pop("completion_key")
        self.sep_token = params["dataset"].pop("sep_token", None)
        self.sep_id = None
        if self.sep_token:
            self.add_token(self.sep_token)
            self.sep_id = self.tokenizer.get_token_id(self.sep_token)

        if params["dataset"]:
            logger.warning(
                "The following dataset params are unused: "
                + ", ".join(params["dataset"].keys())
            )

    def file_read_generator(self, file):
        reader = Reader(file)
        for doc in reader.stream_data():
            if self.prompt_key not in doc or self.completion_key not in doc:
                logger.warning(
                    "prompt_key or completion_key not in file, file may be corrupted"
                )
                continue
            prompt = doc[self.prompt_key]
            completion = doc[self.completion_key]
            yield prompt, completion

    def preprocessing_generator(self, doc):
        prompt, completion = doc
        if self.use_ftfy:
            prompt = ftfy.fix_text(prompt, normalization=self.ftfy_normalizer)
            completion = ftfy.fix_text(
                completion, normalization=self.ftfy_normalizer
            )
        if self.wikitext_detokenize:
            prompt = wikitext_detokenizer(prompt)
            completion = wikitext_detokenizer(completion)

        prompt_encoded = self.tokenizer.encode(prompt)
        completion_encoded = self.tokenizer.encode(completion)

        sample = create_features_summarization(
            prompt_encoded,
            completion_encoded,
            self.max_seq_length,
            self.eos_id
            if not isinstance(self.eos_id, list)
            else self.eos_id[0],
            self.sep_id,
            self.pad_id,
            min_len=self.min_sequence_len,
            inverted_mask=self.inverted_mask,
            input_ids_dtype=self.input_ids_dtype,
            input_mask_dtype=self.input_mask_dtype,
            labels_dtype=self.input_ids_dtype,
        )

        if sample == []:
            self.discarded_files += 1

        yield sample
