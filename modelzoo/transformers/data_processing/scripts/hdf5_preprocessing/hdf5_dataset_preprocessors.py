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
    check_fim_special_tokens,
    create_features_auto_lm,
    create_features_auto_lm_vsl,
    create_features_summarization,
    create_features_summarization_vsl,
    fim,
    handle_bos_token_default,
    split_text_and_tokenize,
    wikitext_detokenizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class LMDataPreprocessor(HDF5BasePreprocessor):
    def __init__(self, params):
        super(LMDataPreprocessor, self).__init__(params)
        self.jsonl_key = params["dataset"].pop("jsonl_key", "text")
        assert (
            "prompt_key" not in params["dataset"]
            and "completion_key" not in params["dataset"]
        ), "Prompt/ Completion key cannot be provided for LMDataProcessor"
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
        if self.split_text_to_tokenize:
            # TODO: implement a better fix for this by updating the tokenizer
            # normalization rules. This is a temporary fix and it may
            # cause issues with the spacing tokens being repeated.
            tokenized_text = split_text_and_tokenize(
                text,
                self.tokenizer,
                max_tok_len=self.chunk_len_to_split,
                remove_bos_in_chunks=self.remove_bos_in_chunks,
            )
        else:
            tokenized_text = self.tokenizer.encode(text)

        if self.eos_id is not None:
            tokenized_text += [self.eos_id]

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

        tokenizable_columns = {"jsonl_key": self.jsonl_key}
        reader = Reader(file, tokenizable_columns)
        for doc in reader.stream_data():
            # update chars and bytes stats on base processor
            self.raw_chars_count += len(doc)
            self.raw_bytes_count += len(doc.encode("utf-8"))

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
        assert (
            "jsonl_key" not in params["dataset"]
        ), "Jsonl key cannot be provided for SummarizationPreprocessor"
        self.completion_key = params["dataset"].pop("completion_key")
        assert self.eos_id is not None, "eos_id must be set for summarization."
        self.sep_token = params["dataset"].pop("sep_token", None)
        self.sep_id = None
        if self.sep_token:
            self.add_token(self.sep_token)
            self.sep_id = self.tokenizer.get_token_id(self.sep_token)
            logging.warning(
                f"A sep token {self.sep_token} was added to tokenizer. This "
                "will change the vocab size. If you are using a pretrained "
                "model, you will need to avoid adding this."
            )

        if params["dataset"]:
            logger.warning(
                "The following dataset params are unused: "
                + ", ".join(params["dataset"].keys())
            )

    def file_read_generator(self, file):
        tokenizable_columns = {
            'prompt_key': self.prompt_key,
            'completion_key': self.completion_key,
        }
        reader = Reader(file, tokenizable_columns)
        for doc in reader.stream_data():
            if self.prompt_key not in doc or self.completion_key not in doc:
                logger.warning(
                    "prompt_key or completion_key not in file, file may be corrupted"
                )
                continue
            prompt = doc[self.prompt_key]
            completion = doc[self.completion_key]
            self.raw_chars_count += len(prompt) + len(completion)
            self.raw_bytes_count += len(prompt.encode("utf-8")) + len(
                completion.encode("utf-8")
            )
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
            self.eos_id,
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


class FIMDataPreprocessor(LMDataPreprocessor):
    def __init__(self, params):
        super(FIMDataPreprocessor, self).__init__(params)
        self.fim_rate = params['processing'].get("fim_rate")
        self.spm_rate = params['processing'].get("spm_rate")

        # Ensures that FIM tokens are specified in config, and that
        # the specified tokens are actually in the tokenizer
        check_fim_special_tokens(params, self.tokenizer)

        self.default_bos_token, self.opt_bos_tok_id = handle_bos_token_default(
            self.tokenizer
        )

        self.suffix_tok_id = self.tokenizer.encode(
            params['processing'].get("fim_suffix_tok")
        )[-1]
        self.prefix_tok_id = self.tokenizer.encode(
            params['processing'].get("fim_prefix_tok")
        )[-1]
        self.middle_tok_id = self.tokenizer.encode(
            params['processing'].get("fim_middle_tok")
        )[-1]

    def preprocessing_generator(self, doc):
        for i, sample in enumerate(self.tokenize_text_auto_lm(doc)):
            if sample != []:
                sample = fim(
                    sample,
                    i,
                    self.tokenizer,
                    self.fim_rate,
                    self.spm_rate,
                    self.suffix_tok_id,
                    self.prefix_tok_id,
                    self.middle_tok_id,
                    self.pad_id,
                    self.eos_id,
                    self.opt_bos_tok_id,
                )
            else:
                self.discarded_files += 1
            yield sample


class VSLLMDataPreprocessor(LMDataPreprocessor):
    use_vsl = True

    def __init__(self, params):
        self.fold_long_doc = params["dataset"].pop("fold_long_doc", True)
        self.position_ids_dtype = params["dataset"].pop(
            "position_ids_dtype", "int32"
        )
        super(VSLLMDataPreprocessor, self).__init__(params)

        self.chunk_lengths = []
        self.tokenized_chunks = []
        self.chunk_count = 0

    def _add_new_chunk(self, tokenized_text, tokenized_length):
        self.chunk_lengths.append(self.max_seq_length - tokenized_length)
        self.tokenized_chunks.append([tokenized_text])
        self.chunk_count += 1

    def tokenize_text(self, text):
        if self.use_ftfy:
            text = ftfy.fix_text(text, normalization=self.ftfy_normalizer)
        if self.wikitext_detokenize:
            text = wikitext_detokenizer(text)

        tokenized_text = self.tokenizer.encode(text)

        if self.eos_id is not None:
            tokenized_text += [self.eos_id]

        tokenized_text_len = len(tokenized_text)
        if tokenized_text_len < self.min_sequence_len:
            self.discarded_files += 1
            return

        if self.rng.random() < self.short_seq_prob:
            tokenized_text = tokenized_text[
                0 : self.rng.randint(2, self.max_seq_length)
            ]
            tokenized_text_len = len(tokenized_text)

        if tokenized_text_len > self.max_seq_length + 1:
            if not self.fold_long_doc:
                self.discarded_files += 1
                return
            for i in range(0, tokenized_text_len, self.max_seq_length):
                if tokenized_text_len - i < self.max_seq_length + 1:
                    tokenized_text = tokenized_text[i:]
                    tokenized_text_len = tokenized_text_len - i
                else:
                    self._add_new_chunk(
                        tokenized_text[i : i + self.max_seq_length + 1],
                        self.max_seq_length,
                    )

        if tokenized_text_len < 2:
            return

        tokenized_text_len -= 1
        create_new_chunk = True
        for idx in range(self.chunk_count - 1, -1, -1):
            if tokenized_text_len <= self.chunk_lengths[idx]:
                self.tokenized_chunks[idx].append(tokenized_text)
                self.chunk_lengths[idx] -= tokenized_text_len
                create_new_chunk = False
                break

        if create_new_chunk:
            self._add_new_chunk(tokenized_text, tokenized_text_len)

    def vsl_sample_generator(self, generation_len):
        for _ in range(generation_len):
            bin = self.tokenized_chunks.pop(0)
            num_pad = self.chunk_lengths.pop(0)
            self.chunk_count -= 1

            yield create_features_auto_lm_vsl(
                bin,
                self.max_seq_length,
                num_pad,
                pad_id=self.pad_id,
                inverted_mask=self.inverted_mask,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
                attention_span_dtype=self.position_ids_dtype,
                position_ids_dtype=self.position_ids_dtype,
            )

    def preprocessing_generator(self, doc):
        self.tokenize_text(doc)

        if self.chunk_count > self.files_per_record:
            for sample in self.vsl_sample_generator(self.files_per_record):
                yield sample
        else:
            yield []


class VSLSummarizationPreprocessor(SummarizationPreprocessor):
    use_vsl = True

    def __init__(self, params):
        self.position_ids_dtype = params["dataset"].pop(
            "position_ids_dtype", "int32"
        )
        super(VSLSummarizationPreprocessor, self).__init__(params)

        self.chunk_lengths = []
        self.tokenized_chunks = []
        self.chunk_count = 0

    def tokenize_text(self, doc):
        prompt, completion = doc
        if self.use_ftfy:
            prompt = ftfy.fix_text(prompt, normalization=self.ftfy_normalizer)
            completion = ftfy.fix_text(
                completion, normalization=self.ftfy_normalizer
            )
        if self.wikitext_detokenize:
            prompt = wikitext_detokenizer(prompt)
            completion = wikitext_detokenizer(completion)

        prompt_ids = self.tokenizer.encode(prompt)
        completion_ids = self.tokenizer.encode(completion)

        total_len = len(prompt_ids) + len(completion_ids)
        if self.sep_id is not None:
            total_len += 1
        if total_len > self.max_seq_length:
            logger.warning(
                "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
            )
            self.discarded_files += 1
            return
        if total_len < self.min_sequence_len:
            logger.warning(
                "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
            )
            self.discarded_files += 1
            return

        create_new_chunk = True
        for idx in range(self.chunk_count - 1, -1, -1):
            if total_len <= self.chunk_lengths[idx]:
                self.tokenized_chunks[idx].append((prompt_ids, completion_ids))
                self.chunk_lengths[idx] -= total_len
                create_new_chunk = False
                break

        if create_new_chunk:
            self.chunk_lengths.append(self.max_seq_length - total_len)
            self.tokenized_chunks.append([(prompt_ids, completion_ids)])
            self.chunk_count += 1

    def vsl_sample_generator(self, generation_len):
        for _ in range(generation_len):
            bin = self.tokenized_chunks.pop(0)
            num_pad = self.chunk_lengths.pop(0)
            self.chunk_count -= 1

            yield create_features_summarization_vsl(
                bin,
                self.max_seq_length,
                num_pad,
                pad_id=self.pad_id,
                eos_id=self.eos_id,
                sep_id=self.sep_id,
                inverted_mask=self.inverted_mask,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
                attention_span_dtype=self.position_ids_dtype,
                position_ids_dtype=self.position_ids_dtype,
            )

    def preprocessing_generator(self, doc):
        self.tokenize_text(doc)

        if self.chunk_count > self.files_per_record:
            for sample in self.vsl_sample_generator(self.files_per_record):
                yield sample
        else:
            yield []
