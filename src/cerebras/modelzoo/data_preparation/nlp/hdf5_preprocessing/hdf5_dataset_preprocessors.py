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
import os

import ftfy
import numpy as np

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.hdf5_base_preprocessor import (
    HDF5BasePreprocessor,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    SYSTEM_PROMPT_REGISTRY,
    DocObject,
    Reader,
    check_fim_special_tokens,
    create_features_auto_lm,
    create_features_auto_lm_vsl,
    create_features_llava_phase1,
    create_features_llava_phase2,
    create_features_summarization,
    create_features_summarization_vsl,
    fim,
    handle_bos_token_default,
    split_text_and_tokenize,
    wikitext_detokenizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ContinueLoopException(Exception):
    pass


class LMDataPreprocessor(HDF5BasePreprocessor):
    num_features = 3

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

        self.prefix = []

    def tokenize_text_auto_lm(self, text):
        if self.use_ftfy:
            text = ftfy.fix_text(text, normalization=self.ftfy_normalizer)
        if self.wikitext_detokenize:
            text = wikitext_detokenizer(text)

        # tokenize text
        if self.split_text_to_tokenize:
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
    num_features = 3

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

        self.prompt_key = params["dataset"].pop("prompt_key", None)
        assert (
            "jsonl_key" not in params["dataset"]
        ), "Jsonl key cannot be provided for SummarizationPreprocessor"
        self.completion_key = params["dataset"].pop("completion_key", None)
        if self.prompt_key is not None and self.completion_key is not None:
            assert (
                "multi_turn_key" not in params["dataset"]
            ), "Must specify either both prompt + completion, or only multi_turn"
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
        self.multi_turn_key = None
        self.multi_turn_content_key = None

    def check_valid_doc(self, doc):
        if self.multi_turn_key and self.multi_turn_content_key:
            if self.multi_turn_content_key not in doc[self.multi_turn_key][0]:
                logger.warning(
                    "multi_turn_content_key not in file, file may be corrupted"
                )
                raise ContinueLoopException
        else:
            if self.prompt_key not in doc or self.completion_key not in doc:
                logger.warning(
                    "prompt_key or completion_key not in file, file may be corrupted"
                )
                raise ContinueLoopException

    def clean_text(self, prompt, completion):
        if self.use_ftfy:
            prompt = ftfy.fix_text(prompt, normalization=self.ftfy_normalizer)
            completion = ftfy.fix_text(
                completion, normalization=self.ftfy_normalizer
            )
        if self.wikitext_detokenize:
            prompt = wikitext_detokenizer(prompt)
            completion = wikitext_detokenizer(completion)
        return prompt, completion

    def get_tokenizable_columns(self):
        if self.multi_turn_key:
            return {}
        else:
            return {
                'prompt_key': self.prompt_key,
                'completion_key': self.completion_key,
            }

    def parse_doc(self, doc):
        # Case for multi-turn dialogue
        if hasattr(self, "multi_turn_key") and self.multi_turn_key:
            doc = doc[self.multi_turn_key]
            assert (
                len(doc) % 2 == 0
            ), "We assume that every prompt has a response"
            if (
                self.multi_turn_content_key
            ):  # allow list of strs or list of dicts
                doc = [x[self.multi_turn_content_key] for x in doc]
            prompt_comp_pairs = [
                (str(doc[i]), str(doc[i + 1])) for i in range(0, len(doc), 2)
            ]

        # Case for single prompt-completion tasks
        else:
            prompt = str(doc[self.prompt_key])
            completion = str(doc[self.completion_key])
            prompt_comp_pairs = [(prompt, completion)]

        return DocObject(
            prompt_comp_pairs=prompt_comp_pairs,
            multi_modal=False,
            img_path=None,
        )

    def file_read_generator(self, file):
        tokenizable_columns = self.get_tokenizable_columns()
        multi_turn_flag = self.multi_turn_key is not None
        reader = Reader(file, tokenizable_columns, multi_turn=multi_turn_flag)

        for doc in reader.stream_data():
            # the exception allows a `continue` from a function
            try:
                self.check_valid_doc(doc)
            except ContinueLoopException:
                self.discarded_files += 1
                continue

            doc_obj = self.parse_doc(doc)

            for prompt, completion in doc_obj.prompt_comp_pairs:
                self.raw_chars_count += len(prompt) + len(completion)
                self.raw_bytes_count += len(prompt.encode("utf-8")) + len(
                    completion.encode("utf-8")
                )
            yield doc_obj

    def preprocessing_generator(self, doc_obj):
        for prompt, completion in doc_obj.prompt_comp_pairs:
            prompt, completion = self.clean_text(prompt, completion)
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
    num_features = 3

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
    num_features = 5
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
    """
    self.chunk_lengths stores List(List(Tuple))
    The outer list is chunks, inner list is sequences, tuples are prompt + completion pairs
    """

    num_features = 5
    use_vsl = True

    def __init__(self, params):
        super(VSLSummarizationPreprocessor, self).__init__(params)
        self.position_ids_dtype = params["dataset"].pop(
            "position_ids_dtype", "int32"
        )

        self.handle_default_add_bos = False
        if (
            hasattr(self.tokenizer, "add_bos_token")
            and self.tokenizer.add_bos_token
        ):
            self.handle_default_add_bos = True

        self.prompt_prefix = params["dataset"].pop("prompt_prefix", None)
        self.completion_prefix = params["dataset"].pop(
            "completion_prefix", None
        )
        self.multi_turn_key = params["dataset"].pop("multi_turn_key", None)
        self.multi_turn_content_key = params["dataset"].pop(
            "multi_turn_content_key", None
        )
        self.eos_after_prompt = params["dataset"].pop("eos_after_prompt", False)

        self.bos_id = None
        self.chunk_lengths = []
        self.tokenized_chunks = []
        self.chunk_count = 0
        self.init_prefix_toks()

    def init_prefix_toks(self):
        if self.prompt_prefix is not None:
            self.prompt_prefix_toks = self.tokenizer.encode(self.prompt_prefix)
            self.prompt_prefix_toks_len = len(self.prompt_prefix_toks)
            if self.handle_default_add_bos:
                self.prompt_prefix_toks_len -= 1
        else:
            self.prompt_prefix_toks_len = 0

        if self.completion_prefix is not None:
            self.comp_prefix_toks = self.tokenizer.encode(
                self.completion_prefix
            )
            if self.handle_default_add_bos:
                self.comp_prefix_toks = self.comp_prefix_toks[1:]
            self.comp_prefix_toks_len = len(self.comp_prefix_toks)
        else:
            self.comp_prefix_toks_len = 0

    def process_default_bos_token(self, prompt_ids, completion_ids, i):
        if self.handle_default_add_bos:
            if i > 0:
                if self.prompt_prefix:
                    prompt_ids = self.prompt_prefix_toks[1:] + prompt_ids[1:]
                else:
                    prompt_ids = prompt_ids[1:]
            else:
                if self.prompt_prefix:
                    prompt_ids = self.prompt_prefix_toks + prompt_ids[1:]
            self.bos_id = completion_ids[0]
            if self.completion_prefix:
                completion_ids = self.comp_prefix_toks + completion_ids[1:]
            else:
                completion_ids = completion_ids[1:]
        return prompt_ids, completion_ids

    def process_doc(self, doc_obj):
        total_len = 0
        tokens = []

        for i, (prompt, completion) in enumerate(doc_obj.prompt_comp_pairs):
            prompt, completion = self.clean_text(prompt, completion)

            prompt_ids = self.tokenizer.encode(prompt)
            completion_ids = self.tokenizer.encode(completion)
            prompt_ids, completion_ids = self.process_default_bos_token(
                prompt_ids, completion_ids, i
            )

            total_len += (
                len(prompt_ids)
                + len(completion_ids)
                + int(self.eos_after_prompt)
            )
            total_len += 1  # for internal eos tokens
            tokens.append((prompt_ids, completion_ids))

            if self.sep_id is not None:
                total_len += 1

        total_len -= 1  # but we will remove the last eos token to create input/label pairs
        doc_obj.tokens = tokens

        return total_len

    def vsl_pack(self, doc_obj):
        """
        Handles the packing of sequences together based on their length. Relies
        on self.process_doc to calculate the lengths
        """
        total_len = self.process_doc(doc_obj)

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
                self.tokenized_chunks[idx].append(doc_obj)
                self.chunk_lengths[idx] -= total_len
                create_new_chunk = False
                break

        if create_new_chunk:
            self.chunk_lengths.append(self.max_seq_length - total_len)
            self.tokenized_chunks.append([doc_obj])
            self.chunk_count += 1

    def vsl_sample_generator(self, generation_len):
        total = 0
        for _ in range(generation_len):
            doc_obj_pack = self.tokenized_chunks.pop(0)
            _ = self.chunk_lengths.pop(0)
            self.chunk_count -= 1

            total += 1
            tokens = create_features_summarization_vsl(
                doc_obj_pack,
                self.max_seq_length,
                self.comp_prefix_toks_len,
                pad_id=self.pad_id,
                eos_id=self.eos_id,
                eos_after_prompt=self.eos_after_prompt,
                sep_id=self.sep_id,
                inverted_mask=self.inverted_mask,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
                attention_span_dtype=self.position_ids_dtype,
                position_ids_dtype=self.position_ids_dtype,
            )
            if tokens == []:
                self.discarded_files += 1
                continue

            yield tokens

    def preprocessing_generator(self, doc):
        self.vsl_pack(doc)

        if self.chunk_count > self.files_per_record:
            for sample in self.vsl_sample_generator(self.files_per_record):
                yield sample
        else:
            yield []


class LlavaBasePreprocessor(SummarizationPreprocessor):
    num_features = 4

    def __init__(self, params):
        super(LlavaBasePreprocessor, self).__init__(params)
        self.multi_turn_key = params["dataset"].pop("multi_turn_key", None)
        self.multi_turn_content_key = params["dataset"].pop(
            "multi_turn_content_key", None
        )
        self.eos_after_prompt = params["dataset"].pop("eos_after_prompt", False)
        self.image_key = params["dataset"].pop("image_key", None)
        self.multi_modal_non_image_ex_key = params["dataset"].pop(
            "multi_modal_non_image_ex_key", None
        )
        self.image_token = params["dataset"].pop("image_token", None)
        self.num_patches = params["dataset"].pop("num_patches", None)
        self.image_dir = params["dataset"].pop("image_dir", None)
        self.bos_id = None
        self.handle_default_add_bos = False
        if (
            hasattr(self.tokenizer, "add_bos_token")
            and self.tokenizer.add_bos_token
        ):
            self.handle_default_add_bos = True
        self.multi_modal_remove_text_prompt = False
        self.multimodal_preprocessor = True

        self.image_patch_start_idx = None

        from transformers import LlamaTokenizer, LlamaTokenizerFast

        if not (
            isinstance(self.tokenizer, LlamaTokenizerFast)
            or isinstance(self.tokenizer, LlamaTokenizer)
        ):
            raise ValueError(
                "Currently we only support models based on Mistral and LLama-2 "
                "for LLaVA. We also only support this within the "
                "HuggingFaceTokenizer flow within the hdf5_preprocessing (as "
                "opposed to NeoXTokenizer etc)."
            )

    def get_tokenizable_columns(self):
        return {}

    def collect_image_patch_start_idx(self, tokens, img_path):
        # Take the first multi-modal example we see, and collect the index
        # where the image-patches are. This is needed for the model-config
        # so we write it to the output-dir
        if (self.image_patch_start_idx is None) and (img_path is not None):
            input_ids = tokens[0]
            equal_to_pad_idxs = np.where(input_ids == self.pad_id)[
                0
            ]  # have to index into np.where
            self.image_patch_start_idx = equal_to_pad_idxs[0]
        else:
            pass

    def check_valid_doc(self, doc):
        if self.multi_turn_key and self.multi_turn_content_key:
            if self.multi_turn_content_key not in doc[self.multi_turn_key][0]:
                logger.warning(
                    "multi_turn_content_key not in file, file may be corrupted"
                )
                raise ContinueLoopException
        else:
            if self.prompt_key not in doc or self.completion_key not in doc:
                logger.warning(
                    "prompt_key or completion_key not in file, file may be corrupted"
                )
                raise ContinueLoopException

        if self.image_key:
            if (
                self.image_key in doc
                and not self.multi_modal_non_image_ex_key in doc
            ):
                path = os.path.join(self.image_dir, doc[self.image_key])
                if not os.path.exists(path):
                    raise ContinueLoopException

    def parse_doc(self, doc):
        multi_modal = False
        img_path = None
        if hasattr(self, "image_key") and self.image_key:
            multi_modal = True
            # We don't always have an image; in LLaVA Phase 2 there are
            # text-only multi-turn dialogue examples
            if self.image_key in doc:
                img_path = doc[self.image_key]
            else:
                if (
                    hasattr(self, "multi_modal_non_image_ex_key")
                    and self.multi_modal_non_image_ex_key is not None
                ):
                    assert self.multi_modal_non_image_ex_key in doc
        # Case for multi-turn dialogue
        if hasattr(self, "multi_turn_key") and self.multi_turn_key:
            doc = doc[self.multi_turn_key]
            assert (
                len(doc) % 2 == 0
            ), "We assume that every prompt has a response"
            if (
                self.multi_turn_content_key
            ):  # allow list of strs or list of dicts
                doc = [x[self.multi_turn_content_key] for x in doc]
            prompt_comp_pairs = [
                (doc[i], doc[i + 1]) for i in range(0, len(doc), 2)
            ]

        # Case for single prompt-completion tasks
        else:
            prompt = doc[self.prompt_key]
            completion = doc[self.completion_key]
            prompt_comp_pairs = [(prompt, completion)]

        return DocObject(
            prompt_comp_pairs=prompt_comp_pairs,
            multi_modal=multi_modal,
            img_path=img_path,
        )

    def process_doc(self, doc_obj):
        tokens = []

        for i, (prompt, completion) in enumerate(doc_obj.prompt_comp_pairs):
            # in multi_modal examples the prompt will have something like:
            # "<image>\nWhat is the main color of the vase in the image?"
            # but we don't want to include <image> in the text prompt
            if doc_obj.img_path is not None:
                prompt = prompt.replace(self.image_token, "")

            prompt, completion = self.clean_text(prompt, completion)

            prompt_ids = self.tokenizer.encode(prompt)
            completion_ids = self.tokenizer.encode(completion)
            prompt_ids, completion_ids = self.process_default_bos_token(
                prompt_ids, completion_ids, i, doc_obj.multi_modal
            )
            if self.multi_modal_remove_text_prompt:
                prompt_ids = []
            tokens.append((prompt_ids, completion_ids))
        doc_obj.tokens = tokens


class LlavaPhaseOnePreprocessor(LlavaBasePreprocessor):
    def __init__(self, params):
        super(LlavaPhaseOnePreprocessor, self).__init__(params)
        self.multi_modal_remove_text_prompt = True

    def preprocessing_generator(self, doc_obj):
        self.process_doc(doc_obj)
        tokens = create_features_llava_phase1(
            doc_obj,
            self.max_seq_length,
            self.num_patches,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            bos_id=self.bos_id,
            eos_after_prompt=self.eos_after_prompt,
            sep_id=self.sep_id,
            inverted_mask=self.inverted_mask,
            handle_default_bos_token=self.handle_default_add_bos,
            input_ids_dtype=self.input_ids_dtype,
            input_mask_dtype=self.input_mask_dtype,
            labels_dtype=self.input_ids_dtype,
        )
        if tokens == []:
            self.discarded_files += 1
            yield []

        else:
            img_path = [doc_obj.img_path]
            # First time we have a multi-modal example, we find the index
            # where image-patches begin and write this to a file in the
            # output-dir. This is used in the model config
            self.collect_image_patch_start_idx(tokens, doc_obj.img_path)
            yield (img_path, tokens)

    def process_default_bos_token(
        self, prompt_ids, completion_ids, i, is_multi_modal
    ):
        if is_multi_modal or (self.handle_default_add_bos and i > 0):
            prompt_ids = prompt_ids[1:]
        if self.handle_default_add_bos:
            self.bos_id = completion_ids[0]
            completion_ids = completion_ids[1:]
        return prompt_ids, completion_ids


class LlavaPhaseTwoPreprocessor(LlavaBasePreprocessor):
    def __init__(self, params):
        self.multi_modal_remove_text_prompt = False
        self.prompt_prefix = params["dataset"].pop("prompt_prefix", None)
        self.completion_prefix = params["dataset"].pop(
            "completion_prefix", None
        )
        super(LlavaPhaseTwoPreprocessor, self).__init__(params)
        self.system_prompt_style = params["dataset"].pop(
            "system_prompt_style", None
        )
        self.init_template_toks()

        self.space_id = self.tokenizer.convert_tokens_to_ids("▁")

    def init_template_toks(self):
        if self.prompt_prefix is not None:
            self.prompt_prefix_toks = self.tokenizer.encode(self.prompt_prefix)
            self.prompt_prefix_toks_len = len(self.prompt_prefix_toks)
            if self.handle_default_add_bos:
                self.prompt_prefix_toks_len -= 1
        else:
            self.prompt_prefix_toks_len = 0

        if self.completion_prefix is not None:
            self.comp_prefix_toks = self.tokenizer.encode(
                self.completion_prefix
            )
            if self.handle_default_add_bos:
                self.comp_prefix_toks = self.comp_prefix_toks[1:]
            self.comp_prefix_toks_len = len(self.comp_prefix_toks)
        else:
            self.comp_prefix_toks_len = 0
        if self.system_prompt_style is not None:
            system_prompt_text = SYSTEM_PROMPT_REGISTRY[
                self.system_prompt_style
            ]
            self.system_prompt_toks = self.tokenizer.encode(system_prompt_text)
            self.system_prompt_len = len(self.system_prompt_toks)
        else:
            self.system_prompt_len = 0
            self.system_prompt_toks = []

    def process_default_bos_token(
        self, prompt_ids, completion_ids, i, is_multi_modal
    ):
        if self.handle_default_add_bos:
            prompt_ids = prompt_ids[1:]
            completion_ids = completion_ids[1:]
        if self.prompt_prefix is not None:
            if self.handle_default_add_bos:
                prompt_ids = self.prompt_prefix_toks[1:] + prompt_ids
            else:
                prompt_ids = self.prompt_prefix_toks + prompt_ids
        if self.completion_prefix is not None:
            completion_ids = self.comp_prefix_toks + completion_ids
        return prompt_ids, completion_ids

    def preprocessing_generator(self, doc_obj):
        self.process_doc(doc_obj)

        tokens = create_features_llava_phase2(
            doc_obj,
            self.system_prompt_toks,
            self.max_seq_length,
            self.num_patches,
            self.prompt_prefix_toks_len,
            self.comp_prefix_toks_len,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            space_id=self.space_id,
            eos_after_prompt=self.eos_after_prompt,
            sep_id=self.sep_id,
            inverted_mask=self.inverted_mask,
            handle_default_bos_token=self.handle_default_add_bos,
            input_ids_dtype=self.input_ids_dtype,
            input_mask_dtype=self.input_mask_dtype,
            labels_dtype=self.input_ids_dtype,
        )
        if tokens == []:
            self.discarded_files += 1
            yield []

        else:
            img_path = [doc_obj.img_path]
            # First time we have a multi-modal example, we find the index
            # where image-patches begin and write this to a file in the
            # output-dir. This is used in the model config
            self.collect_image_patch_start_idx(tokens, doc_obj.img_path)
            yield (img_path, tokens)
