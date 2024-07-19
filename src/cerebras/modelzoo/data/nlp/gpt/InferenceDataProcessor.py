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
from abc import abstractmethod
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.common.utils.input.utils import SamplesSaver

# RequestType defines how we preprocess task data samples.
# For Eleuther Eval Harness (EEH),
# we have generative tasks ("generate_until") where we tokenize the context string only
# and extract its length and the stop token words as metadata;
# whereas for nongenerative tasks ("loglikehood"), we tokenize both the context and continuation
# strings and extract the tokenized lengths as metadata for postprocessing
# https://github.com/EleutherAI/lm-evaluation-harness/blob/65b8761db922513dada0320b860fabb1b4f01dc3/lm_eval/api/instance.py#L7
#
# For BigCode Eval Harness,
# we specify "bigcode_eh" as the request type and return the sample idx
# and prompt encoding length as metadata
RequestType = IntEnum(
    "RequestType", ["eeh_loglikelihood", "eeh_generate_until", "bigcode_eh"]
)


class EvalHarnessDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples_file_list: List[Tuple[str, int]],
        dataset_size: int,
    ):
        super().__init__()

        if not samples_file_list:
            raise RuntimeError(
                "No samples files to load. Please provide a list of "
                "valid paths to .npy files to load Eval Harness data samples."
            )

        self._samples_file_paths = []  # Paths of chunked `.npy` files
        self._num_chunked_samples = []  # Num samples in chunked files

        for samples_file_path, samples_len in samples_file_list:
            if not os.path.isfile(samples_file_path):
                raise ValueError(
                    f"Samples file path is invalid: {samples_file_path}"
                )
            self._samples_file_paths.append(samples_file_path)
            self._num_chunked_samples.append(samples_len)

        self._dataset_size = dataset_size
        self._chunk_idx = 0
        self._cumulative_num_samples = np.cumsum(self._num_chunked_samples)
        self._prev_chunk_len = 0
        self._samples = None

        self.map_fn = None

    def map(self, fn):
        if self.map_fn is not None:
            raise ValueError(
                f"You may only apply one map function to EvalHarnessDataset"
            )
        self.map_fn = fn

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, idx):
        if idx >= self._cumulative_num_samples[-1]:
            raise IndexError(
                f"Sample index {idx} is out of bounds for samples of size "
                f"{self._cumulative_num_samples[-1]}"
            )
        elif idx >= self._cumulative_num_samples[self._chunk_idx]:
            # Pick the correct chunked file that comprises the sample
            self._chunk_idx = np.searchsorted(self._cumulative_num_samples, idx)
            if idx == self._cumulative_num_samples[self._chunk_idx]:
                self._chunk_idx += 1
            self._prev_chunk_len = self._cumulative_num_samples[self._chunk_idx]
            self._samples = None

        if self._samples is None:
            samples_file = self._samples_file_paths[self._chunk_idx]
            try:
                with open(samples_file, 'rb') as f:
                    self._samples = np.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read chunked samples file: {samples_file}"
                ) from e

        sample_idx = idx - self._prev_chunk_len
        sample = self._samples[sample_idx]

        if self.map_fn is not None:
            return self.map_fn(sample)
        return sample


@registry.register_datasetprocessor("InferenceDataProcessor")
class InferenceDataProcessor:
    def __init__(self, params, samples_file_list, dataset_size):
        super().__init__()

        self.batch_size = params["batch_size"]
        self.num_workers = params.get("num_workers", 0)
        if self.num_workers is not None and self.num_workers > 1:
            raise ValueError(
                "Eval harness does not support multiple process data "
                "loading for `num_workers` > 1, but specified "
                f"{self.num_workers} worker processes.\nPlease ensure that "
                "`num_workers` is either 0 (default) or 1."
            )
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)
        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False

        # NOTE: `drop_last` shouldn't have any impact since we're padding
        # the inputs with zero tensors to make sure these are a multiple
        # of `batch_size` because, similar to EEH's flow, we process all
        # input requests for the user-specified evaluation tasks
        self.drop_last = params.get("drop_last", True)

        self.dataset = EvalHarnessDataset(samples_file_list, dataset_size)
        self.sampler = cstorch.utils.data.DistributedSampler(
            data_source=self.dataset,
            shuffle=False,
            shard=True,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_samples=dataset_size,
        )

    @classmethod
    def from_request_type(
        cls,
        request_type: RequestType,
        params: Dict[str, Any],
        samples_file_list: List[str],
        dataset_size: int,
    ) -> "InferenceDataProcessor":
        if request_type == RequestType.eeh_loglikelihood.value:
            return InferenceDataProcessorLL(
                params, samples_file_list, dataset_size
            )
        elif request_type == RequestType.eeh_generate_until.value:
            return InferenceDataProcessorGU(
                params, samples_file_list, dataset_size
            )
        elif request_type == RequestType.bigcode_eh.value:
            return InferenceDataProcessorBCEH(
                params, samples_file_list, dataset_size
            )
        else:
            raise TypeError(
                f"Invalid request type: {request_type}. At present, only "
                "`RequestType.eeh_loglikelihood`, `RequestType.eeh_generate_until` "
                "and `RequestType.bigcode_eh`request types are supported."
            )

    @staticmethod
    @abstractmethod
    def _create_data_sample(
        request,
        max_sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
        eos_token_id: int,
        inf_start_token: Optional[int] = None,
        max_gen_tokens: Optional[int] = None,
        padded_sample: bool = False,
        sample_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Creates np data sample from the given raw text request. This helper
        is called by `gen_data_samples` and it specifies data sample creation for
        the different request types. Subclasses of `InferenceDataProcessor` override
        this method to define how each sample is constructed from a given request.
        """

    @staticmethod
    def gen_data_samples(
        requests: List,
        batch_size: int,
        max_sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
        eos_token_id: int,
        samples_saver: SamplesSaver,
        request_type: RequestType,
        inf_start_token: Optional[int] = None,
        max_gen_tokens: Optional[int] = None,
    ) -> Tuple[List[str], int, List[Tuple[int, int]]]:
        """Preprocess raw text requests as fetched from
        EEH script into data samples consumable by GPT2
        model and dump these to numpy file.

        Args:
            requests: List of EEH's Instance dataclass objects
                holding raw text data
            batch_size: The batch size
            max_sequence_length: The maximum length of each
                sample
            tokenizer: The tokenizer used to tokenize raw text data
            eos_token_id: int representing the end-of-sentence
                token
            samples_saver: `SamplesSaver` object to manage the
                saving of data samples to file.
            request_type: The type of request for which the data sample
                is to be created
            inf_start_token: (generative tasks-only) int representing
                the start token for generative inference
            max_gen_tokens: (generative tasks-only) The max number of
                tokens to generate

        Returns:
            (List[str], int, tuple) tuple of
            - list of file paths where the samples are dumped;
            - int representing the size of the dataset (total no. of samples;
            - tuple of request metadata needed for EEH postprocessing.
        """
        is_generative = request_type == RequestType.eeh_generate_until
        is_bceh = request_type == RequestType.bigcode_eh
        if (is_bceh or is_generative) and (
            inf_start_token is None or max_gen_tokens is None
        ):
            raise RuntimeError(
                "Some inference settings are missing. Please ensure that "
                "`start_token` and `max_tokens` are specified in the "
                "model params for generative inference tasks."
            )

        if is_bceh:
            data_sample_fn = InferenceDataProcessorBCEH._create_data_sample
        elif is_generative:
            data_sample_fn = InferenceDataProcessorGU._create_data_sample
        else:
            data_sample_fn = InferenceDataProcessorLL._create_data_sample

        requests_len = len(requests)
        requests_metadata = []

        ## Generate data samples from request
        if is_bceh:
            requests_list = requests
        else:
            requests_list = [request.args for request in requests]
        for idx, request in tqdm(enumerate(requests_list)):
            sample, metadata = data_sample_fn(
                request,
                max_sequence_length=max_sequence_length,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                inf_start_token=inf_start_token,
                max_gen_tokens=max_gen_tokens,
                sample_idx=idx,
            )
            # Add the data sample to the `SamplesSaver` object
            samples_saver.add_sample(sample)
            requests_metadata.append(metadata)

        # Ensure that requests is a multiple of batch size
        # by padding remainder samples with zeros
        if requests_len % batch_size != 0:
            num_padding_sequences = batch_size - (requests_len % batch_size)
            for _ in range(num_padding_sequences):
                dummy_sample, metadata = data_sample_fn(
                    (),
                    max_sequence_length,
                    tokenizer,
                    eos_token_id,
                    padded_sample=True,
                )
                samples_saver.add_sample(dummy_sample)
                requests_metadata.append(metadata)

        ## Step 3: `add_sample` saves numpy array samples to file
        ## so these can be loaded by input generating workers. The
        ## `flush` method saves any remaining data samples to file.
        samples_saver.flush()

        return (
            samples_saver.samples_files,
            samples_saver.dataset_size,
            requests_metadata,
        )

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

        return dataloader


@registry.register_datasetprocessor("InferenceDataProcessorLL")
class InferenceDataProcessorLL(InferenceDataProcessor):
    """Subclass for processing EEH `loglikelihood` requests."""

    def __init__(self, params, samples_file_list, dataset_size):
        super().__init__(params, samples_file_list, dataset_size)
        self.dataset.map(
            fn=lambda x: {
                f: x[i]
                for i, f in enumerate(
                    [
                        "input_ids",
                        "continuation",
                        "attention_mask",
                        "labels",
                    ]
                )
            }
        )

    @staticmethod
    def _create_data_sample(
        request,
        max_sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
        eos_token_id: int,
        inf_start_token: Optional[int] = None,
        max_gen_tokens: Optional[int] = None,
        padded_sample: bool = False,
        sample_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, tuple]:
        if padded_sample:
            return np.zeros((4, max_sequence_length), dtype=np.int32), (0, 0)

        context, continuation = request

        ## Step 1: Tokenize request
        context_enc, continuation_enc = _encode_pair(
            context, continuation, tokenizer, eos_token_id
        )

        # FROM EEH script:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/c9bbec6e7de418b9082379da82797522eb173054/lm_eval/models/huggingface.py#L706-L709
        # sanity check
        if not len(context_enc) > 0:
            raise RuntimeError(f"Failed to tokenize input `{context}`")
        if not len(continuation_enc) > 0:
            raise RuntimeError(f"Failed to tokenize input `{continuation}`")

        # Hard failure similar to EEH's assertion below:
        # assert len(continuation_enc) <= self.max_length
        # if samples' context cannot be captured
        # Subtracting 1 from msl to account for EOS token at
        # the end of the input
        if len(continuation_enc) >= max_sequence_length - 1:
            raise RuntimeError(
                f"Continuation enconding length {len(continuation_enc)} "
                f"is longer than msl of {max_sequence_length}. Please choose "
                "a larger msl or consider skipping eval for this task."
            )

        # Truncate context from the left if the input (context_enc + cont_enc) len
        # exceeds maximum sequence length
        if len(context_enc) + len(continuation_enc) >= max_sequence_length:
            context_enc = context_enc[
                -(max_sequence_length - 1 - len(continuation_enc)) :
            ]

        ## Step 2: Preprocess tokenized requests to create data samples
        sample = np.array((context_enc + continuation_enc), dtype=np.int32)
        input_ids = sample[:-1]
        label_ids = sample[1:]

        # Cast the requests to this format [(input_ids, continuation_ids, mask, labels)]
        # Currently we only have input_ids
        sample_full = np.zeros((4, max_sequence_length), dtype=np.int32)
        # input_ids
        sample_full[0][: len(input_ids)] = input_ids
        # continuation_ids
        sample_full[1][
            len(context_enc) - 1 : len(context_enc) + len(continuation_enc) - 1
        ] = continuation_enc
        # attention_mask
        sample_full[2][
            len(context_enc) - 1 : len(context_enc) + len(continuation_enc) - 1
        ] = 1
        # label_ids
        sample_full[3][: len(label_ids)] = label_ids + [eos_token_id]

        # Return sample and the lengths of context & continuation tokens as metadata
        return sample_full, (len(context_enc), len(continuation_enc))


@registry.register_datasetprocessor("InferenceDataProcessorGU")
class InferenceDataProcessorGU(InferenceDataProcessor):
    """Subclass for processing EEH `generate_until` requests."""

    def __init__(self, params, samples_file_list, dataset_size):
        super().__init__(params, samples_file_list, dataset_size)
        self.dataset.map(
            fn=lambda x: {f: x[i] for i, f in enumerate(["input_ids"])}
        )

    @staticmethod
    def _create_data_sample(
        request,
        max_sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
        eos_token_id: int,
        inf_start_token: Optional[int] = None,
        max_gen_tokens: Optional[int] = None,
        padded_sample: bool = False,
        sample_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, tuple]:
        if padded_sample:
            return np.zeros((1, max_sequence_length), dtype=np.int32), ()

        context, until = request
        until = until["until"]

        if isinstance(until, str):
            until = [until]

        # Add eos token to until (stop words)
        eos_str = tokenizer.decode([eos_token_id], skip_special_tokens=False)
        until.append(eos_str)

        context_enc = get_token_ids(context, tokenizer)

        # Truncate context so that generation fits within msl
        context_enc = context_enc[-(max_sequence_length - max_gen_tokens) :]
        input_ids = np.array(context_enc, dtype=np.int32)

        sample_full = np.full(
            shape=(1, max_sequence_length),
            fill_value=inf_start_token,
            dtype=np.int32,
        )

        sample_full[0][: len(input_ids)] = input_ids

        # Return sample and (until tokens, ctx length) as metadata for generative tasks
        return sample_full, (until, len(context_enc))


@registry.register_datasetprocessor("InferenceDataProcessorBCEH")
class InferenceDataProcessorBCEH(InferenceDataProcessor):
    """Subclass for processing BigCode data, i.e. `bigcode_eh` requests."""

    def __init__(self, params, samples_file_list, dataset_size):
        super().__init__(params, samples_file_list, dataset_size)
        self.dataset.map(
            fn=lambda x: {f: x[i] for i, f in enumerate(["input_ids"])}
        )

    @staticmethod
    def _create_data_sample(
        request,
        max_sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
        eos_token_id: int,
        inf_start_token: Optional[int] = None,
        max_gen_tokens: Optional[int] = None,
        padded_sample: bool = False,
        sample_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, tuple]:
        if padded_sample:
            return np.zeros((1, max_sequence_length), dtype=np.int32), ()

        prompt_enc = get_token_ids(request, tokenizer)

        # Truncate context so that generation fits within msl
        prompt_enc = prompt_enc[-(max_sequence_length - max_gen_tokens) :]
        input_ids = np.array(prompt_enc, dtype=np.int32)

        sample_full = np.full(
            shape=(1, max_sequence_length),
            fill_value=inf_start_token,
            dtype=np.int32,
        )

        sample_full[0][: len(input_ids)] = input_ids

        # Return sample and (sample_idx, prompt_enc) as metadata for bigcode generative tasks
        return sample_full, (sample_idx, len(prompt_enc))


def _encode_pair(
    context: str,
    continuation: str,
    tokenizer: PreTrainedTokenizerBase,
    eos_token_id: int,
) -> Tuple[List[int], List[int]]:
    """Encodes a pair of context and continuation strings
    using the specified tokenizer.
    This is an implementation from:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/c9bbec6e7de418b9082379da82797522eb173054/lm_eval/models/huggingface.py#L545

    Args:
        context (str): Context string for a given task.
        continuation (str): Continuation string for a given task.
        tokenizer (PreTrainedTokenizerBase): Tokenizer class from huggingface transformers
            library.
        eos_token_id (int): int representing the end-of-sentence token id.

    Returns:
        (List[int], List[int]): A tuple pair of context and continuation
            encodings.
    """
    if context == "":
        # end of text as context
        context_enc = [eos_token_id]
        continuation_enc = get_token_ids(continuation, tokenizer)
    else:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = get_token_ids(context + continuation, tokenizer)
        context_enc = get_token_ids(context, tokenizer)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

    return context_enc, continuation_enc


def get_token_ids(text: str, tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """Get encoded token ids from a string using the specified tokenizer.

    Args:
        text (str): The input string.
        tokenizer (PreTrainedTokenizerBase): Tokenizer class from huggingface transformers
            library.

    Returns:
        List[int]: List of token ids.
    """
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_stop_words(
    stop_words: List[str], tokenizer: PreTrainedTokenizerBase
) -> List[List[int]]:
    """Helper to construct a list of stop token sequences
    from the given list of stop words using the specified tokenizer.

    For stop words that tokenize to a single token, we iterate the tokenizer's
    vocab and add all the token_ids that detokenize to the stop word. This is done
    to handle the case where different token ids map to the same stop word,
    since RT uses stop tokens, not words to stop inferring.

    For stop words that tokenize to multiple token sequence, we add the sequence
    directly.

    Args:
        stop_words (str): The input string.
        tokenizer (PreTrainedTokenizerBase): Tokenizer class from huggingface transformers
            library.

    Returns:
        List[List[int]]: Sorted (by first token id) list of stop token sequences.
    """
    stop_sequences = []
    for stop_word in stop_words:
        stop_seq = get_token_ids(stop_word, tokenizer)
        found = False
        if len(stop_seq) == 1:
            for _, token_id in tokenizer.get_vocab().items():
                decoded = tokenizer.decode([token_id], skip_special_tokens=True)
                if decoded == stop_word:
                    found = True
                    stop_sequences.append([token_id])
        if not found:
            stop_sequences.append(stop_seq)
    return sorted(stop_sequences)
