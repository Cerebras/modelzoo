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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from modelzoo.common.pytorch.input.utils import SamplesSaver, SamplesViewer
from modelzoo.common.pytorch.input_utils import get_streaming_batch_size


class InferenceDataProcessor(torch.utils.data.IterableDataset):
    def __init__(self, params, samples_file_list, dataset_size):
        super().__init__()

        self.batch_size = get_streaming_batch_size(params["batch_size"])
        self.num_workers = params.get("num_workers", 0)
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

        # Features in HDF5 files
        self.features_list = [
            "input_ids",
            "continuation",
            "attention_mask",
            "labels",
        ]
        for samples_file_path in samples_file_list:
            if not os.path.isfile(samples_file_path):
                raise ValueError(
                    f"Samples file path is invalid: {samples_file_path}"
                )
        self.samples_iter = SamplesViewer(samples_file_list)
        self.dataset_size = dataset_size

    @staticmethod
    def gen_data_samples(
        requests: List[Tuple[str, str]],
        batch_size: int,
        max_sequence_length: int,
        eos_token_id: int,
        samples_saver: SamplesSaver,
        tokenizer_file_path: Optional[str] = None,
    ) -> Tuple[List[str], int, Tuple[int, int]]:
        """Preprocess raw text requests as fetched from
        EEH script into data samples consumable by GPT2
        model and dump these to numpy file.

        Args:
            requests: List of raw text requests with each
                request captured as a tuple pair of context
                string and continuation string
            max_sequence_length: The maximum length of each
                sample
            batch_size: The batch size
            eos_token_id: int representing the end-of-sentence
                token
            samples_saver: `SamplesSaver` object to manage the
                saving of data samples to file.
            tokenizer_file_path: Path to the tokenizer file if
                for a custom tokenizer is used. If not specified,
                `gpt2` tokenizer is used by default.

        Returns:
            (List[str], int, tuple) tuple of list of file paths where the
            samples are dumped, the size of the dataset (total no. of samples),
            and tuple of context and continuation token lengths
        """
        if tokenizer_file_path is not None:
            tokenizer = Tokenizer.from_file(tokenizer_file_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if hasattr(tokenizer, 'eos_token_id'):
            eos_token_id = tokenizer.eos_token_id

        requests_len = len(requests)
        token_lengths = []

        for context, continuation in tqdm(requests):
            ## Step 1: Tokenize requests
            context_enc, continuation_enc = _encode_pair(
                context, continuation, tokenizer, eos_token_id
            )

            # FROM EEH script:
            # https://github.com/EleutherAI/lm-evaluation-harness/blob/62ca18400ebe0fc0dbb14274b27170d2d5ae9e3d/lm_eval/base.py#L253-L255
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
                len(context_enc)
                - 1 : len(context_enc)
                + len(continuation_enc)
                - 1
            ] = continuation_enc
            # attention_mask
            sample_full[2][
                len(context_enc)
                - 1 : len(context_enc)
                + len(continuation_enc)
                - 1
            ] = 1
            # label_ids
            sample_full[3][: len(label_ids)] = label_ids + [eos_token_id]

            # Add the data sample to the `SamplesSaver` object
            samples_saver.add_sample(sample_full)
            token_lengths.append((len(context_enc), len(continuation_enc)))

        # Ensure that requests is a multiple of batch size
        # by padding remainder samples with zeros
        if requests_len % batch_size != 0:
            num_padding_sequences = batch_size - (requests_len % batch_size)
            for _ in range(num_padding_sequences):
                samples_saver.add_sample(
                    np.zeros((4, max_sequence_length), dtype=np.int32)
                )
                token_lengths.append((0, 0))

        ## Step 3: `add_sample` saves numpy array samples to file
        ## so these can be loaded by input generating workers. The
        ## `flush` method saves any remaining data samples to file.
        samples_saver.flush()

        return (
            samples_saver.samples_files,
            samples_saver.dataset_size,
            token_lengths,
        )

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        for sample in self.samples_iter:
            yield {
                feature: sample[i]
                for i, feature in enumerate(self.features_list)
            }

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

        return dataloader


def _encode_pair(
    context: str,
    continuation: str,
    tokenizer: Union[Tokenizer, PreTrainedTokenizerBase],
    eos_token_id: int,
) -> Tuple[List[int], List[int]]:
    """Encodes a pair of context and continuation strings
    using the specified tokenizer.
    This is an implementation from: 
    https://github.com/EleutherAI/lm-evaluation-harness/blob/008fc2a23245c40384f2312718433eeb1e0f87a9/lm_eval/base.py#L200

    Args:
        context (str): Context string for a given task.
        continuation (str): Continuation string for a given task.
        tokenizer (Tokenizer): Tokenizer class from huggingface tokenizers
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


def get_token_ids(
    text: str, tokenizer: Union[Tokenizer, PreTrainedTokenizerBase]
) -> List[int]:
    """Get encoded token ids from a string using the specified tokenizer.

    Args:
        text (str): The input string.
        tokenizer (Tokenizer): Tokenizer class from huggingface tokenizers
            library.

    Returns:
        List[int]: List of token ids.
    """
    encoded_ids = tokenizer.encode(text, add_special_tokens=False)

    if not isinstance(encoded_ids, list):
        # depending on the tokenizer instance from pre-trained or file,
        # the encoded_ids can be a list or an Encoding instance.
        encoded_ids = encoded_ids.ids
    return encoded_ids
