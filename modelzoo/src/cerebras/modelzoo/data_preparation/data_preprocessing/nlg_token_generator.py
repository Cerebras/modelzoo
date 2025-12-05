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

import numpy as np


class NLGTokenGenerator:
    """Token Generator for NLG data sets such as E2E, DART, and WebNLG.
    Assumes the dataset has already been tokenized.
    Expect .jsonl input files that contains a "context" and a "completion" key.
    Used with GptHDF5DataProcessor.
    """

    def __init__(self, max_seq_length):

        self.max_seq_length = max_seq_length
        self.features = ["input_ids", "attention_mask", "labels"]

    def encode(self, semantic_data_array):

        context, completion = self.parse_semantic_data_array(
            semantic_data_array
        )

        raw_chars_count = 0  ## As the dataset is already tokenized into tokens. Raw dataset is not available.
        raw_bytes_count = 0  ## As the dataset is already tokenized into tokens. Raw dataset is not available.
        files_processed = 0
        discarded_files = 0
        normalized_chars_count = raw_chars_count
        normalized_bytes_count = raw_bytes_count

        input_ids = np.concatenate((context, completion[:-1]))
        labels = np.concatenate((context[1:], completion))

        num_pad_tokens = self.max_seq_length - len(input_ids)
        num_masked_tokens = self.max_seq_length - len(completion)
        input_ids = np.pad(input_ids, (0, self.max_seq_length - len(input_ids)))
        labels = np.pad(labels, (0, self.max_seq_length - len(labels)))
        indices = np.arange(self.max_seq_length)

        attention_mask = np.where(indices < len(context) - 1, 0, indices)
        attention_mask = np.where(
            attention_mask >= len(context) - 1 + len(completion),
            0,
            attention_mask,
        )
        attention_mask = np.where(attention_mask != 0, 1, 0)
        sample = np.stack([input_ids, attention_mask, labels]).reshape(
            1, 3, self.max_seq_length
        )
        loss_valid_tokens = int(attention_mask.sum())
        num_tokens = int(input_ids.shape[0])

        if sample.size == 0:
            discarded_files += 1

        files_processed += 1
        data_stats = {
            "discarded_files": discarded_files,
            "processed_files": files_processed,
            "successful_files": files_processed - discarded_files,
            "raw_chars_count": raw_chars_count,
            "raw_bytes_count": raw_bytes_count,
            "num_pad_tokens": num_pad_tokens,
            "num_masked_tokens": num_masked_tokens,
            "loss_valid_tokens": loss_valid_tokens,
            "num_tokens": num_tokens,
            "normalized_chars_count": normalized_chars_count,
            "normalized_bytes_count": normalized_bytes_count,
            "n_examples": successful,
        }
        data = {"data": sample}

        return data, data_stats

    def parse_semantic_data_array(self, semantic_data_array):

        context = semantic_data_array[0]['content'][0]['text']
        completion = semantic_data_array[1]['content'][0]['text']

        return context, completion
