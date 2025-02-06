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
Processor for PyTorch Transformer training.
"""

import csv
import os
from typing import Literal

import numpy as np

from cerebras.modelzoo.data.nlp.t5.t5_utils import (
    create_transformer_input_features,
    parse_text,
)
from cerebras.modelzoo.data.nlp.t5.T5DynamicDataProcessor import (
    T5DynamicDataProcessor,
    T5DynamicDataProcessorConfig,
)


class TransformerDynamicDataProcessorConfig(T5DynamicDataProcessorConfig):
    data_processor: Literal["TransformerDynamicDataProcessor"]


class TransformerDynamicDataProcessor(T5DynamicDataProcessor):
    """
    Reads text files containing the input text tokens.

    Args:
        config: The configuration object for the processor.
    """

    def __init__(self, config: TransformerDynamicDataProcessorConfig):
        super().__init__(config)

    def get_meta_data(self, data_dir):
        """
        Read data from meta files.
        :param str data_dir: Path to the input directory.
        :return: Processed meta data.
        """
        if not isinstance(data_dir, list):
            data_dir = [data_dir]

        meta_data = {}
        for file_name in data_dir:
            meta_file = os.path.join(file_name, "meta.dat")
            assert os.path.exists(
                meta_file
            ), f"Meta file is missing in the input directory: {data_dir}."

            with open(meta_file, "r") as fin:
                for line in fin.readlines():
                    line = line.strip().split()
                    # Format: `str_file_path tgt_file_path src_file_size`
                    meta_data[(line[0], line[1])] = int(line[2])
        return meta_data

    def load_buffer(self):
        """
        Generator to read the data in chunks of size of `data_buffer`.
        We read data from both source and target input datasets to
        prepare features for side by side translation task.
        :returns: Yields the data stored in the `data_buffer`.

        """
        self.processed_buffers = 0
        self.data_buffer = []
        while self.processed_buffers < len(self.text_files_per_task_per_worker):
            (
                current_file_path,
                num_examples,
                start_id,
            ) = self.text_files_per_task_per_worker[self.processed_buffers]

            src_file_path, tgt_file_path = current_file_path
            with open(src_file_path, "r", newline="") as src_fin:
                src_data_reader = csv.reader(src_fin, delimiter="\n")
                with open(tgt_file_path, "r", newline="") as tgt_fin:
                    tgt_data_reader = csv.reader(tgt_fin, delimiter="\n")

                    for row_id, (src_row, tgt_row) in enumerate(
                        zip(src_data_reader, tgt_data_reader)
                    ):
                        if start_id <= row_id < start_id + num_examples:
                            # Adding both source and target input sequences aligned.
                            self.data_buffer.append((src_row, tgt_row))
                        else:
                            continue

                        if len(self.data_buffer) == self.shuffle_buffer:
                            if self.shuffle:
                                self.rng.shuffle(self.data_buffer)

                            for ind in range(len(self.data_buffer)):
                                yield self.data_buffer[ind]
                            self.data_buffer = []

                    self.processed_buffers += 1

        if self.shuffle:
            self.rng.shuffle(self.data_buffer)

        for ind in range(len(self.data_buffer)):
            yield self.data_buffer[ind]
        self.data_buffer = []

    def get_single_item(self):
        """
        Iterating over the data to construct input features.

        :return: A dict with training features:
            * np.array[int.32] input_ids: Numpy array with encoder input token indices.
                Shape: (`src_max_sequence_length`).
            * np.array[int.32] decoder_input_ids: Numpy array with decoder input token indices.
                Shape: (`tgt_max_sequence_length`).
            * np.array[int.32] attention_mask: Numpy array with attention mask for encoder.
               Shape: (`src_max_sequence_length`).
            * np.array[int.32] decoder_attention_mask: Numpy array with attention mask for decoder.
               Shape: (`tgt_max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels for teacher forcing mode.
               Shape: (`tgt_max_sequence_length`).
        """
        # Iterate over the data rows to create input features.
        for data_row in self.load_buffer():
            # `data_row` is a tuple with source and target
            # input sequences.
            src_tokens = parse_text(
                "".join(data_row[0]), do_lower=self.do_lower
            )
            tgt_tokens = parse_text(
                "".join(data_row[1]), do_lower=self.do_lower
            )

            features = create_transformer_input_features(
                src_tokens,
                tgt_tokens,
                self.src_max_sequence_length,
                self.tgt_max_sequence_length,
                self.input_pad_id,
                self.attn_mask_pad_id,
                self.labels_pad_id,
                self.src_tokenize,
                self.special_tokens_indices["sos_token"],
                self.special_tokens_indices["eos_token"],
            )
            # This example might be filtered out based on the sequence length.
            # See `create_transformer_input_features` for more details.
            if features:
                yield features

    def element_length_fn(self, features):
        """
        Takes a single sample and returns the sequence length of that sample
        to be used for VTS bucketing.
        """
        return np.maximum(
            np.sum(features["attention_mask"]),
            np.sum(features["decoder_attention_mask"]),
        )

    def __len__(self):
        """
        Since samples are filtered by max_length, this will return an upper
        limit. See: transformers/pytorch/t5/input/utils.py
        create_transformer_input_features(...)
        """
        return self.num_examples
