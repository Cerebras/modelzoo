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

import json

import numpy as np

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.hdf5_base_preprocessor import (
    HDF5BasePreprocessor,
)


class NLGPreprocessor(HDF5BasePreprocessor):
    """HDF5 preprocessor for NLG data sets such as E2E, DART, and WebNLG.
    Assumes the dataset has already been tokenized.
    Expect .jsonl input files that contains a "context" and a "completion" key.
    Used with GptHDF5DataProcessor.
    """

    def __init__(self, params):
        super(NLGPreprocessor, self).__init__(params)

    def file_read_generator(self, file):
        with open(file, 'r') as jsonl_file:
            for line in jsonl_file:
                j_dict = json.loads(line)
                context = j_dict["context"]
                completion = j_dict["completion"]
                yield context, completion

    def preprocessing_generator(self, doc):
        context, completion = doc

        input_ids = np.concatenate((context, completion[:-1]))
        labels = np.concatenate((context[1:], completion))

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

        sample = np.stack((input_ids, attention_mask, labels))

        yield sample
