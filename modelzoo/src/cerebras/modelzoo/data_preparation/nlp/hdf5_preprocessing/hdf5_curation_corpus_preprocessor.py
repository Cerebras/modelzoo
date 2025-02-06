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

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.hdf5_dataset_preprocessors import (
    SummarizationPreprocessor,
)


class CurationCorpusPreprocessor(SummarizationPreprocessor):
    """A customized version of the SummarizationPreprocessor for the Curation Corpus dataset in .jsonl format.
    Handles .jsonl's with a article content key and a summariaztion key.
    Used with GptHDF5DataProcessor.
    """

    def __init__(self, params):
        super(CurationCorpusPreprocessor, self).__init__(params)

    def file_read_generator(self, file):
        with open(file, 'r') as jsonl_file:
            for line in jsonl_file:
                j_dict = json.loads(line)
                prompt = j_dict[self.prompt_key]
                completion = j_dict[self.completion_key]
                if not completion:
                    print("empty label, example discarded")
                    continue
                if prompt == "Exception":
                    print("bad prompt, example discarded")
                    continue
                yield prompt, completion
