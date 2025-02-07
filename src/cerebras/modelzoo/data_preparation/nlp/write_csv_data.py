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

# isort: off
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
# isort: on
from cerebras.modelzoo.data_preparation.nlp.bert.bertsum_data_processor import (
    convert_to_json_files,
    create_parser,
    tokenize,
)
from cerebras.modelzoo.data_preparation.nlp.extractive_summarisation_utils import (
    BertCSVFormatter,
)


def convert_to_bert_format_files(params):
    """
    Format input to tf records.
    Takes params.input_path, convert it to bert
    format and store it under params.output_path".
    """
    BertCSVFormatter(params).process()


if __name__ == "__main__":
    parser = create_parser()

    params = parser.parse_args()

    if params.mode == "tokenize":
        tokenize(params)
    elif params.mode == "convert_to_json_files":
        convert_to_json_files(params)
    elif params.mode == "convert_to_bert_format_files":
        convert_to_bert_format_files(params)
    else:
        raise ValueError(
            "Unknown `mode` passed in command line argument. Please run "
            "`--help` for a list of available modes."
        )
