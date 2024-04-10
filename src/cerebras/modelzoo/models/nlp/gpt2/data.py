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

import sys

from cerebras.modelzoo.data.nlp.gpt.DummyDataProcessor import (  # noqa
    DummyDataProcessor,
)

from cerebras.modelzoo.data.nlp.gpt.HuggingFaceDataProcessorEli5 import (  # noqa
    HuggingFaceDataProcessorEli5,
)
from cerebras.modelzoo.data.nlp.gpt.HuggingFaceIterableDataProcessorEli5 import (  # noqa
    HuggingFaceIterableDataProcessorEli5,
)

from cerebras.modelzoo.data.nlp.gpt.DummyIterableDataProcessor import (  # noqa
    DummyIterableDataProcessor,
)

from cerebras.modelzoo.data.nlp.gpt.GptHDF5DataProcessor import (  # noqa
    GptHDF5DataProcessor,
)

from cerebras.modelzoo.data.nlp.gpt.GptHDF5MapDataProcessor import (  # noqa
    GptHDF5MapDataProcessor,
)


def train_input_dataloader(params):
    return getattr(
        sys.modules[__name__], params["train_input"]["data_processor"]
    )(params["train_input"]).create_dataloader()


def eval_input_dataloader(params):
    return getattr(
        sys.modules[__name__], params["eval_input"]["data_processor"]
    )(params["eval_input"]).create_dataloader()


def inference_input_dataloader(params):
    return getattr(
        sys.modules[__name__], params["inference_input"]["data_processor"]
    )(params["inference_input"]).create_dataloader()
