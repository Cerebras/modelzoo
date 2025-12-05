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
Config classes of GPT data Configs.

"""

from typing import Any, List, Literal, Optional, Union

from pydantic import Field

from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.common.config import (
    GenericDataProcessorConfig,
    HuggingFaceDataProcessorConfig,
)
from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessorConfig,
)


class DummyDataProcessorConfig(GenericDataProcessorConfig):
    data_processor: Literal["DummyDataProcessor"]


class DummyIterableDataProcessorConfig(GenericDataProcessorConfig):
    data_processor: Literal["DummyIterableDataProcessor"]


class GptHDF5DataProcessorConfig(HDF5IterableDataProcessorConfig):
    data_processor: Literal["GptHDF5DataProcessor"]

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    "The path to the HDF5 files."
    max_sequence_length: Optional[int] = None
    """ The sequence length of samples
        produced by the dataloader. When using the corpus data format,
        the same preprocessed data will work with any max sequence
        length, so this may be set at runtime. When using the sample
        format this must be set to None"""
    drop_last: bool = True
    use_vsl: bool = False

    batch_size: int = ...
    "Batch size."

    shuffle: bool = ...
    "Flag to enable data shuffling."

    shuffle_seed: Optional[int] = None
    "Shuffle seed."

    use_vsl: bool = False
    """Flag to enable variable sequence length training.
    It requires the dataset to have two extra features: the
    `attention_span` of keys and the `position_ids` of tokens."""

    repeat: Optional[Any] = Field(None, deprecated=True)
    use_multiple_workers: Optional[Any] = Field(None, deprecated=True)


class HuggingFaceIterableDataProcessorEli5Config(
    HuggingFaceDataProcessorConfig
):
    data_processor: Literal["HuggingFaceIterableDataProcessorEli5"]

    split: str = ...
    num_workers: int = 0


class InferenceDataProcessorConfig(DataConfig):
    data_processor: Literal["InferenceDataProcessor"]

    num_workers: int = 0
    prefetch_factor: Optional[int] = 10
    persistent_workers: bool = False
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """


class InferenceDataProcessorLLConfig(InferenceDataProcessorConfig):
    data_processor: Literal["InferenceDataProcessorLL"]


class InferenceDataProcessorGUConfig(InferenceDataProcessorConfig):
    data_processor: Literal["InferenceDataProcessorGU"]
