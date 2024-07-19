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
Config classes of T5 data Configs

"""

from dataclasses import dataclass
from typing import List, Optional, Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataProcessorConfig,
)
from cerebras.modelzoo.data.common.config import (
    GenericDataProcessorConfig,
    HDF5IterableDataProcessorConfig,
    HuggingFaceDataProcessorConfig,
)


@registry.register_data_config("DummyDataProcessor")
@dataclass
class DummyDataProcessorConfig(GenericDataProcessorConfig):
    pass


@registry.register_data_config("DummyIterableDataProcessor")
@dataclass
class DummyIterableDataProcessorConfig(GenericDataProcessorConfig):
    pass


@registry.register_data_config("GptHDF5DataProcessor")
@dataclass
class GptHDF5DataProcessorConfig(HDF5IterableDataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    max_sequence_length: Optional[int] = None
    """ The sequence length of samples
        produced by the dataloader. When using the corpus data format,
        the same preprocessed data will work with any max sequence
        length, so this may be set at runtime. When using the sample
        format this must be set to None"""
    drop_last: bool = True
    use_vsl: bool = False


@registry.register_data_config("GptHDF5MapDataProcessor")
@dataclass
class GptHDF5MapDataProcessorConfig(DataProcessorConfig):
    data_dir: Optional[Union[str, List[str]]] = None
    "The path to the HDF5 files."
    use_worker_cache: bool = False
    max_sequence_length: Optional[int] = None
    """ The sequence length of samples
        produced by the dataloader. When using the corpus data format,
        the same preprocessed data will work with any max sequence
        length, so this may be set at runtime. When using the sample
        format this must be set to None"""
    mixture: Optional[List[dict]] = None
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    num_samples: Optional[int] = None
    num_workers: int = 0
    "The number of PyTorch processes used in the dataloader"
    prefetch_factor: int = 10
    "The number of batches to prefetch in the dataloader"
    persistent_workers: bool = True
    "Whether or not to keep workers persistent between epochs"
    sort_files: bool = True
    """ whether or not the reader should sort the input
        files. This is included for backwards compatibility and should
        almost always be set to True"""
    use_vsl: bool = False
    """ Flag to enable variable sequence length training.
        It requires the dataset to have two extra features"""
    pad_last: bool = False
    data_subset: Optional[str] = None
    dataset_map_fn: Optional[str] = None


@registry.register_data_config("HuggingFaceDataProcessorEli5")
@dataclass
class HuggingFaceDataProcessorEli5Config(HuggingFaceDataProcessorConfig):
    split: str = required
    num_workers: int = 0


@registry.register_data_config("HuggingFaceIterableDataProcessorEli5")
@dataclass
class HuggingFaceIterableDataProcessorEli5Config(
    HuggingFaceDataProcessorConfig
):
    split: str = required
    num_workers: int = 0


@registry.register_data_config("InferenceDataProcessor")
@dataclass
class InferenceDataProcessorConfig(DataProcessorConfig):
    num_workers: int = 0
    prefetch_factor: int = 10
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


@registry.register_data_config("InferenceDataProcessorLL")
@dataclass
class InferenceDataProcessorLLConfig(InferenceDataProcessorConfig):
    pass


@registry.register_data_config("InferenceDataProcessorGU")
@dataclass
class InferenceDataProcessorGUConfig(InferenceDataProcessorConfig):
    pass
