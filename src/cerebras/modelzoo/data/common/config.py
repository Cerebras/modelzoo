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
from typing import List, Optional

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataProcessorConfig,
)


@registry.register_data_config("GenericDataProcessor")
@dataclass
class GenericDataProcessorConfig(DataProcessorConfig):
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """


@registry.register_data_config("HDF5IterableDataProcessor")
@dataclass
class HDF5IterableDataProcessorConfig(DataProcessorConfig):
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    prefetch_factor: int = 10
    persistent_workers: int = True


@registry.register_data_config("SyntheticDataProcessor")
@dataclass
class SyntheticDataProcessorConfig(DataProcessorConfig):
    num_examples: Optional[int] = None
    sampler: Optional[str] = None
    batch_sampler: Optional[List[List[int]]] = None
    pin_memory: bool = False
    drop_last: bool = False
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    timeout: bool = False
    synthetic_special_tokens_index: Optional[dict] = None


@registry.register_data_config("HuggingFaceDataProcessor")
@dataclass
class HuggingFaceDataProcessorConfig(DataProcessorConfig):
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    prefetch_factor: int = 10
    persistent_workers: bool = True
