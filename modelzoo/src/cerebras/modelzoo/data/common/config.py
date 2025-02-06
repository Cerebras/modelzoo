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
Config classes of T5 data Configs.

"""

from typing import Literal, Optional

from cerebras.modelzoo.config import DataConfig


class GenericDataProcessorConfig(DataConfig):
    data_processor: Literal["GenericDataProcessor"]

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

    num_workers: int = 0
    "How many subprocesses to use for data loading."

    prefetch_factor: Optional[int] = 10
    "Number of batches loaded in advance by each worker."

    persistent_workers: bool = True
    """If True, the data loader will not shutdown
    the worker processes after a dataset has been consumed once."""


class HuggingFaceDataProcessorConfig(DataConfig):
    data_processor: Literal["HuggingFaceDataProcessor"]

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
    prefetch_factor: Optional[int] = 10
    persistent_workers: bool = True
