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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataProcessorConfig,
)


@registry.register_data_config("LlavaHDF5MapDataProcessor")
@dataclass
class LlavaHDF5MapDataProcessorConfig(DataProcessorConfig):
    img_data_dir: str = required
    image_data_size: List[int] = field(default_factory=list)
    transforms: List[dict] = field(default_factory=list)
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    use_worker_cache: bool = False
    """ whether or not to copy data to storage
        that is directly attached to each individual worker node.
        Useful when your network storage is unusually slow, but
        otherwise discouraged."""
    max_sequence_length: Optional[int] = None
    """ The sequence length of samples
        produced by the dataloader. When using the corpus data format,
        the same preprocessed data will work with any max sequence
        length, so this may be set at runtime. When using the sample
        format this must be set to None"""
    mixture: Optional[List[dict]] = None
    mixed_precision: Optional[bool] = None
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
    sort_files: bool = True
    """ whether or not the reader should sort the input
        files. This is included for backwards compatibility and should
        almost always be set to True"""
    use_vsl: bool = False
    """ Flag to enable variable sequence length training.
        It requires the dataset to have two extra features"""
    pad_last: bool = False
    data_subset: Optional[str] = None
    """
        An optional specification to only consider a
        subset of the full dataset, useful for sequence length
        scheduling and multi-epoch testing. Expected to be a comma
        separated list of ranges, e.g. 0.0-0.5 or 0.1-0.3,0.7-1.0.
        Specifying 0.0-0.5 creates a dataset from the first half of
        the data on disk and disregards the second half.
    """
    dataset_map_fn: Optional[str] = None
    num_workers: int = 0
    """
        The number of PyTorch processes used in the dataloader
    """
    prefetch_factor: int = 10
    persistent_workers: bool = False
