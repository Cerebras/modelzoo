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
from typing import List, Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    required,
)
from cerebras.modelzoo.data.common.config import HDF5IterableDataProcessorConfig


@registry.register_data_config("DpoHDF5DataProcessor")
@dataclass
class DpoHDF5DataProcessorConfig(HDF5IterableDataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    use_vsl: bool = False
    """ Flag to enable variable sequence length training.
        It requires the dataset to have two extra features"""
