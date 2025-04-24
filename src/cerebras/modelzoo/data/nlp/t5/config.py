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

from typing import List, Literal, Union

from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessorConfig,
)


class T5HDF5DataProcessorConfig(HDF5IterableDataProcessorConfig):
    data_processor: Literal["T5HDF5DataProcessor"]

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    "The path to the HDF5 files."
    num_workers: int = 0
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    use_vsl: bool = True
    """ Flag to enable variable sequence length training.
        It requires the dataset to have two extra features"""
