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
from typing import Optional

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataProcessorConfig,
)


@registry.register_data_config("RawDatasetProcessor")
@dataclass
class RawDatasetProcessorConfig(DataProcessorConfig):
    # This preprocessing sections defines all the necessary parameters needed for
    # inline processing. This is an optional parameter needed only when inline
    # preprocessing is enabled
    preprocessing: Optional[dict] = None
    drop_last: bool = True
    prefetch_factor: int = 10
    persistent_workers: bool = True
    seed: Optional[int] = None
