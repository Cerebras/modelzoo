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
from typing import List, Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataProcessorConfig,
)


@dataclass
class ProcessorConfig(DataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    num_classes: int = required
    mixed_precision: bool = required
    transforms: List[dict] = field(default_factory=list)
    image_size: int = 224
    noaugment: bool = False
    drop_last: bool = True
    num_workers: int = 0
    prefetch_factor: int = 10
    persistent_workers: bool = True
    sampler: str = "random"
    ra_sampler_num_repeat: int = 3
    mixup_alpha: float = 0.1
    cutmix_alpha: float = 0.1


@registry.register_data_config("ImageNet1KProcessor")
@dataclass
class ImageNet1KProcessorConfig(ProcessorConfig):
    use_worker_cache: bool = required


@registry.register_data_config("ImageNet21KProcessor")
@dataclass
class ImageNet21KProcessorConfig(ProcessorConfig):
    pass
