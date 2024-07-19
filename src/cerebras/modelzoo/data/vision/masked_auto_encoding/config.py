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
from typing import List

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataProcessorConfig,
)
from cerebras.modelzoo.data.vision.classification.config import (
    ImageNet1KProcessorConfig,
    ImageNet21KProcessorConfig,
)


@dataclass
class MAEProcessorConfig(DataProcessorConfig):
    image_size: List[int] = field(default_factory=list)
    patch_size: List[int] = field(default_factory=list)
    image_channels: List[int] = required
    mask_ratio: float = 0.75


@registry.register_data_config("ImageNet1KMAEProcessor")
@dataclass
class ImageNet1KMAEProcessorConfig(
    ImageNet1KProcessorConfig, MAEProcessorConfig
):
    pass


@registry.register_data_config("ImageNet21KProcessor")
@dataclass
class ImageNet21KProcessorConfig(
    ImageNet21KProcessorConfig, MAEProcessorConfig
):
    pass
