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


@registry.register_data_config("DiffusionBaseProcessor")
@dataclass
class DiffusionBaseProcessorConfig(DataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    use_worker_cache: bool = False
    mixed_precision: Optional[bool] = None
    num_classes: int = required
    noaugment: bool = False
    fp16_type: Optional[bool] = None
    transforms: Optional[List[dict]] = None
    vae_scaling_factor: Optional[float] = None
    label_dropout_rate: Optional[float] = None
    latent_size: Optional[List[int]] = None
    latent_channels: Optional[int] = None
    num_diffusion_steps: Optional[int] = None
    schedule_name: Optional[str] = None
    drop_last: bool = True


@registry.register_data_config("DiffusionImageNet1KProcessor")
@dataclass
class DiffusionImageNet1KProcessorConfig(DiffusionBaseProcessorConfig):
    pass


@registry.register_data_config("DiffusionLatentImageNet1KProcessor")
@dataclass
class DiffusionLatentImageNet1KProcessorConfig(DiffusionBaseProcessorConfig):
    pass
