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
Config classes of Input Configs

"""

import logging
from dataclasses import dataclass, fields
from typing import Optional, Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
    required,
)


@dataclass
class DataProcessorConfig(BaseConfig):
    batch_size: int = required
    "Batch size to be used"
    shuffle: bool = True
    "Whether or not to shuffle the dataset"
    shuffle_seed: int = 0
    "Seed used for deterministic shuffling"
    num_workers: int = 0
    "The number of PyTorch processes used in the dataloader"
    prefetch_factor: Optional[int] = None
    "The number of batches to prefetch in the dataloader"
    persistent_workers: Optional[bool] = None
    "Whether or not to keep workers persistent between epochs"


@dataclass
class DataConfig(BaseConfig):
    data_processor: str = required
    micro_batch_size: Optional[Union[dict, int, str]] = "auto"
    "Name of the data processor to use"
    params: Optional[dict] = None

    # Custom init for DataConfig, where we want to capture all fixed params as members.
    # Data params is a dict that is populated by checking all additional params supplied to us.
    # These are dataloader specific and we use signature of that data processor to validate these.
    def __init__(self, **kwargs):
        class_fields = fields(self)
        for class_field in class_fields:
            field_name = class_field.name
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
        self.params = {key: value for key, value in kwargs.items()}
        super().__init__()

    def __post_init__(self):
        super().__post_init__()
        data_processor_config = registry.get_data_config(self.data_processor)
        if data_processor_config is None:
            logging.debug(
                f"DATA CONFIG : Could not find data processor config: {self.data_processor} in the registry, cannot complete validation of data params."
            )
        else:
            try:
                params_config = data_processor_config(**(self.params))
                params_config.__validate__()
            except Exception as e:  # pylint: disable=broad-except
                logging.debug(
                    f"DATA CONFIG : Invalid data processor configuration supplied for {data_processor_config}."
                    f"Please fix error : {e}  or contact Cerebras support"
                )
