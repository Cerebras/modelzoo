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


@registry.register_data_config("UNetDataProcessor")
@dataclass
class UNetDataProcessorConfig(DataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    num_classes: int = required
    loss: str = required
    normalize_data_method: str = required
    augment_data: bool = True
    num_workers: int = 0
    drop_last: bool = True
    prefetch_factor: int = 10
    persistent_workers: bool = True
    mixed_precision: Optional[bool] = None
    use_fast_dataloader: bool = False
    duplicate_act_worker_data: bool = False


@registry.register_data_config("CityscapesDataProcessor")
@dataclass
class CityscapesDataProcessorConfig(UNetDataProcessorConfig):
    use_worker_cache: bool = required
    image_shape: List[int] = field(default_factory=list)
    max_image_shape: List[int] = field(default_factory=list)


@registry.register_data_config("Hdf5BaseDataProcessor")
@dataclass
class Hdf5BaseDataProcessorConfig(DataProcessorConfig):
    use_worker_cache: bool = required
    data_dir: Union[str, List[str]] = required
    num_classes: int = required
    normalize_data_method: str = required
    image_shape: List[int] = field(default_factory=list)
    loss: str = required
    augment_data: bool = True
    shuffle_buffer: Optional[int] = None
    num_workers: int = 0
    drop_last: bool = True
    prefetch_factor: int = 10
    persistent_workers: bool = True
    mixed_precision: Optional[bool] = None
    use_fast_dataloader: bool = False
    duplicate_act_worker_data: bool = False


@registry.register_data_config("Hdf5DataProcessor")
@dataclass
class Hdf5DataProcessorConfig(Hdf5BaseDataProcessorConfig):
    pass


@registry.register_data_config("InriaAerialDataProcessor")
@dataclass
class InriaAerialDataProcessorConfig(DataProcessorConfig):
    use_worker_cache: bool = required
    data_dir: Union[str, List[str]] = required
    num_classes: int = required
    image_shape: List[int] = field(default_factory=list)
    duplicate_act_worker_data: bool = required
    loss: str = required
    normalize_data_method: Optional[str] = None
    augment_data: bool = True
    num_workers: int = 0
    drop_last: bool = True
    prefetch_factor: int = 10
    persistent_workers: bool = True
    mixed_precision: Optional[bool] = None
    overfit: bool = False
    overfit_num_batches: Optional[int] = None
    overfit_indices: Optional[List[int]] = None
    use_fast_dataloader: bool = False


@registry.register_data_config("SeverstalBinaryClassDataProcessor")
@dataclass
class SeverstalBinaryClassDataProcessorConfig(UNetDataProcessorConfig):
    use_worker_cache: bool = required
    train_test_split: float = required
    class_id: int = required
    image_shape: List[int] = field(default_factory=list)
    max_image_shape: List[int] = field(default_factory=list)


@registry.register_data_config("Hdf5BaseIterDataProcessor")
@dataclass
class Hdf5BaseIterDataProcessorConfig(DataProcessorConfig):
    use_worker_cache: bool = required
    data_dir: Union[str, List[str]] = field(default_factory=list)
    num_classes: int = required
    image_shape: List[int] = field(default_factory=list)
    loss: str = required
    normalize_data_method: Optional[str] = None
    augment_data: bool = True
    num_workers: int = 0
    shuffle_buffer: Optional[int] = None
    drop_last: bool = True
    prefetch_factor: int = 10
    persistent_workers: bool = True
    mixed_precision: Optional[bool] = None


@registry.register_data_config("SkmDataProcessor")
@dataclass
class SkmDataProcessorConfig(Hdf5BaseIterDataProcessorConfig):
    echo_type: str = "echo1"
    aggregate_cartilage: bool = True
