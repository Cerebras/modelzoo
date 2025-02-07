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

from abc import ABC, abstractmethod
from typing import ClassVar

from cerebras.modelzoo.config import NamedConfig, create_config_class


class BaseSSLImageTransformConfig(NamedConfig, ABC):
    discriminator: ClassVar[str] = "name"

    @property
    @abstractmethod
    def __transform_cls__(self):
        pass

    def __call__(self, **kwargs):
        return create_config_class(self.__transform_cls__).model_validate(
            self
        )()


class BaseSSLImageTransform(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def output_keys(self):
        # keys in dictionary returned by collate_fn
        pass

    @abstractmethod
    def __call__(self, image):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    @abstractmethod
    def visualize_transform(self, batch):
        pass
