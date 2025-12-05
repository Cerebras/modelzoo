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

import torch

from cerebras.modelzoo.models.vision.generic_image_encoders.base.BaseSSLImageTransform import (
    BaseSSLImageTransform,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.config import (
    PassThroughTransformConfig,
)


class PassThroughTransform(BaseSSLImageTransform):
    def __init__(self, config: PassThroughTransformConfig):
        if isinstance(config, dict):
            config = PassThroughTransformConfig(**config)

        self._output_keys = config.output_keys

    def __call__(self, *args, **kwargs):
        return args, kwargs

    # Fix this to not be hard coded
    @property
    def output_keys(self):
        return self._output_keys

    def collate_fn(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch

    def visualize_transform(self, data):
        pass
