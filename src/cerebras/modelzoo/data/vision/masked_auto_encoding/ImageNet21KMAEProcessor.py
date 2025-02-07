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

from typing import Literal

from cerebras.modelzoo.data.vision.classification.data.imagenet21k import (
    ImageNet21KProcessor,
    ImageNet21KProcessorConfig,
)
from cerebras.modelzoo.data.vision.masked_auto_encoding.MAEProcessor import (
    MAEProcessor,
    MAEProcessorConfig,
)


class ImageNet21KMAEProcessorConfig(
    MAEProcessorConfig, ImageNet21KProcessorConfig
):
    data_processor: Literal["ImageNet21KMAEProcessor"]


class ImageNet21KMAEProcessor(MAEProcessor, ImageNet21KProcessor):
    def __init__(self, config: ImageNet21KMAEProcessorConfig):
        if isinstance(config, dict):
            config = ImageNet21KMAEProcessorConfig(**config)
        MAEProcessor.__init__(self, config)
        ImageNet21KProcessor.__init__(self, config)

    def create_dataloader(self):
        dataloader = super().create_dataloader()
        dataloader.collate_fn = self.mae_collate_fn
        return dataloader
