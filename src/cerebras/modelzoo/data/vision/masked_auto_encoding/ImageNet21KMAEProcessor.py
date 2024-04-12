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

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.vision.classification.data.imagenet21k import (
    ImageNet21KProcessor,
)
from cerebras.modelzoo.data.vision.masked_auto_encoding.MAEProcessor import (
    MAEProcessor,
)


@registry.register_datasetprocessor("ImageNet21KMAEProcessor")
class ImageNet21KMAEProcessor(MAEProcessor, ImageNet21KProcessor):
    def __init__(self, params):
        super().__init__(params=params)

    def create_dataloader(self, dataset, is_training=False):
        dataloader = super().create_dataloader(dataset, is_training)
        dataloader.collate_fn = self.mae_collate_fn
        return dataloader
