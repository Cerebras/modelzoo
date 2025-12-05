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

# Adapted from: https://github.com/MIC-DKFZ/batchgenerators (commit id: 01f225d)
#
# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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

import numpy as np

from cerebras.modelzoo.data.vision.segmentation.transforms.noise_augmentations import (
    augment_gaussian_noise,
)


class GaussianNoiseTransform:
    def __init__(
        self,
        noise_variance=(0, 0.1),
        p_per_sample=1,
        p_per_channel: float = 1,
        per_channel: bool = False,
        data_key="data",
    ):
        """
        Adds additive Gaussian Noise

        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:

        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_noise(
                    data_dict[self.data_key][b],
                    self.noise_variance,
                    self.p_per_channel,
                    self.per_channel,
                )
        return data_dict
