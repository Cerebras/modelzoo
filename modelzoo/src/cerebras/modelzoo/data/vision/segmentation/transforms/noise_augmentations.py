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

from builtins import range
from typing import Tuple

import numpy as np


def augment_gaussian_noise(
    data_sample: np.ndarray,
    noise_variance: Tuple[float, float] = (0, 0.1),
    p_per_channel: float = 1,
    per_channel: bool = False,
) -> np.ndarray:
    if not per_channel:
        variance = (
            noise_variance[0]
            if noise_variance[0] == noise_variance[1]
            else np.random.uniform(noise_variance[0], noise_variance[1])
        )
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            # lol good luck reading this
            variance_here = (
                variance
                if variance is not None
                else (
                    noise_variance[0]
                    if noise_variance[0] == noise_variance[1]
                    else np.random.uniform(noise_variance[0], noise_variance[1])
                )
            )
            # bug fixed: https://github.com/MIC-DKFZ/batchgenerators/issues/86
            data_sample[c] = data_sample[c] + np.random.normal(
                0.0, variance_here, size=data_sample[c].shape
            )
    return data_sample
