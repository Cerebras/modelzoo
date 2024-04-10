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

# from: https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/
# augmentations/color_augmentations.py (commit id: 01f225d)

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
from typing import Callable, Tuple, Union

import numpy as np


def augment_contrast(
    data_sample: np.ndarray,
    contrast_range: Union[Tuple[float, float], Callable[[], float]] = (
        0.75,
        1.25,
    ),
    preserve_range: bool = True,
    per_channel: bool = True,
    p_per_channel: float = 1,
) -> np.ndarray:
    if not per_channel:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(
                    max(contrast_range[0], 1), contrast_range[1]
                )

        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                if callable(contrast_range):
                    factor = contrast_range()
                else:
                    if np.random.random() < 0.5 and contrast_range[0] < 1:
                        factor = np.random.uniform(contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(
                            max(contrast_range[0], 1), contrast_range[1]
                        )

                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


def augment_brightness_multiplicative(
    data_sample, multiplier_range=(0.5, 2), per_channel=True
):
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(
                multiplier_range[0], multiplier_range[1]
            )
            data_sample[c] *= multiplier
    return data_sample


def augment_gamma(
    data_sample,
    gamma_range=(0.5, 2),
    invert_image=False,
    epsilon=1e-7,
    per_channel=False,
    retain_stats: Union[bool, Callable[[], bool]] = False,
):
    if invert_image:
        data_sample = -data_sample

    if not per_channel:
        retain_stats_here = (
            retain_stats() if callable(retain_stats) else retain_stats
        )
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = (
            np.power(((data_sample - minm) / float(rnge + epsilon)), gamma)
            * rnge
            + minm
        )
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = (
                retain_stats() if callable(retain_stats) else retain_stats
            )
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(
                    max(gamma_range[0], 1), gamma_range[1]
                )
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = (
                np.power(
                    ((data_sample[c] - minm) / float(rnge + epsilon)), gamma
                )
                * float(rnge + epsilon)
                + minm
            )
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = (
                    data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                )
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = -data_sample
    return data_sample
