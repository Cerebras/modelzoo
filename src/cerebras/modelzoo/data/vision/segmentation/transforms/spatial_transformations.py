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

import numpy as np

from cerebras.modelzoo.data.internal.vision.segmentation.transforms.augmentation_utils import (
    create_zero_centered_coordinate_mesh,
    elastic_deform_coordinates,
    interpolate_img,
    rotate_coords_2d,
    rotate_coords_3d,
    scale_coords,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.crop_and_pad_augmentations import (
    center_crop as center_crop_aug,
)
from cerebras.modelzoo.data.vision.segmentation.transforms.crop_and_pad_augmentations import (
    random_crop as random_crop_aug,
)


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]"
        )
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg


def augment_spatial(
    data,
    seg,
    patch_size,
    patch_center_dist_from_border=30,
    do_elastic_deform=True,
    alpha=(0.0, 1000.0),
    sigma=(10.0, 13.0),
    do_rotation=True,
    angle_x=(0, 2 * np.pi),
    angle_y=(0, 2 * np.pi),
    angle_z=(0, 2 * np.pi),
    do_scale=True,
    scale=(0.75, 1.25),
    border_mode_data='nearest',
    border_cval_data=0,
    order_data=3,
    border_mode_seg='constant',
    border_cval_seg=0,
    order_seg=0,
    random_crop=True,
    p_el_per_sample=1,
    p_scale_per_sample=1,
    p_rot_per_sample=1,
    independent_scale_for_each_axis=False,
    p_rot_per_axis: float = 1,
    p_independent_scale_per_axis: int = 1,
):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros(
                (seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]),
                dtype=np.float32,
            )
        else:
            seg_result = np.zeros(
                (
                    seg.shape[0],
                    seg.shape[1],
                    patch_size[0],
                    patch_size[1],
                    patch_size[2],
                ),
                dtype=np.float32,
            )

    if dim == 2:
        data_result = np.zeros(
            (data.shape[0], data.shape[1], patch_size[0], patch_size[1]),
            dtype=np.float32,
        )
    else:
        data_result = np.zeros(
            (
                data.shape[0],
                data.shape[1],
                patch_size[0],
                patch_size[1],
                patch_size[2],
            ),
            dtype=np.float32,
        )

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if (
                independent_scale_for_each_axis
                and np.random.uniform() < p_independent_scale_per_axis
            ):
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(
                        patch_center_dist_from_border[d],
                        data.shape[d + 2] - patch_center_dist_from_border[d],
                    )
                else:
                    ctr = data.shape[d + 2] / 2.0 - 0.5
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(
                    data[sample_id, channel_id],
                    coords,
                    order_data,
                    border_mode_data,
                    cval=border_cval_data,
                )
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(
                        seg[sample_id, channel_id],
                        coords,
                        order_seg,
                        border_mode_seg,
                        cval=border_cval_seg,
                        is_seg=True,
                    )
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id : sample_id + 1]
            if random_crop:
                margin = [
                    patch_center_dist_from_border[d] - patch_size[d] // 2
                    for d in range(dim)
                ]
                d, s = random_crop_aug(
                    data[sample_id : sample_id + 1], s, patch_size, margin
                )
            else:
                d, s = center_crop_aug(
                    data[sample_id : sample_id + 1], patch_size, s
                )
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result
