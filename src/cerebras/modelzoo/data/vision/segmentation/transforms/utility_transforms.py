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

import torch


class NumpyToTensor:
    def __init__(self, cast_to=None):
        """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
        :param cast_to [list]: images will be cast to cast_to[0], targets will be cast to cast_to[1].
        """
        self.cast_to = cast_to

    def __call__(self, **data_dict):
        data_dict['data'] = (
            torch.from_numpy(data_dict['data']).contiguous().to(self.cast_to[0])
        )
        data_dict['target'] = (
            torch.from_numpy(data_dict['target'])
            .contiguous()
            .to(self.cast_to[1])
        )
        return data_dict


class RemoveLabelTransform:
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(
        self, remove_label, replace_with=0, input_key="seg", output_key="seg"
    ):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict


class RenameTransform:
    '''
    Saves the value of data_dict[in_key] to data_dict[out_key]. Optionally removes data_dict[in_key] from the dict.
    '''

    def __init__(self, in_key, out_key, delete_old=False):
        self.delete_old = delete_old
        self.out_key = out_key
        self.in_key = in_key

    def __call__(self, **data_dict):
        data_dict[self.out_key] = data_dict[self.in_key]
        if self.delete_old:
            del data_dict[self.in_key]
        return data_dict


class OneHotTransform:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, **data_dict):
        data_dict['target'] = torch.tensor(
            data_dict['target'][:, 0, :], dtype=torch.long
        )
        # out shape: (H, W, num_classes)
        data_dict['target'] = torch.nn.functional.one_hot(
            data_dict['target'], num_classes=self.num_classes
        )
        data_dict['target'] = data_dict['target'].to(torch.float32)
        data_dict['target'] = torch.permute(
            data_dict['target'], (0, -1, 1, 2, 3)
        )
        return data_dict


class OneHotTransformKits:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data_dict):
        data_dict['label'] = torch.tensor(data_dict['label'], dtype=torch.long)
        # out shape: (H, W, num_classes)
        data_dict['label'] = torch.nn.functional.one_hot(
            data_dict['label'], num_classes=self.num_classes
        )
        data_dict['label'] = data_dict['label'].to(torch.float32)
        data_dict['label'] = torch.permute(data_dict['label'], (0, -1, 1, 2, 3))
        return data_dict
