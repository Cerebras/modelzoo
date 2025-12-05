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

import numpy as np


def construct_attention_mask(datadict, eos_id, pad_id, input_key='input_ids'):
    """
    Constructs attention masks based on the provided input_key in the datadict.
    The mask is constructed based on the presence of pad_id and eos_id.
    """
    input_data = datadict[input_key]

    attention_mask = []

    for i in range(input_data.shape[0]):
        pad_indices = np.where(input_data[i] == pad_id)[0]

        if eos_id != pad_id:
            # Handle case where eos_id and pad_id are different
            non_pad_len = (
                pad_indices[0] if len(pad_indices) > 0 else input_data.shape[1]
            )
        else:
            # Handle case where eos_id is the same as pad_id
            if len(pad_indices) > 0:
                pad_idx = 0
                while (
                    pad_idx + 1 < len(pad_indices)
                    and pad_indices[pad_idx] + 1 < input_data.shape[1]
                    and input_data[i][pad_indices[pad_idx] + 1] != pad_id
                ):
                    pad_idx += 1
                if pad_idx == len(pad_indices) - 1:
                    # All eos, no pad
                    non_pad_len = input_data.shape[1]
                else:
                    # Last eos just before pad, input_ids need to be chopped off
                    non_pad_len = pad_indices[pad_idx]
            else:
                non_pad_len = input_data.shape[1]

        attention_mask.append(
            [1] * non_pad_len + [0] * (input_data.shape[1] - non_pad_len)
        )

    return attention_mask
