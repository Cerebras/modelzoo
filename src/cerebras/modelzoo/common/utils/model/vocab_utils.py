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

import os

from cerebras.modelzoo.common.model_utils.count_lines import count_lines


def get_vocab_size(vocab_file, vocab_size=None):
    """
    Function to get vocab size and validate with vocabulary file.
    :params str vocab_file: Path to vocabulary file.
    :params int vocab_size: Size of vocabulary file.
    :returns integer value indicating the size of vocabulary file.
    """
    if not vocab_file and not vocab_size:
        raise ValueError(
            f"Either `vocab_file` or `vocab_size` must be specified."
        )

    if vocab_file:
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"`vocab_file`: {vocab_file} not found.")
        vocab_file_lines = count_lines(vocab_file)
        if vocab_size:
            assert vocab_size == vocab_file_lines, (
                f"param `vocab_size` {vocab_size} does not match `vocab_size` "
                f"{vocab_file_lines} in `vocab_file` {vocab_file}."
            )
        else:
            vocab_size = vocab_file_lines

    assert vocab_size, f"`vocab_size` {vocab_size} is invalid."

    return vocab_size
