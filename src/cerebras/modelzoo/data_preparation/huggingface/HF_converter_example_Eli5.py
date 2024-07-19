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

""" Example script to convert HF Eli5 dataset to HDF5 dataset"""

# isort: off
import os
import sys

# isort: on

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from cerebras.modelzoo.data_preparation.huggingface.HuggingFace_Eli5 import (
    HuggingFace_Eli5,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.convert_dataset_to_HDF5 import (
    convert_dataset_to_HDF5,
)


def main():

    dataset, data_collator = HuggingFace_Eli5(
        split="train", num_workers=8, sequence_length=128
    )
    convert_dataset_to_HDF5(
        dataset=dataset,
        data_collator=data_collator,
        output_dir="./eli5_hdf5_dataset/",
    )


if __name__ == "__main__":
    main()
