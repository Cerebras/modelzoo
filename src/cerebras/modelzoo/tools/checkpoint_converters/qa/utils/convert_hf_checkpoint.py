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

import argparse

import torch
import torch.nn

"""
This script takes a pytorch_model.bin file from a HuggingFace model card, and converts it into an .mdl file that interfaces with Cerebras run.py training. 
For example, you could take the output of this script and use it in the command-line args.

CPU/GPU example:
python run.py {CPU,GPU} --params /path/ --mode train --checkpoint_path fill/in/here --disable_strict_checkpoint_loading

CSX example:
python run.py CSX {pipeline,weight_streaming} --params /path/ --mode train --checkpoint_path fill/in/here --disable_strict_checkpoint_loading

Sample usage:
python convert_hf_checkpoint.py \
    --input_checkpoint_path /cb/ml/t5/lm_adapted/small/pytorch_model.bin \
    --out_path /cb/ml/t5/lm_adapted/small/t5_small_lm_adapted.mdl
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_checkpoint_path",
        required=True,
        help="""Path to HuggingFace checkpoint to be converted into our 
        format. Expects the pytorch_model.bin that is given in the model
        card of a page""",
    )
    parser.add_argument(
        "--out_path",
        required=True,
        help="""Path to save location for the converted checkpoint. 
        Convention within Cerebras is to use .mdl extension""",
    )
    args = parser.parse_args()
    return args


def convert_checkpoint(path, out_path):
    # convert keys
    model_state_dict = torch.load(path)

    # the state_dict in our pipeline expects a dictionary with keys
    # for model, optimizer, etc. so we minimally wrap the model state_dict
    # in a wrapper dict
    out_dict = dict()
    out_dict["model"] = model_state_dict
    torch.save(
        out_dict,
        out_path,
    )


def main():
    args = parse_args()
    convert_checkpoint(args.input_checkpoint_path, args.out_path)


if __name__ == "__main__":
    main()
