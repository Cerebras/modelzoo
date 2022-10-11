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
import shutil

import yaml


def check_and_create_output_dirs(output_dir, filetype):
    contains_filetype = False
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            if filetype in fname:
                contains_filetype = True
                break

    if contains_filetype:
        _in = input(
            f"Output directory already contains {filetype} file(s)."
            + " Do you want to delete the folder to write"
            + " new records in the same output folder name? (yes/no): "
        )
        if _in.lower() in ["y", "yes"]:
            shutil.rmtree(output_dir)
        elif _in.lower() in ["n", "no"]:
            raise IsADirectoryError(
                "Create a new folder for the files you want to write!!"
            )
        else:
            raise ValueError(f"Inputs can be yes, no, y or n. Received {_in}!!")

    os.makedirs(output_dir, exist_ok=True)


def save_params(params, model_dir, fname="params.yaml"):
    """
    Writes and saves a dictionary to a file in the model_dir.

    :param dict params: dict we want to write to a file in model_dir
    :param string model_dir: Directory we want to write to
    :param string fname: Name of file in model_dir we want to save to.
    """
    if not model_dir:
        raise ValueError(
            "model_dir is not provided. For saving params, user-defined"
            + " model_dir must be passed either through the command line"
            + " or from the yaml"
        )

    params_fname = os.path.join(model_dir, fname,)
    try:
        os.makedirs(os.path.dirname(params_fname), exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"Invalid path {model_dir} provided. Check the model_dir path!"
        )

    with open(params_fname, "w+") as _fout:
        yaml.dump(params, _fout, default_flow_style=False)
