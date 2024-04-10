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

import copy
import json
import logging

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def dump_result(
    results,
    json_params_file,
    eos_id=None,
    pad_id=None,
    vocab_size=None,
):
    """
    Write outputs of execution
    """
    with open(json_params_file, "r") as _fin:
        data = json.load(_fin)

    post_process = {}
    post_process["discarded_files"] = results.pop("discarded")
    post_process["processed_files"] = results.pop("processed")
    post_process["successful_files"] = results.pop("successful")
    post_process["n_examples"] = results.pop("examples")
    post_process["raw_chars_count"] = results.pop("raw_chars_count")
    post_process["raw_bytes_count"] = results.pop("raw_bytes_count")

    ## put remaining key,value pairs in post process
    for key, value in results.items():
        post_process[key] = value

    if eos_id is not None:
        post_process["eos_id"] = eos_id
    if pad_id is not None:
        post_process["pad_id"] = pad_id
    if vocab_size is not None:
        post_process["vocab_size"] = vocab_size

    data["post-process"] = post_process

    with open(json_params_file, "w") as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)


def dump_args(args, json_params_file):
    """
    Write the input params to file.
    """
    logger.info(f"User arguments can be found at {json_params_file}.")

    redundant_params = [
        "eos_id",
        "pad_id",
        "display_pbar",
        "files_per_record",
        "output_name",
        "write_remainder",
    ]

    relevant_args = copy.deepcopy(args)
    # Iterate through the dictionary and remove the redundant params
    for key in redundant_params:
        for sub_dict in relevant_args.values():
            if key in sub_dict:
                del sub_dict[key]

    # write initial params to file
    with open(json_params_file, "w") as _fout:
        json.dump(args, _fout, indent=4, sort_keys=True)
