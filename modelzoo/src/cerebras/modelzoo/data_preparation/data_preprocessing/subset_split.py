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
import logging
import os

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SubsetSplit:

    def __init__(self, params):
        self.params = params
        self.setup_params = self.params.get("setup", {})
        self.data_params = self.setup_params.get("data", {})
        self.output_dir = self.setup_params.get("output_dir")
        self.type_of_data = self.data_params.get("type")

        if not self.output_dir:
            raise ValueError(
                "Output directory is not specified in the parameters."
            )

    def get_input_dir(self, data_params):
        if self.data_params.get("type") == "local":
            return self.data_params.get("source")
        raise ValueError("Unsupported data source type or missing source.")

    def process_sub_folder(self, params, sub_dir):
        updated_params = copy.deepcopy(params)

        updated_params["setup"]["data"]["source"] = sub_dir
        updated_params["setup"]["output_dir"] = os.path.join(
            params["setup"]["output_dir"], os.path.basename(sub_dir)
        )
        data_splits_dir = updated_params["setup"].get("data_splits_dir")
        if data_splits_dir:
            updated_params["setup"]["data_splits_dir"] = os.path.join(
                data_splits_dir, os.path.basename(sub_dir)
            )
        else:
            updated_params["setup"]["data_splits_dir"] = os.path.join(
                os.path.dirname(params["setup"]["output_dir"]),
                "data_splits_dir",
                os.path.basename(sub_dir),
            )
        logger.info(f"Updated params: {updated_params}")
        return updated_params

    def process_huggingface_params(self, params, subset):
        updated_params = copy.deepcopy(params)

        updated_params["setup"]["data"]["data_dir"] = subset
        updated_params["setup"]["data"]["verification_mode"] = "no_checks"
        updated_params["setup"]["output_dir"] = os.path.join(
            params["setup"]["output_dir"], f"{subset}"
        )
        data_splits_dir = updated_params["setup"].get("data_splits_dir")
        if data_splits_dir:
            updated_params["setup"]["data_splits_dir"] = os.path.join(
                data_splits_dir, f"{subset}"
            )
        return updated_params

    def generate_subsets(self):
        top_level_as_subsets = self.data_params.pop(
            "top_level_as_subsets", False
        )
        subsets = self.data_params.pop("subsets", False)

        if not (top_level_as_subsets or subsets):
            return [self.params]

        if top_level_as_subsets and subsets:
            raise ValueError(
                "Please specify only one of `top_level_as_subsets` or `subsets`!"
            )

        if self.type_of_data == "huggingface":
            if not subsets:
                logger.info(
                    f"No subsets are provided for individual processing; falling back to default behaviour."
                )
                return [self.params]

            return [
                self.process_huggingface_params(self.params, subset)
                for subset in subsets
            ]

        else:
            input_dir = self.get_input_dir(self.data_params)
            sub_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
            sub_dirs.sort()

            if top_level_as_subsets:
                logger.info("Processing each sub-folder as a subset...")
                return [
                    self.process_sub_folder(self.params, sub_dir)
                    for sub_dir in sub_dirs
                ]

            if subsets:
                logger.info(
                    "Processing the subset of inputs given in the input directory..."
                )
                subset_dirs = set(
                    os.path.join(input_dir, subset) for subset in subsets
                )

                invalid_dirs = [
                    dir for dir in subset_dirs if not os.path.exists(dir)
                ]

                if invalid_dirs:
                    logger.warning(
                        f"The following subset directories are invalid and will be skipped: {invalid_dirs}"
                    )

                valid_subset_dirs = subset_dirs - set(invalid_dirs)

                return [
                    self.process_sub_folder(self.params, sub_dir)
                    for sub_dir in sub_dirs
                    if sub_dir in valid_subset_dirs
                ]

            return [self.params]
