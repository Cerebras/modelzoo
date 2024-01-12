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


def create_preprocessing_params_with_defaults(params):
    """Preprocessing params for augmentations"""
    pp_params = dict()
    pp_params["noaugment"] = params.get("noaugment", False)
    pp_params["mixed_precision"] = params["mixed_precision"]
    pp_params["fp16_type"] = params["fp16_type"]
    pp_params["transforms"] = params["transforms"]

    return pp_params
