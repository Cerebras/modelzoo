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

"""
Standalone script to validate a config.
"""


def validate_config(params_file: str):
    """
    Validate the config located at `params_file`.

    Args:
        params_file: The path to the config file to validate
    """
    import logging
    import os

    import yaml

    from cerebras.modelzoo.trainer.utils import (
        convert_legacy_params_to_trainer_params,
        is_legacy_params,
    )
    from cerebras.modelzoo.trainer.validate import validate_trainer_params

    if not os.path.exists(params_file):
        raise FileNotFoundError(f"File {params_file} not found")

    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    if is_legacy_params(params):
        params = convert_legacy_params_to_trainer_params(params)

    validate_trainer_params(params)
    logging.info("Config validation was successful!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=str, help="The params file to validate")
    validate_config(parser.parse_args().params)
