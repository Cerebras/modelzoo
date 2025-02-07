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

""" PyTorch CLI Utilities"""

import argparse
import sys
from typing import Callable, List, Optional

from cerebras.modelzoo.common.utils.run.cli_parser import (
    get_params_from_args as base_get_params_from_args,
)
from cerebras.modelzoo.common.utils.run.utils import DeviceType


def get_params_from_args(
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """Parse commandline and return params"""

    def pytorch_specific_args_parser():
        extra_args = extra_args_parser_fn() if extra_args_parser_fn else {}
        if isinstance(extra_args, list):
            for item in extra_args:
                # pylint: disable=protected-access
                item._action_groups[1].title = (
                    "User-Defined and/or Model Specific Arguments"
                )
        parser = argparse.ArgumentParser(parents=extra_args, add_help=False)
        # Arguments must be added as part of a group in order to propagate the
        # correct help message to users.
        parser_opt = parser.add_argument_group(
            "Optional Arguments, PyTorch Specific"
        )
        parser_opt.add_argument(
            "--num_epochs",
            type=int,
            default=None,
            help="Specifies the number of epochs to run",
        )
        parser_opt.add_argument(
            "--num_steps",
            type=int,
            default=None,
            help="Specifies the number of steps to run",
        )
        return {
            DeviceType.ANY: [parser],
        }

    # Parse arguments from the command line and get the params
    # from the specified config file
    params = base_get_params_from_args(
        argv=sys.argv[1:],
        extra_args_parser_fn=pytorch_specific_args_parser,
    )

    return params
