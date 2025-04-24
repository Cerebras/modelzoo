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
This module contains the WeightCompression callback class which is used to apply
weight compression to the model.
"""

from typing import List, Union

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback


class WeightCompression(Callback):
    """
    Callback class to apply weight compression to the model.
    """

    def __init__(self, compressions: Union[dict, List[dict]]):
        """
        Args:
            compressions: Compression configuration to apply to the model.
        """
        if compressions:

            def get_compression_from_dict(single_config):
                if not isinstance(single_config, dict):
                    raise ValueError(
                        "Improper compression format due to "
                        "configuration not being a dictionary"
                    )
                if "format" not in single_config:
                    raise ValueError(
                        "Improper compression format due to "
                        "configuration not having \"format\" as a field"
                    )
                if "param_filter" not in single_config:
                    raise ValueError(
                        "Improper compression format due to "
                        "configuration not having \"param_filter\" as a field"
                    )

                return cstorch.experimental.Compression(
                    single_config["format"], single_config["param_filter"]
                )

            if isinstance(compressions, dict):
                # then turn this single dictionary value to a compression
                self.compressions = [get_compression_from_dict(compressions)]
            elif isinstance(compressions, list):
                self.compressions = list(
                    map(get_compression_from_dict, compressions)
                )
            else:
                raise ValueError(
                    "Expected `compressions` to be a dict or a list of dicts."
                )
        else:
            self.compressions = []

    def setup(self, trainer):
        for compression in self.compressions:
            trainer.model.apply(compression)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
