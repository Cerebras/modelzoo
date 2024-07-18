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

from cerebras.modelzoo.trainer.callbacks import Callback


class WeightCompression(Callback):
    """
    Callback class to apply weight compression to the model.
    """

    def __init__(self, compressions: dict):
        """
        Args:
            compressions: Compression configuration to apply to the model.
        """
        if compressions:
            # TODO: Move this configure_compression function to this file
            from cerebras.modelzoo.common.utils.utils import (
                configure_compression,
            )

            self.compressions = configure_compression(compressions)
        else:
            self.compressions = []

    def setup(self, trainer):
        for compression in self.compressions:
            trainer.model.apply(compression)
