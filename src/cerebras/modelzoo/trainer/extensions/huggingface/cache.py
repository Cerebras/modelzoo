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

"""This module contains callbacks for handling HuggingFace cache directory."""

from cerebras.appliance.environment import appliance_environ
from cerebras.modelzoo.trainer.callbacks import Callback


class HFCacheDir(Callback):
    """A callback that sets up the HuggingFace cache directory."""

    def __init__(self, cache_dir: str):
        """
        Args:
            cache_dir: The cache directory to use for HuggingFace utilities.
        """
        self._cache_dir = cache_dir

    def setup(self, trainer):
        appliance_environ["TRANSFORMERS_CACHE"] = self._cache_dir
        appliance_environ["HF_HOME"] = self._cache_dir
        appliance_environ["HF_DATASETS_CACHE"] = self._cache_dir

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
