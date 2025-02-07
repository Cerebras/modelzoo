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

from abc import ABC, abstractmethod


class SamplerBase(ABC):
    @abstractmethod
    def set_timesteps(
        self, num_diffusion_steps, num_inference_steps, custom_timesteps
    ):
        """
        Computes timesteps to be used during sampling

        Args:
            num_diffusion_steps (`int`): Total number of steps the model was trained on
            num_inference_steps (`str`): string containing comma-separated numbers,
                indicating the step count per section.
                For example, if there's 300 `num_diffusion_steps` and num_inference_steps=`10,15,20`
                then the first 100 timesteps are strided to be 10 timesteps, the second 100
                are strided to be 15 timesteps, and the final 100 are strided to be 20.
                Can either pass `custom_timesteps` (or) `num_inference_steps`, but not both.
            custom_timesteps (`List[int]`): User specified list of timesteps to be used during sampling.
        """

    @abstractmethod
    def previous_timestep(self, timestep):
        """
        Returns the previous timestep based on current timestep.
        Depends on the timesteps computed in `self.set_timesteps`
        """

    @abstractmethod
    def step(self):
        """
        Predict the sample at the previous timestep by reversing the SDE.
        Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        """
