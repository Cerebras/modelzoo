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

from cerebras.modelzoo.models.vision.dit.samplers.DDIMSampler import DDIMSampler
from cerebras.modelzoo.models.vision.dit.samplers.DDPMSampler import DDPMSampler

SMP2FN = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
}


def get_sampler(sampler):
    """
    Return sampler based on str `sampler`
    """
    if sampler is not None:
        sampler = sampler.lower()
    if sampler in SMP2FN:
        return SMP2FN[sampler]
    else:
        raise KeyError(
            f"function {sampler} not found in mapping {list(SMP2FN.keys())}"
        )


def get_all_samplers():
    """
    Return dictionary of all samplers
    """
    return SMP2FN
