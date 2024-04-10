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
Config classes of Sparsity Configs

"""
# pylint: disable=wildcard-import
import copy
from typing import Union

from cerebras.modelzoo.config_manager.config_classes.base.base_config import *
from cerebras.pytorch.sparse import map_sparsity_algorithm


@dataclass
class SparsityAlgorithmConfig(BaseConfig):
    algorithm: Literal["static", "gmp", "set", "rigl"] = "static"
    """
    Sparsity Algorithm to apply. ["static", "gmp", "set", "rigl"]
    """

    params: dict = required
    """
    Parameters such as init_method, sparsity, param_name_patterns, schedule etc
    These vary based on the type of sparsity chosen.
    Please see sparsity documentaion for valid parameters for each type
    """

    def __init__(self, **kwargs):
        for field_name, field_type in self.__annotations__.items():
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
        self.params = kwargs
        super().__init__()

    def __post_init__(self):
        param_dict = copy.deepcopy(asdict(self))
        # This will error out if we are not able to map the params to
        # the mentioned sparsity class
        map_sparsity_algorithm(algorithm=self.algorithm)
        super().__post_init__()


@dataclass
class SparsityBaseConfig(BaseConfig):
    groups: List[SparsityAlgorithmConfig] = required

    # Custom init for sparsity, where we want to capture all fixed sparsity params as members.
    # sparsity params is a dict that is populated by checking all additional params supplied to us.
    # These are sparsity specific and we use signature of that sparsity to validate these.
    def __init__(self, *args, **kwargs):
        "Sparsity related params: see Sparsity documentation"
        if len(args) > 0:
            if len(kwargs) > 0:
                raise ValueError(
                    "Expected a list of dicts or a single dict, not both"
                )
            self.groups = [SparsityAlgorithmConfig(**group) for group in args]
        else:
            self.groups = [SparsityAlgorithmConfig(**kwargs)]
        super().__init__()


SparsityConfig = Union[float, SparsityBaseConfig]
