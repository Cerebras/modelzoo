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

"""Wrapper class used to process TensorSpecs using a custom yaml tag."""

import yaml


class TensorSpec:
    """Wrapper class used to wrap the leaf nodes in SyntheticDataProcessor's
    input.

    TensorSpecs hold a dictionary of arguments used to specify a tensor. An
    instance of this class is constructed to wrap a dictionary if the dictionary
    in the input contains at least one of 'shape', 'dtype', or 'tensor_factory'
    keys.

    Example list element format in yaml file:
      shape: ...
      dtype: ...
    This class merely holds the provided dictionary of kwargs. See
    models/common/pytorch/input/SyntheticDataProcessor.py for more docs and
    use cases.

    Args:
        kwargs: Any variable number of keyword arguments written as a dictionary
        under the tag in the .yaml file as seen in the example above.
    """

    def __init__(self, **kwargs):
        self.specs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}, specs={self.specs}"


def tensor_spec_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
):
    """Constructor used to register TensorSpec in the yaml loader."""
    try:
        return TensorSpec(**loader.construct_mapping(node))
    except:
        raise ValueError(
            f"Empty TensorSpec found. Please provide at least a 'shape' "
            f"and 'dtype' field to complete the tensor specification."
        )
