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

"""Utilities for generating synthetic data based on some specification."""

import torch
from torch.utils._pytree import (
    _dict_flatten,
    _dict_unflatten,
    _register_pytree_node,
    tree_flatten,
    tree_unflatten,
)

from cerebras.modelzoo.data.common.tensor_spec import TensorSpec

try:
    import cerebras.pytorch as cstorch
except:
    cstorch = None


def custom_dict_flatten(d: dict):
    """Constructs TensorSpec instances to contain the leaf nodes
    of the tree structure before flattening.
    """
    if "shape" or "dtype" or "tensor_factory" in d:
        return [[TensorSpec(**d)], "TensorSpec"]
    return _dict_flatten(d)


def custom_dict_unflatten(values, context):
    """After mapping the TensorSpecs to tensors/callables, return them directly
    as leaf nodes instead of reconstructing a dictionary when unflattening.
    """
    if context == "TensorSpec":
        return values[0]
    return _dict_unflatten(values, context)


class SyntheticDataProcessor:
    """Creates a synthetic dataset.

    Constructs a SyntheticDataset from the user-provided nested structure of
    input tensors and returns a torch.utils.data.DataLoader from the
    SyntheticDataset and the regular torch.utils.data.DataLoader inputs
    specified in params.yaml. The torch.utils.data.DataLoader is returned by
    calling the create_dataloader() method.

    Args:
        params: Dictionary containing dataset inputs and specifications.
            Within this dictionary, the user provides the additional
            'synthetic_inputs' field  that corresponds to a nested tree
            structure of input tensor specifications used to construct the
            SyntheticDataset.

        In params.yaml:
            data_processor: "SyntheticDataProcessor". Must set this input to
                use this class
            batch_size: int
            shuffle_seed: Optional[int] = None. If it is not None, then
                torch.manual_seed(seed=shuffle_seed) will be called when
                creating the dataloader.
            num_examples: Optional[int] = None. If it is not None, then
                the it specifies the number of examples/samples in the
                SyntheticDataset. Otherwise, the SyntheticDataset will
                generate samples indefinitely.

            .. regular torch.utils.DataLoader inputs
            ...
            synthetic_inputs:
                ..
                   shape: Collection of positive ints
                   dtype: PyTorch dtype
                   OR
                   tensor_factory: name of PyTorch function
                   args:
                        size:
                        dtype:
                        ...
    """

    def __init__(self, params):
        if cstorch is None:
            raise RuntimeError(
                f"Unable to import cerebras.pytorch. In order to use "
                f"SyntheticDataProcessor, please ensure you have access to "
                f"the cerebras_pytorch package."
            )

        # Regular torch.utils.DataLoader inputs
        self._batch_size = params.get("batch_size", None)
        if not self._batch_size:
            raise ValueError(
                f"No 'batch_size' field specified. Please enter a positive "
                f"integer batch_size."
            )
        if not isinstance(self._batch_size, int) or self._batch_size <= 0:
            raise ValueError(
                f"Expected batch_size to be a positive integer but got "
                f"{self._batch_size}."
            )

        self._shuffle = params.get("shuffle", False)
        self._sampler = params.get("sampler", None)
        self._batch_sampler = params.get("batch_sampler", None)
        self._num_workers = params.get("num_workers", 0)
        self._pin_memory = params.get("pin_memory", False)
        self._drop_last = params.get("drop_last", False)
        self._timeout = params.get("timeout", 0)

        # SyntheticDataset specific inputs
        self._seed = params.get("shuffle_seed", None)
        self._num_examples = params.get("num_examples", None)
        if self._num_examples is not None:
            if (
                not isinstance(self._num_examples, int)
                or self._num_examples <= 0
            ):
                raise ValueError(
                    f"Expected num_examples to be a positive integer but got "
                    f"{self._num_examples}."
                )

        if self._drop_last and self._num_examples < self._batch_size:
            raise ValueError(
                f"This dataset does not return any batches because number of "
                f"examples in the dataset ({self._num_examples}) is less than "
                f"the batch size ({self._batch_size}) and `drop_last` is True."
            )

        self._tensors = []
        synthetic_inputs = params.get("synthetic_inputs", {})
        if synthetic_inputs:
            _register_pytree_node(
                dict, custom_dict_flatten, custom_dict_unflatten
            )
            leaf_nodes, self._spec_tree = tree_flatten(synthetic_inputs)
            for tensor_spec in leaf_nodes:
                if not isinstance(tensor_spec, TensorSpec):
                    raise TypeError(
                        f"Expected all leaf nodes in 'synthetic_inputs' to be "
                        f"of type TensorSpec but got {type(tensor_spec)}. "
                        f"Please ensure that all leaf nodes under "
                        f"'synthetic_inputs' are instances of TensorSpec. "
                        f"These instances are created by specifying either a "
                        f"'shape' and 'dtype' keys or a 'tensor_factory' "
                        f"key in a dict (mutually exclusive)."
                    )
                self._tensors.append(self._process_tensor(tensor_spec.specs))
            self._tensor_specs = tree_unflatten(self._tensors, self._spec_tree)
            _register_pytree_node(dict, _dict_flatten, _dict_unflatten)
        else:
            raise ValueError(
                f"Expected 'synthetic_inputs' field but found none. Please "
                f"specify this field and provide tensor information according "
                f"to the documentation."
            )

    def _torch_dtype_from_str(self, dtype):
        """Takes in the user input string for dtype and returns the
        corresponding torch.dtype.
        """
        torch_dtype = getattr(torch, dtype, None)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"Invalid torch dtype '{dtype}'. Please ensure all tensors use "
                f"a valid torch dtype."
            )
        return torch_dtype

    def _process_tensor(self, tensor_spec):
        """Parses the tensor_spec and returns a corresponding synthetic tensor."""

        if not tensor_spec:
            raise ValueError(
                f"Empty TensorSpec found. Please provide at least a 'shape' "
                f"and 'dtype' field to complete the tensor specification."
            )
        shape = tensor_spec.get("shape", None)
        dtype = tensor_spec.get("dtype", None)
        tensor_factory = tensor_spec.get("tensor_factory", None)

        # Enforce mutually exclusive inputs
        mutex = shape and dtype and not tensor_factory
        mutex = mutex or (not shape and not dtype and tensor_factory)
        if not mutex:
            possible_inputs = ['shape', 'dtype', 'tensor_factory']
            found = [
                i
                for i, j in locals().items()
                if i in possible_inputs and j is not None
            ]
            raise ValueError(
                f"Expected either 'shape' and 'dtype' fields or 'tensor_factory' "
                f"field specified (mutually exclusive) but instead found the "
                f"following fields: {found}. Please ensure each tensor either "
                f"has a 'shape' and 'dtype' field OR a 'tensor_factory' field."
            )

        if shape and dtype:
            if not all(isinstance(e, int) and e > 0 for e in shape):
                raise ValueError(
                    f"Expected shape to be a collection of positive integers "
                    f"but got {shape}. Please ensure all tensor shapes are "
                    f"collections of positive integers."
                )
            torch_dtype = self._torch_dtype_from_str(dtype)
            return torch.zeros(shape, dtype=torch_dtype)

        elif tensor_factory:
            torch_args = tensor_spec.get("args", None)
            if not torch_args:
                raise ValueError(
                    f"Expected 'args' field but found none for the "
                    f"tensor_factory '{tensor_factory}'. Please specify this "
                    f"field and fill it with the arguments for the chosen "
                    f"tensor generation function."
                )
            if not torch_args.get("dtype", None):
                raise ValueError(
                    f"Expected 'dtype' argument for tensor_factory '{tensor_factory}' "
                    f"in the 'args' field, but found none. Please specify this "
                    f"argument with the desired tensor dtype."
                )
            torch_dtype = self._torch_dtype_from_str(torch_args["dtype"])
            torch_args["dtype"] = torch_dtype

            # Raises torch AttributeError if the provided function is invalid
            try:
                test_tensor = getattr(torch, tensor_factory)(**torch_args)
            except Exception as e:
                raise ValueError(
                    f"Provided tensor_factory '{tensor_factory}' is invalid "
                    f"Please ensure you are using a supported PyTorch callable "
                    f"that returns a torch tensor."
                ) from e

            if not isinstance(test_tensor, torch.Tensor):
                raise ValueError(
                    f"Expected tensor_factory {tensor_factory} to return a "
                    f"torch.Tensor but instead got {type(test_tensor)}. Please "
                    f"ensure that tensor_factory contains a valid PyTorch "
                    f"callable that returns a torch tensor."
                )

            return lambda x: getattr(torch, tensor_factory)(**torch_args)

    def create_dataloader(self):
        """Returns torch.utils.data.DataLoader that corresponds to the created
        SyntheticDataset.
        """
        if self._shuffle and self._seed is not None:
            torch.manual_seed(self._seed)
        return torch.utils.data.DataLoader(
            cstorch.utils.data.SyntheticDataset(
                self._tensor_specs, num_samples=self._num_examples
            ),
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            sampler=self._sampler,
            batch_sampler=self._batch_sampler,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
            timeout=self._timeout,
        )
