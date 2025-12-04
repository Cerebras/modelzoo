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

"""General purpose Pytorch Utilities."""

import argparse
import os
import random
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import torch
import yaml
from jsonschema import validate


def visit_structure(
    data_structure: Union[Any, list, tuple, dict],
    select_fn: Callable[[Any], bool],
    strict: bool = False,
    scope: Optional[List[str]] = None,
) -> Generator[Tuple[List[str], Any], None, None]:
    """Recursively traverse nested structure and return the items accepted by
    the selector.

    Args:
        data_structure: A nested data structure to traverse recursively.
        select_fn: A callable that returns true if the item passed should be
            selected.
        strict: Strictly checks that an item in the nested structure is either
            a list/dict/tuple or selected by the select_fn. Otherwise, raises
            an error. Defaults to False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A tuples of (scope, item) for each item selected by the select_fn.
    """
    scope = scope or []
    if isinstance(data_structure, (list, tuple)):
        for i, v in enumerate(data_structure):
            yield from visit_structure(v, select_fn, strict, scope + [str(i)])
    elif isinstance(data_structure, dict):
        for k, v in data_structure.items():
            yield from visit_structure(v, select_fn, strict, scope + [str(k)])
    elif select_fn(data_structure):
        yield scope, data_structure
    elif strict:
        raise ValueError(f"Unknown data structure: {data_structure}")


class BufferedShuffleDataset(
    torch.utils.data.IterableDataset
):  # pylint:disable=abstract-method
    """Dataset shuffled from the original dataset.

    This class is useful to shuffle an existing instance of an IterableDataset.
    The buffer with `buffer_size` is filled with the items from the dataset first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.
    `buffer_size` is required to be larger than 0. For `buffer_size == 1`, the
    dataset is not shuffled. In order to fully shuffle the whole dataset, `buffer_size`
    is required to be greater than or equal to the size of dataset.
    When it is used with :class:`~torch.utils.data.DataLoader`, each item in the
    dataset will be yielded from the :class:`~torch.utils.data.DataLoader` iterator.
    And, the method to set up a random seed is different based on :attr:`num_workers`.
    For single-process mode (:attr:`num_workers == 0`), the random seed is required to
    be set before the :class:`~torch.utils.data.DataLoader` in the main process.

    Arguments:
        dataset (IterableDataset): The original IterableDataset.
        buffer_size (int): The buffer size for shuffling.

    Example:
        For multi-process mode (:attr:`num_workers > 0`), the random seed is set by a callable
        function in each worker.

        >>> ds = BufferedShuffleDataset(dataset)
        >>> random.seed(...)
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        >>> ds = BufferedShuffleDataset(dataset)
        >>> def init_fn(worker_id):
        ...     random.seed(...)
        >>> print(list(torch.utils.data.DataLoader(ds, ..., num_workers=n, worker_init_fn=init_fn)))
    """

    def __init__(self, dataset, buffer_size):
        super(BufferedShuffleDataset, self).__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()

    def __len__(self):
        return len(self.dataset)


class IterableDatasetSampler(
    torch.utils.data.IterableDataset
):  # pylint:disable=abstract-method
    """
    This sampler can be used with a multi-worker distributed dataloader.
    All workers on all nodes get a copy of the IterableDataset but only yield
    samples according to the world size and their rank.
    """

    def __init__(self, iterable_dataset, world_size=1, rank=0):
        self.iterable_dataset = iterable_dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        mod = self.world_size
        shift = self.rank

        if worker_info:
            mod *= worker_info.num_workers
            shift = self.rank * worker_info.num_workers + worker_info.id

        for i, element in enumerate(self.iterable_dataset):
            if (shift + i) % mod == 0:
                yield element


def to_cpu(tensor):
    """Move tensor from device to cpu."""
    if isinstance(tensor, torch.Tensor):
        return tensor.to("cpu")
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(
            t.to("cpu") if isinstance(t, torch.Tensor) else t for t in tensor
        )
    if isinstance(tensor, dict):
        return {
            k: t.to("cpu") if isinstance(t, torch.Tensor) else t
            for k, t in tensor.items()
        }

    raise TypeError(
        "Invalid type. Expected Tensor or list/tuple of Tensors. "
        f"Got: {type(tensor)}"
    )


def to_tensor(value, device=None):
    """
    If the provided value is a Python int or float, it converts them
    into PyTorch Tensors of type int32 and float32 respectively.
    Otherwise, it just returns the value.
    """
    if isinstance(value, int):
        return torch.tensor(value, dtype=torch.int32, device=device)
    elif isinstance(value, float):
        return torch.tensor(value, dtype=torch.float32, device=device)
    elif isinstance(value, tuple):
        return tuple(map(to_tensor, value))
    elif isinstance(value, list):
        return list(map(to_tensor, value))
    else:
        return value


class SampleGenerator(object):
    """Iterator which returns multiple samples of a given input data.

    Can be used in place of a PyTorch `DataLoader` to generate synthetic data.

    Args:
        data: The data which should be returned at each iterator step.
        sample_count: The maximum number of `data` samples to be returned.
    """

    def __init__(self, data, sample_count):
        self._data = data
        self._sample_count = sample_count
        self._count = 0

    def __iter__(self):
        return SampleGenerator(self._data, self._sample_count)

    def __len__(self):
        return self._sample_count

    def __next__(self):
        return self.next()

    def next(self):
        """Generate next data sample."""
        if self._count >= self._sample_count:
            raise StopIteration
        self._count += 1
        return self._data


class RunConfigParamsValidator:
    """Validate Run Configs."""

    def __init__(
        self,
        extras: Optional[Callable[[], List[argparse.ArgumentParser]]] = None,
    ):
        with open(
            os.path.join(
                os.path.dirname(__file__), "schema/runconfig_schema.yaml"
            ),
            "r",
        ) as fin:
            self.runconfig_schema = yaml.safe_load(fin)

        if extras:
            for parser in extras():
                for arg in parser._actions:
                    self.runconfig_schema["properties"][arg.dest] = {}

    def validate(self, config):
        """Validate params match existing schema."""

        if "use_cs_grad_accum" in config:
            raise ValueError(
                f"use_cs_grad_accum is no longer a valid option. To control gradient accumulation "
                f"settings on CSX, set micro_batch_size: (\"auto\" | None) in the "
                f"train_input and/or eval_input section of the params yaml file."
            )

        validate(instance=config, schema=self.runconfig_schema)
