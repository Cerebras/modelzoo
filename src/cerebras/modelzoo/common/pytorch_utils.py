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

"""General purpose Pytorch Utilities"""
import argparse
import logging
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from warnings import warn

import torch
import yaml
from jsonschema import validate
from packaging.version import parse

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.appliance.utils.file import create_symlink


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
    """Move tensor from device to cpu"""
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


def setup_logging(
    chief_logging_level: str,
    streamer_logging_level: str,
    logging_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
):
    """Configure default logging format"""

    class CustomFormatter(logging.Formatter):
        """Cerebras Preferred Log Formatting"""

        def __init__(self):
            ordinal = dist.get_ordinal()
            num_tasks = dist.num_tasks() - 1

            if num_tasks > 1 and dist.is_streamer():
                ordinal_msg = f"[{ordinal}/{num_tasks}]"
            else:
                ordinal_msg = ""

            fmt = f"%(asctime)s %(levelname)s: {ordinal_msg}  %(message)s"
            super().__init__(fmt=fmt)

            self.info_formatter = None
            # Only enable shorter info logging depending on environment variable
            # This is so that we have the option to experiment with this in the future
            if "USE_SHORT_INFO_LOGGING" in os.environ:
                fmt = "{}%(message)s".format(
                    f"{ordinal_msg}:  " if ordinal > 0 else ""
                )
                self.info_formatter = logging.Formatter(fmt)

        def format(self, record):
            if self.info_formatter and record.levelno == logging.INFO:
                return logging.Formatter.format(self.info_formatter, record)

            return super().format(record)

    def build_block_filter(handler_type: str):
        """Build a filter to block records from a specific handler."""

        def block_filter(record):
            if hasattr(record, "block"):
                return record.block != handler_type
            return True

        return block_filter

    handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    handler.addFilter(build_block_filter("console"))
    handlers.append(handler)
    if logging_dir:
        logging_file = os.path.join(logging_dir, f"run.log")
        handler = logging.FileHandler(logging_file)
        handler.setFormatter(CustomFormatter())
        handler.addFilter(build_block_filter("file"))
        handlers.append(handler)
        # set up run log symlink
        symlink_dir = Path(model_dir) if model_dir else Path(logging_dir)
        run_log_symlink = symlink_dir / "latest_run.log"
        create_symlink(
            run_log_symlink, Path(logging_file).relative_to(symlink_dir)
        )

    def get_level_name(level):
        if not isinstance(level, str):
            raise ValueError(
                f"Invalid logging level: `{level}`. "
                f"Expected a string or int level."
            )

        try:
            level = int(level)
        except ValueError:
            level = level.upper()

        # Custom levels defined by cerebras.appliance
        if level == "TRACE":
            level = logging.DEBUG - 5
        elif level == "VERBOSE":
            level = logging.INFO - 5
        else:
            if (
                isinstance(level, str)
                and level not in logging._nameToLevel  # pylint: disable=W0212
            ):
                # pylint: disable=protected-access
                raise ValueError(
                    f"Invalid logging level: `{level}`. Expected one of "
                    f"{list(logging._nameToLevel.keys())}."
                )

            level = logging.getLevelName(level)

        return level

    if dist.is_master_ordinal():
        level = get_level_name(chief_logging_level or "info")
    else:
        level = get_level_name(streamer_logging_level or "error")

    # Remove any handlers that may have been inadvertently set before
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=level, handlers=handlers)

    original_hook = sys.excepthook

    def cerebras_logging_hook(exc_type, exc_value, exc_traceback):
        """Pipe uncaught exceptions through logger"""
        msg = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        # Block console logging to avoid duplicate messages since exceptions
        # are logged by python interpreter by default anyways.
        logging.error(f"Uncaught exception:\n{msg}", extra={"block": "console"})

        # Run the original except hook which prints the exception to stderr
        original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = cerebras_logging_hook


def setup_artifact_dir(model_dir: str, mode: str):
    """
    Create a unique subdirectory for this run by generating a time stamp so
    that parallel runs using the same model_dir don't overwrite common files.
    """
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    cerebras_logs_path = Path(model_dir) / "cerebras_logs"
    artifact_dir = cerebras_logs_path / mode / time_stamp
    artifact_dir.mkdir(parents=True)

    # Create a symlink to the artifact_dir so that it's easy to find the latest run.
    # The symlink needs to be at the same level as the subdirectories.
    latest = cerebras_logs_path.joinpath("latest")
    # symlink to relative path
    create_symlink(
        latest,
        artifact_dir.relative_to(cerebras_logs_path),
        target_is_directory=True,
    )
    return str(artifact_dir)


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
        """Generate next data sample"""
        if self._count >= self._sample_count:
            raise StopIteration
        self._count += 1
        return self._data


class RunConfigParamsValidator:
    """Validate Run Configs"""

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
        """Validate params match existing schema"""

        if "use_cs_grad_accum" in config:
            raise ValueError(
                f"use_cs_grad_accum is no longer a valid option. To control gradient accumulation "
                f"settings on CSX, set micro_batch_size: (\"auto\" | None) in the "
                f"train_input and/or eval_input section of the params yaml file."
            )

        validate(instance=config, schema=self.runconfig_schema)


def get_checkpoints(model_dir: str) -> List[str]:
    """Gather checkpoints in a model directory"""
    matches = []
    for filename in os.listdir(model_dir):
        m = re.match(r"checkpoint_(\d+)\.mdl", filename)
        if m:
            matches.append(m)
    matches.sort(key=lambda x: int(x.group(1)))  # Sort by index not lexically
    checkpoints = [os.path.join(model_dir, match.group()) for match in matches]
    return checkpoints


def is_mup_run(params):
    """
    Check if the run is configured with muP hyperparameter settings
    """
    scale_qk_dot_by_d = params.get('model', {}).get('scale_qk_dot_by_d', False)
    embeddings_scale = params.get('model', {}).get('embeddings_scale', None)
    output_logits_scale = params.get('model', {}).get(
        'output_logits_scale', None
    )
    runconfig_params = params.get('runconfig', {})

    if runconfig_params.get('mode', None) == 'train':
        adjust_learning_rate = (
            params.get('optimizer', {})
            .get('adjust_learning_rate', {})
            .get('decoder_kernel', {})
        )
        return (
            scale_qk_dot_by_d
            and embeddings_scale
            and output_logits_scale
            and adjust_learning_rate
        )
    elif runconfig_params.get('mode', None) == 'eval':
        return scale_qk_dot_by_d and embeddings_scale and output_logits_scale
    return False


def load_from_checkpoint_file(
    checkpoint_path: str, check_compatibility: bool = True
) -> dict:
    """Loads state dict from checkpoint path and checks for version compatibilty."""
    logging.info(f"Loading weights from checkpoint {checkpoint_path}")
    state_dict = cstorch.load(checkpoint_path)
    if check_compatibility:
        check_checkpoint_compatibility(state_dict)
    return state_dict


def check_checkpoint_compatibility(state_dict: Dict[str, Any]):
    """Checks that the checkpoint is compatible with the current version of modelzoo."""
    import cerebras.modelzoo as modelzoo
    import cerebras.modelzoo.tools.convert_checkpoint as convert_ckpt

    if "__metadata__" in state_dict:
        # extract the last item in the list as this is the most recent metadata
        checkpoint_version = state_dict["__metadata__"][-1].get("version", "")
        if not checkpoint_version:
            return
        checkpoint_version = parse(checkpoint_version)
        if checkpoint_version.local is None:
            current_version = parse(cstorch.__version__)
            if (
                checkpoint_version.major != current_version.major
                or checkpoint_version.minor != current_version.minor
            ):
                converter_path = os.path.relpath(
                    convert_ckpt.__file__,
                    os.path.dirname(modelzoo.__file__),
                )
                warn(
                    f"Checkpoint version may be incompatible with Modelzoo version. Got "
                    f"checkpoint version {str(checkpoint_version)} but Modelzoo version "
                    f"is {str(current_version)}. You may need to run {converter_path} on the "
                    f"incompatible checkpoint."
                )
