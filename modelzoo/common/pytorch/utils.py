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
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import torch
import yaml
from jsonschema import validate


def get_input_dtype(to_float16: bool):
    """Determine input datatype based on environment"""
    from modelzoo.common.pytorch import cb_model as cm

    try:
        from modelzoo.common.pytorch import amp

        # pylint: disable=protected-access
        half_dtype = amp._amp_state.half_dtype
    except:  # pylint: disable=bare-except
        from modelzoo.common.pytorch.run_utils import half_dtype_instance

        half_dtype = half_dtype_instance.half_dtype

    if (
        to_float16
        and not cm.use_cs()
        and not cm.is_appliance()
        and not torch.cuda.is_available()
    ):
        print(
            f"to_float16 == True for input dtype is not supported with vanilla "
            f"PyTorch CPU workflow. Setting to_float16 to False."
        )
        to_float16 = False
    dtype = half_dtype if to_float16 else torch.float32
    return dtype


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
):
    """Configure default logging format"""
    from modelzoo.common.pytorch import cb_model as cm

    class CustomFormatter(logging.Formatter):
        """Cerebras Preferred Log Formatting"""

        def __init__(self):
            ordinal = cm.get_ordinal()
            num_tasks = cm.num_tasks() - 1

            if num_tasks > 1 and cm.is_streamer():
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

    handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    handlers.append(handler)
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        logging_file = os.path.join(logging_dir, f"run_{time_stamp}.log")
        handler = logging.FileHandler(logging_file)
        handler.setFormatter(CustomFormatter())
        handlers.append(handler)

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

        # Custom levels defined by cerebras_appliance
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

    if cm.is_master_ordinal():
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
        logging.error(f"Uncaught exception:\n{msg}")
        if (
            original_hook != sys.__excepthook__
            and original_hook != cerebras_logging_hook
        ):
            original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = cerebras_logging_hook


def named_parameters_requiring_grad(model):
    """
    Returns the named paramters that should be passed to the optimizer
    i.e. are trainable because they require gradients.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def get_adaptive_lr_layers(model, lr_adjustment_layer_type):
    """
    Args:
        model: Pytorch model
        lr_adjustment_layer_type list: type of layer for which lr scaler is provided

    Returns:
        list: list of layer names for the given lr_adjustment_layer_type
    """
    adaptive_lr_layers = []
    for n, _ in model.named_parameters():
        if lr_adjustment_layer_type == 'decoder_kernel':
            if 'weight' in n:
                if 'linear' in n or 'dense' in n:
                    adaptive_lr_layers.append(n)
        elif lr_adjustment_layer_type == 'embedding':
            if 'embedding' in n and 'weight' in n:
                adaptive_lr_layers.append(n)
    return adaptive_lr_layers


def should_apply_weight_decay(model, param_name):
    """

    Args:
        model: Pytorch model
        param_name (torch.nn.Parameter): model param name

    Returns:
        bool: whether to apply weight decay for the give param_name
    """
    norm_modules = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.GroupNorm,
        torch.nn.SyncBatchNorm,
    )
    if 'bias' in param_name:
        return False
    for name, module in model.named_modules():
        if name in param_name:
            if isinstance(module, norm_modules):
                return False
    return True


def partition_params_groups_with_weight_decay(
    model, param_groups, weight_decay_rate
):
    """

    Args:
        model : Pytorch model
        param_groups (list): optimizer param_groups. Currently it will be just 1 group
        weight_decay_rate (float): value of weight decay rate from yaml

    Returns:
        list: param_groups as list of dicts, split based on the weight_decay rate
    """
    refined_params_groups = []
    for _ in range(2 * len(param_groups)):
        refined_params_groups.append({"params": []})
    for idx, param_group_ in enumerate(param_groups):
        # Set group's weight decay params
        refined_params_groups[2 * idx]["weight_decay"] = weight_decay_rate
        refined_params_groups[2 * idx + 1]["weight_decay"] = 0.0
        for name, param in param_group_["params"]:
            if should_apply_weight_decay(model, name):
                refined_params_groups[2 * idx]["params"].append((name, param))
            else:
                refined_params_groups[2 * idx + 1]["params"].append(
                    (name, param)
                )

    return refined_params_groups


def partition_params_groups_with_adjusted_lr(
    model, param_optimizer_grouped, lr_adjustment_layers, lr_adjustment_scalars,
):
    """
    Generates param_groups based on the lr_adjustment_layers
    Each lr adjustment layer_type will have a group asociated with it.

    Args:
        model : Pytorch model
        param_optimizer_grouped (list): param_groups before the split based on lr_adjustment_layers
        lr_adjustment_layers (list): list of layer types with different lr adjustment scalars
        lr_adjustment_scalars (list): lr adjustment scalars

    Returns:
        list: list of dicts of param groups
    """
    if lr_adjustment_layers:
        param_groups_with_lr_adjustment = []
        for param_group in param_optimizer_grouped:
            refined_param_groups = []
            for idx in range(len(lr_adjustment_layers) + 1):
                refined_param_groups.append(
                    {
                        "params": [],
                        "weight_decay": param_group["weight_decay"],
                        "adjust_learning_rate": lr_adjustment_scalars[idx]
                        if idx < len(lr_adjustment_layers)
                        else 1.0,
                    }
                )
            # collect all the params whose layer_type is not in lr_adjustment_layers
            # in the last param group
            adaptive_lr_layers = [
                get_adaptive_lr_layers(model, lr_adjust_layer_type_)
                for lr_adjust_layer_type_ in lr_adjustment_layers
            ]
            for name, param in param_group["params"]:
                param_in_adjust_lr_groups = False
                for idx, _ in enumerate(lr_adjustment_layers):
                    # check if param belongs to one of the adaptive lr layer types
                    if any(
                        adaptive_lr_layer_ in name
                        for adaptive_lr_layer_ in adaptive_lr_layers[idx]
                    ):
                        refined_param_groups[idx]["params"].append(
                            (name, param)
                        )
                        param_in_adjust_lr_groups = True
                # if param doesn't belongs to one of the adaptive lr layer types,
                # put it in the last refined_param_group
                if not param_in_adjust_lr_groups:
                    refined_param_groups[-1]["params"].append((name, param))

            # remove empty param groups
            refined_param_groups = [
                param_group_
                for param_group_ in refined_param_groups
                if param_group_["params"]
            ]
            param_groups_with_lr_adjustment.append(refined_param_groups)

    else:
        param_groups_with_lr_adjustment = param_optimizer_grouped

    # flatten the param group list if nested
    param_groups_with_lr_adjustment_flattened = []
    for groups in param_groups_with_lr_adjustment:
        if isinstance(groups, list):
            for group_ in groups:
                param_groups_with_lr_adjustment_flattened.append(group_)
        else:
            param_groups_with_lr_adjustment_flattened.append(groups)

    return param_groups_with_lr_adjustment_flattened


def monkeypatch_grad_scaler_step_if_finite():
    """
    Add torch.cuda.amp.GradScaler.step_if_finite API to match
    cbtorch.amp.GradScaler.
    """
    from torch.cuda.amp import GradScaler

    def _step_if_finite(self, optimizer, *args, **kwargs):
        # slightly modified GradScaler._maybe_opt_step, but considering all
        # optimizers' grads' finiteness.
        retval = None
        # pylint: disable=protected-access
        if not sum(
            v.item()
            for state in self._per_optimizer_states.values()
            for v in state["found_inf_per_device"].values()
        ):
            retval = optimizer.step(*args, **kwargs)
        return retval

    GradScaler.step_if_finite = _step_if_finite


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
