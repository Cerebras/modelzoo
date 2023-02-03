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

import argparse
import collections
import logging
import os
import random
import sys
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import torch
import yaml


class ExecutionStrategy:
    pipeline = "pipeline"
    weight_streaming = "weight_streaming"

    @classmethod
    def strategies(cls):
        return [cls.pipeline, cls.weight_streaming]


def read_params_file(params_file: str) -> dict:
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def get_debug_args(debug_args_path, debug_ini_path):
    """Appliance mode DebugArgs."""
    from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
        DebugArgs,
    )
    from cerebras_appliance.run_utils import get_debug_args as parse_debug_args

    if debug_args_path:
        debug_args = parse_debug_args(debug_args_path)
    else:
        debug_args = DebugArgs()

    debug_ini_fp = os.path.join(debug_ini_path, "debug.ini")
    if os.path.exists(debug_ini_fp):
        with open(debug_ini_fp, "r") as fp:
            ini_content = fp.read()
            debug_args.debug_ini.frozen_content = ini_content
    return debug_args


def get_params(params_file: str, config: Optional[str] = None,) -> dict:
    """Reads params from file and returns them as a dict.

    Args:
        params_file: The YAML file holding the params.
        config: Optional config to load from the params. If None, the default
            config is returned. Defaults to None.
    Returns:
        A dict containing the params.
    """
    params = read_params_file(params_file)
    if config:
        params = params[config]

    return params


def update_defaults(params: dict, default_params: dict) -> dict:
    """Updates the params dict with global default for a key
    if a key is not present.
    Works on nested dictionaries and recursively updates defaults
    for nested dictionaries.
    All other types, apart from dict are considered as base type
    and aren't updated recursively.
    Args:
        params: dict holding the params.
        default_params: dict holding the default params.
    Returns:
        A dict containing the params, with the defaults updated
    """
    for k, v in default_params.items():
        if isinstance(v, collections.abc.Mapping):
            params[k] = update_defaults(params.get(k, {}), v)
        elif k not in params:
            params[k] = v
    return params


def update_params_from_args(args: argparse.Namespace, params: dict):
    """Update params in-place with the arguments from args.

    Args:
        args: The namespace containing args from the commandline.
        params: The params to be updated.
    """
    if args:
        for (k, v) in list(vars(args).items()):
            params[k] = v if v is not None else params.get(k)

    if params.get("is_pretrained_checkpoint") and not params.get(
        "checkpoint_path"
    ):
        raise RuntimeError(
            "'--is_pretrained_checkpoint' can only be used if a "
            "'--checkpoint_path' is provided."
        )

    mode = params.get("mode")
    if mode != "train" and params.get("multireplica"):
        logging.warning(
            f"Multireplica is only supported in `train` mode. Disabling it for "
            f"{mode} mode."
        )

    model_dir = params["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    params.setdefault("service_dir", model_dir)


def update_params_from_file(params, params_file):
    if os.path.exists(params_file):
        default_params = read_params_file(params_file)
        update_defaults(params, default_params)


def parse_cs_ip(cs_ip: str):
    if cs_ip and len(str(cs_ip).split(":")) == 1:
        cs_ip = f"{cs_ip}:9000"  # Default port number of CM
    return cs_ip


def get_parser(
    run_dir: Optional[str] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
) -> argparse.ArgumentParser:
    """Returns an ArgumentParser for parding commandline options.

    Returns:
        A parser instance.
    """
    parents = extra_args_parser_fn() if extra_args_parser_fn else []
    parser = argparse.ArgumentParser(parents=parents)

    default_model_dir = None
    # Set default model dir to be inside same directory
    # as the top level run.py
    if run_dir:
        default_model_dir = os.path.join(run_dir, "model_dir")

    if not default_model_dir:
        raise ValueError("Could not get default model directory")

    parser.add_argument(
        "-cs",
        "--cs_ip",
        type=parse_cs_ip,
        default=None,
        help="IP Address of Cerebras System",
    )
    parser.add_argument(
        "-cpu",
        "--cpu",
        action="store_true",
        default=None,
        help="Use CPU even if GPU is present in non-CS workflow",
    )
    parser.add_argument(
        "-dist_addr",
        "--dist_addr",
        default="localhost:8888",
        help="To init master_addr and master_port of distributed, ex. localhost:8888",
    )
    parser.add_argument(
        "-dist_backend",
        "--dist_backend",
        choices=["nccl", "mpi", "gloo"],
        default="nccl",
        help="Distributed backend engine",
    )
    parser.add_argument(
        "-init_method",
        "--init_method",
        default="env://",
        help="URL specifying how to initialize the process group",
    )
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Specifies a specific key of the params file to return",
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["train", "eval", "train_and_eval"],
        help="Can train, eval, or train_and_eval.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        default=None,
        help="Enables compile only workflow",
    )
    parser.add_argument(
        "--multireplica",
        action="store_true",
        default=None,
        help="Enables multireplica mode",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        default=None,
        help="Enables validate only workflow"
        "validate_only stops the compilation at ws_km stage for weight streaming mode."
        "for pipeline mode, the compilation is stopped at the optimize_graph stage.",
    )
    parser.add_argument(
        "--appliance",
        action="store_true",
        default=None,
        help="Enables appliance mode training/evaluation",
    )
    parser.add_argument(
        "--execution_strategy",
        choices=ExecutionStrategy.strategies(),
        type=str,
        default=None,
        help="Execution strategy for running the model. For appliance mode, it "
        "defaults to weight_streaming. For non-appliance mode it "
        "defaults to pipeline",
    )
    parser.add_argument(
        "--fabric_config_file",
        type=str,
        default=None,
        help="Path to the fabric config file",
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        default=default_model_dir,
        help="Model directory where checkpoints will be written.",
    )
    parser.add_argument(
        "-c",
        "--compile_dir",
        default=None,
        help="Compile directory where compile artifacts will be written.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )
    parser.add_argument(
        "--is_pretrained_checkpoint",
        action="store_true",
        help=(
            "Flag indicating that the provided checkpoint is from a "
            "pre-training run. If set, training will begin from step 0 "
            "after loading the matching weights from the checkpoint and "
            "ignoring the optimizer state if present in the checkpoint."
        ),
    )
    parser.add_argument(
        "--logging",
        default=None,
        help="Specifies the default logging level. Defaults to INFO.",
    )

    # Appliance mode specific arguments
    parser.add_argument(
        "--credentials_path",
        default=None,
        help="credentials for cluster access",
    )
    parser.add_argument(
        "--mgmt_address",
        default="cluster-server.cerebras.com:443",
        help="<host>:<port> for cluster management",
    )
    parser.add_argument(
        "--num_csx", default=1, type=int, help="number of CS nodes",
    )
    parser.add_argument(
        "--num_wgt_servers",
        default=None,
        type=int,
        help="Maximum number of weight servers to use in weight streaming "
        "execution strategy.",
    )
    parser.add_argument(
        "--num_workers_per_csx",
        default=0,
        type=int,
        help="Number of workers to use for streaming inputs per CS node. If "
        "0, a default value based on the model will be chosen. Defaults "
        "to 0.",
    )
    parser.add_argument(
        "--debug_args_path", default=None, help="path to debugs args file",
    )
    parser.add_argument(
        "--mount_dirs",
        nargs="+",
        help="a list of paths to be mounted to the appliance containers",
    )
    parser.add_argument(
        "--python_paths",
        nargs="+",
        help="a list of paths to be exported into PYTHONPATH for worker containers",
    )
    parser.add_argument(
        "--transfer_processes",
        type=int,
        default=None,
        help="Number of processes to use when transferring weights.",
    )
    parser.add_argument(
        "--job_labels",
        nargs="+",
        help="a list of equal-sign-separated key value pairs served as job labels",
    )
    return parser


def get_params_from_args(
    run_dir: Optional[str] = None,
    argv: Optional[List] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
) -> dict:
    """
    Parse the arguments and get the params dict from the resulting args

    Args:
        run_dir: The path to the `run.py` file
        argv: The args to be parse. Defaults to sys.argv if not provided
    """
    parser = get_parser(run_dir, extra_args_parser_fn)
    args = parser.parse_args(argv if argv else sys.argv[1:])
    params = get_params(args.params, args.config)
    update_params_from_args(args, params["runconfig"])
    return params


def get_input_dtype(to_float16: bool):
    from modelzoo.common.pytorch import cb_model as cm

    try:
        from modelzoo.common.pytorch import amp

        half_dtype = amp._amp_state.half_dtype
    except:
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


class BufferedShuffleDataset(torch.utils.data.IterableDataset):
    r"""Dataset shuffled from the original dataset.
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
        >>> ds = BufferedShuffleDataset(dataset)
        >>> random.seed(...)
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
    For multi-process mode (:attr:`num_workers > 0`), the random seed is set by a callable
    function in each worker.
        >>> ds = BufferedShuffleDataset(dataset)
        >>> def init_fn(worker_id):
        ...     random.seed(...)
        >>> print(list(torch.utils.data.DataLoader(ds, ..., num_workers=n, worker_init_fn=init_fn)))
    Arguments:
        dataset (IterableDataset): The original IterableDataset.
        buffer_size (int): The buffer size for shuffling.
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


def to_cpu(tensor):
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


def setup_logging(chief_logging_level: str, streamer_logging_level: str):
    from modelzoo.common.pytorch import cb_model as cm

    class CustomFormatter(logging.Formatter):
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

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())

    def get_level_name(level):
        level = level.upper()
        assert level in (
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        ), f"Invalid logging level: {level}"
        return logging.getLevelName(level)

    if cm.is_master_ordinal() or cm.is_sync_mode():
        level = get_level_name(chief_logging_level or "info")
    else:
        level = get_level_name(streamer_logging_level or "error")

    # Remove any handlers that may have been inadvertently set before
    logging.getLogger().handlers.clear()

    logging.basicConfig(level=level, handlers=[handler])


def group_optimizer_params(
    trainable_params, no_decay_layers, weight_decay_rate
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in trainable_params
                if not any(nd in n for nd in no_decay_layers)
            ],
            "weight_decay": weight_decay_rate,
        },
        {
            "params": [
                p
                for n, p in trainable_params
                if any(nd in n for nd in no_decay_layers)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


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
        if self._count >= self._sample_count:
            raise StopIteration
        self._count += 1
        return self._data
