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
Config classes of Run Configs
"""
from typing import Dict, List, Literal, Optional, Union

# pylint: disable=wildcard-import
from cerebras.modelzoo.config_manager.config_classes.base.base_config import *
from cerebras.modelzoo.config_manager.config_validators import PositiveInteger


@dataclass
class PytorchProfilerConfig(BaseConfig):
    start_step: int = required
    "step where to begin profiling"

    end_step: int = required
    "step where to end profiling"


@dataclass
class RunConfig(BaseConfig):
    steps_per_epoch: Optional[int] = None
    "The number of steps per epoch."

    max_steps: Optional[int] = None
    """
    Specifies the maximum number of steps for training.
    `max_steps` is optional unless neither `num_epochs`
    nor `num_steps` are provided,
    """

    mgmt_address: Optional[str] = None
    """
    The address of the management service used for coordinating the training job as
    `<host>:<port>`.
    """

    mount_dirs: Optional[List[str]] = None
    """
    A list of paths to be mounted to the appliance containers.
    It should generally contain path to the directory
    containing the Cerebras model zoo and data dir.
    """
    num_epochs: Optional[int] = None
    "The number of epochs to train for."

    python_paths: Optional[List[str]] = None
    """
    A list of paths to be exported into `PYTHONPATH` for worker containers.
    It should generally contain path to
    the directory containing the Cerebras model zoo.
    """

    compile_dir: Optional[str] = None
    "Compile directory where compile artifacts will be written."

    checkpoint_path: Optional[str] = None
    "The path to load checkpoints from during training."

    credentials_path: Optional[str] = None
    """
    Credentials for cluster access. If `None`, the value from a pre-configured location
    will be used if available.
    """

    debug_args_path: Optional[str] = None
    "Path to debugs args file."

    retrace_every_iteration: Optional[bool] = None

    eval_steps: Optional[int] = None
    "Specifies the number of steps to run the model evaluation."

    init_method: str = "env://"

    job_time_sec: Optional[int] = None

    job_labels: Optional[List[str]] = None
    "A list of equal-sign-separated key value pairs served as job labels."

    job_priority: Optional[str] = None
    "Priority of the job in scheduling queue."

    seed: Optional[int] = None
    "The seed to use for random number generation for reproducibility."

    mgmt_namespace: Optional[str] = None

    load_checkpoint_states: Optional[str] = None
    """
    Comma-separated string of keys used in conjunction with `checkpoint_path` to
    explicitly specify what components' state should be loaded if present in a checkpoint.
    If this flag is used, any component whose key isn't specified will not load state from
    the checkpoint. For example, if `load_checkpoint_states` is `model`, we only load the
    model state and enforce resetting of optimizer states and training steps after loading
    a given checkpoint; i.e., matching weights are initialized from checkpoint provided by
    `checkpoint_path`, training starts from step 0, and optimizer states present in the
    checkpoint are ignored." This is useful for fine-tuning runs on different tasks
    (e.g., classification, Q&A, etc.)
    where weights from a pre-trained model trained on language modeling (LM) tasks are
    loaded or fine-tuning on a
    different dataset on the same LM task. If `dataloader` state exists in the checkpoint
    that will also be ignored. In this case, the dataloaders will yield samples from
    the beginning. However, if `load_checkpoint_states` is `model,dataloader`then only
    the model and dataloader states will be loaded. By default, this config is `None`
    meaning that we load state for every compononent found in the checkpoint.
    """

    target_device: Literal["CPU", "GPU", "CSX"] = "CSX"
    """
    The target device to run the training on. One of: `CPU`, `GPU`, `CSX`.
    Required in command line.
    """

    mode: Literal[
        "train",
        "eval",
        "eval_all",
        "train_and_eval",
        "inference",
    ] = "train"
    """
    The mode of the training job, either '`train`', '`eval`', `eval_all` or
    `train_and_eval`.
    """

    wsc_log_level: Optional[
        Union[Literal["INFO", "DEBUG", "VERBOSE", "20", "10"], dict]
    ] = "INFO"
    """
    Specifes the logging level for particular Wafer-Scale Cluster servers or tasks.
    Input can be either a single value setting a global log level
    (i.e. `--wsc_log_level DEBUG`) or a list of
    equal-sign-separated key value pairs in the format of `<task or server>=<log level>`.
    A task and server can be combined to specify a server only during a specific task
    (i.e. `<execute>.<crd>`). The log level can be either an int or a string
    (i.e. `INFO`, `DEBUG`, `VERBOSE`, `20`, `10`).
    See [more](https://docs.python.org/3/library/logging.html#logging-levels).
    """

    autoload_last_checkpoint: Optional[bool] = True
    "Flag to automatically load the last checkpoint in the `model_dir`."

    check_loss_values: bool = True
    """
    Flag to check the loss values to see if it is `Nan/inf`.
    Defaults to True
    """

    disable_strict_checkpoint_loading: Optional[bool] = False
    """
    Flag used in conjunction with `checkpoint_path`, to avoid enforcing strict model
    state loading. Defaults to False
    """

    dist_addr: str = "localhost:8888"
    """
    To init master_addr and master_port of distributed.
    Defaults to 'localhost:8888'
    """

    dist_backend: str = "nccl"
    "Distributed backend engine. Defaults to 'nccl'"

    checkpoint_steps: Optional[int] = None
    """
    The number of steps between saving model checkpoints during training.
    `0` means no checkpoints saved. Defaults to 0
    """

    disable_version_check: Optional[bool] = False

    drop_data: Optional[bool] = False

    enable_distributed: bool = False
    "Flag to enable distributed training on GPU. Defaults to False"

    model_dir: str = "./model_dir"
    """
    The directory where the model checkpoints and other metadata will
    be saved during training. Defaults to './model_dir'
    """

    save_initial_checkpoint: bool = False
    """
    Whether to save an initial checkpoint before training starts.
    Defaults to False
    """

    precision_opt_level: Optional[int] = None
    """
    Setting to control the level of numerical precision used for training
    runs for large NLP modelzoo.
    See [more]
    (https://docs.cerebras.net/en/latest/general/performance-optimization.html
    ?#precision-optimization-level)
    Defaults to 1
    """

    num_workers_per_csx: int = 0
    """
    Number of input workers, per CSX, to use for streaming samples.
    This setting depends on whether the model is compute-bound or input-bound and how
    efficient the dataloader implementation is. For compute-bound modelzoo.(e.g., LLM),
    even 1 input worker per csx is enough to saturate the input buffers on CSX systems.
    But for smaller modelzoo.a larger number may be used. We currently default to 1 worker
    per CSX.
    defaults to 0
    """
    validate_only: Optional[bool] = False
    """
    Enables validate only workflow, stops the compilation at kernel matching stage.
    Defaults to False
    """

    logging: Optional[str] = "INFO"
    """
    Logging Specifies the logging level during training.
    Defaults to 'INFO'
    """

    sync_batchnorm: bool = False
    """
    Whether to use synchronized batch normalization on multi GPU setup.
    Defaults to False
    """

    compile_only: Optional[bool] = False
    """
    Enables compile only workflow.
    Defaults to False
    """

    log_steps: int = 1
    """
    Specifies the number of steps between logging during training.
    Same number controls the summary steps in Tensorboard.
    """

    num_steps: Optional[int] = None
    "The number of steps to train for."

    transfer_processes: Optional[int] = None
    "Number of transfer processes used for weight transfer"

    num_wgt_servers: Optional[int] = None
    """
    Upper bound on the number of MemoryX servers used for storing the model weights.
    Compilation may choose a smaller number depending on the model topology.
    A sensible upper bound (currently 24) is selected if a value is not provided.
    """

    num_csx: int = config_field(
        default=1,
        constraint=PositiveInteger,
    )
    "The number of CSX systems to use in Cerebras WSE cluster. Defaults to 1"

    num_act_servers: Optional[int] = config_field(
        default=1,
        constraint=PositiveInteger,
    )
    """
    Number of activation servers per CS-X dedicated to stream samples to the WSE.
    Input workers stream data to these activation servers, and the activation servers
    to hold and further stream the data to the WSE.
    For LLMs, we generally choose 1 because they're compute-bound.
    For CV modelzoo.we choose a higher number, a crude rule of thumb is to have one
    activation server for every 4 workers (i.e. `num_workers_per_csx // 4
    if num_workers_per_csx > 4, else 1`). It is suggested to keep the
    default values for this param when possible.
    defaults to 1
    """

    eval_frequency: Optional[int] = None
    "Specifies the evaluation frequency during training. Only used for `train_and_eval`mode"

    execute_crd_memory_gi: Optional[int] = None
    "Optional parameter to specifu the memory used for execution. Default : None"

    compile_crd_memory_gi: Optional[int] = None
    "Optional parameter to specifu the memory used for compile. Default : None"

    op_profiler_config: Optional[PytorchProfilerConfig] = None
    dump_activations: bool = False
    enable_distributed: bool = False
    log_input_summaries: bool = False
    main_process_id: int = 0
    max_checkpoints: Optional[int] = None
    summary_dir: Optional[str] = None
    lazy_initialization: bool = True
    use_cstorch_optimizer_step: bool = False
    wrk_memory_gi: Optional[int] = None
    act_memory_gi: Optional[int] = None
    cmd_memory_gi: Optional[int] = None
    wgt_memory_gi: Optional[int] = None
    experimental: dict = field(default_factory=dict)

    ini: Optional[Dict[str, Union[bool, int, float, str]]] = None
    "Internal debug flags for Wafer Scale Cluster compiler and runtime."

    debug_args: Optional[Dict[str, Union[bool, int, float, str]]] = None
    "Internal debug flags for Wafer Scale Cluster compiler and runtime."
