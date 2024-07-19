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
This module contains the base Callback class as well as a number of core
callbacks directly invoked by the Trainer as well as other optional callbacks
that can be used to extend the functionality of the Trainer.
"""

# isort: off
from .callback import (
    Callback,
    GLOBAL_CALLBACK_REGISTRY,
    register_global_callback,
    ValidationCallback,
)

# Core Callbacks
from .artifact_dir import ArtifactDirCallback
from .backend import BackendCallback
from .checkpoint import (
    Checkpoint,
    LoadCheckpointStates,
    KeepNCheckpoints,
    SaveCheckpointState,
)
from .dataloader import DataLoaderCallback, LogInputSummaries
from .grad_accum import GradientAccumulationCallback
from .loop import LoopCallback, TrainingLoop, ValidationLoop
from .logging import Logging
from .model import ModelCallback
from .optimizer import LogOptimizerParamGroup, OptimizerCallback
from .precision import MixedPrecision, Precision
from .reproducibility import Reproducibility
from .schedulers import SchedulersCallback, SchedulersInput
from .sparsity import SparsityCallback, LogSparsity

# Optional Callbacks
from .compression import WeightCompression
from .flags import (
    GlobalFlags,
    ScopedTrainFlags,
    ScopedValidateFlags,
    DebugArgsPath,
)
from .listener import Listener
from .loss import CheckLoss
from .metadata import ModelZooParamsMetadata
from .metrics import ModelEvalMetrics
from .norm import ComputeNorm
from .numerics import DumpActivations
from .profiler import (
    Profiler,
    RateProfiler,
    OpProfiler,
    SavePerformanceData,
    FlopUtilization,
)
from .selective_grad import SelectiveGrad

# isort: on

# __all__ is required for docs to autogenerate correctly
__all__ = [
    "Callback",
    "register_global_callback",
    "ValidationCallback",
    # Core Callbacks
    "ArtifactDirCallback",
    "BackendCallback",
    "Checkpoint",
    "LoadCheckpointStates",
    "KeepNCheckpoints",
    "SaveCheckpointState",
    "DataLoaderCallback",
    "GradientAccumulationCallback",
    "LoopCallback",
    "TrainingLoop",
    "ValidationLoop",
    "Logging",
    "ModelCallback",
    "OptimizerCallback",
    "Precision",
    "MixedPrecision",
    "Reproducibility",
    "SchedulersCallback",
    "SparsityCallback",
    # Add-on Callbacks
    "LogInputSummaries",
    "LogOptimizerParamGroup",
    "LogSparsity",
    "WeightCompression",
    "GlobalFlags",
    "ScopedTrainFlags",
    "ScopedValidateFlags",
    "DebugArgsPath",
    "Listener",
    "CheckLoss",
    "ModelZooParamsMetadata",
    "ModelEvalMetrics",
    "ComputeNorm",
    "DumpActivations",
    "Profiler",
    "RateProfiler",
    "OpProfiler",
    "SavePerformanceData",
    "FlopUtilization",
    "SelectiveGrad",
]
