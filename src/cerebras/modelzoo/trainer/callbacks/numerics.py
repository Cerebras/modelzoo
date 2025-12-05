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
Provides a callback which setups up and utilizes the DumpContext, a debug
utility for dumping activations and gradients on a CPU/GPU run, and setting up
debug names for dumped WSE activations to be automatically correlated.
"""

import functools
import os
import warnings
from collections import defaultdict
from contextlib import ContextDecorator, nullcontext
from pathlib import Path
from typing import Optional
from warnings import warn

import numpy as np
import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback, MixedPrecision
from cerebras.pytorch.utils.nest import visit_torch_tensors


class DumpActivations(Callback):
    """Callback to dump activations for CPU/GPU runs."""

    def __init__(
        self, outdir: Optional[str] = None, buffer_steps: Optional[int] = None
    ):
        """
        Args:
            outdir: The output directory at which to dump the activations
            buffer_steps: If given, flush to a new .npz file after this many steps.
        """
        self._outdir = outdir
        self._buffer_steps = buffer_steps
        self._dump_ctx = None

    def setup(self, trainer):
        if trainer.backend.is_csx:
            warn(
                "Activation dumping is not supported on CSX."
                "Disabling activation dumping for this run."
            )
            self._dump_ctx = nullcontext()
            self._dump_ctx.flush = lambda: None
        else:
            outdir = Path(
                self._outdir
                if self._outdir
                else trainer.model_dir / "act_dumps"
            )
            self._dump_ctx = DumpContext(
                outdir,
                trainer.model,
                self._buffer_steps,
            )

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self._dump_ctx.__enter__()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self._dump_ctx.__exit__()

    def on_train_end(self, trainer, model, loop, loop_idx):
        self._dump_ctx.flush()

    def on_validate_batch_start(self, trainer, model, batch, batch_idx):
        self._dump_ctx.__enter__()

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self._dump_ctx.__exit__()

    def on_validate_end(self, trainer, model, loop):
        self._dump_ctx.flush()

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class DumpContext(ContextDecorator):
    """
    A debug utility context manager. When provided with a torch.nn.Module, the
    resulting context manager can be entered to enable dumping of all module
    forward and backward outputs to a npz, for comparing numerics between
    implementations.
    """

    def __init__(
        self, outdir: str, model: torch.nn.Module, buffer_steps: int = None
    ):
        """
        Sets up global module hoooks to either dump intermediate activations on
        CPU/GPU or name the traced tensors for correlating with debug dumps on
        CS2.

        The recursive name of the torch.nn.Module is memoized, and the output
        of FWD and BWD of each module is saved as keys in a .npz file.

        Args:
            outdir: Where to output dumps_{i}.npz
            model: root module to name its children
            buffer_steps: If given, flush to a new .npz file after this many
             steps
        """
        self._outdir = outdir

        os.makedirs(self._outdir, exist_ok=True)

        # The actual hook functions to install
        self._forward_pre_hook = None
        self._forward_hook = None
        self._backward_hook = None
        self._full_backward_hook = None

        self.setup_hooks(model)

        # Any installed hooks, set during enable_collection()
        self._module_hooks = []
        self._call_counter = {}

        self._buffer_steps = buffer_steps
        self._flush_count = 0
        self._buffer = defaultdict(list)

    def __enter__(self):
        self.enable_collection()
        return self

    def __del__(self):
        self.flush()

    def __exit__(self, *exc):
        self.disable_collection()

        # Check if we need to flush by using the first buffer's size as a
        # proxy for how many steps we've captured.
        if self._buffer_steps and self._buffer:
            first_buffer = next(iter(self._buffer))
            if len(first_buffer) >= self._buffer_steps:
                self.flush()

    def setup_hooks(self, model):
        """
        Define hooking functions on the given torch.nn.Module, but don't
        install them.

        Args:
            model: torch.nn.Module that serves as the root for recursive names
        """
        if cstorch.use_cs():
            # Not enabled for CSX, dumping only works on CPU/GPU
            return
        cstorch.add_debug_name(model)

        # Helpers for hooks
        def get_name(module, counter_increment=0):
            name = cstorch.get_debug_name(module)

            def_counter = 0 if counter_increment >= 0 else 1
            counter = self._call_counter.setdefault(name, def_counter)
            self._call_counter[name] += counter_increment
            if counter != def_counter:
                name = f"{name}.call{counter}"

            return name

        def save_tensors(top_scope, tensors):
            for scope, tensor in visit_torch_tensors(tensors, scope=top_scope):
                tensor = tensor.detach().to("cpu").clone()
                if tensor.dtype == torch.bfloat16:
                    warnings.warn(
                        "Encountered bfloat16 tensor in summary collection. "
                        "Numpy does not natively support bfloat16, so any "
                        "torch.bfloat16 tensors will be saved as np.float32."
                    )
                    tensor = tensor.float()

                name = ".".join(scope)
                numpy = tensor.numpy()

                self._buffer[name].append(numpy)

        # pylint: disable=redefined-builtin
        def save_output(key, module, input, output):
            """
            Saves to numpy arrays in the output directory.
            """

            counter_increment = 1
            if key == "bwd":
                counter_increment = -1
                # hook args are `grad_input, grad_output`, where grad_input
                # is the _gradient_ of the module's input i.e. the output
                # of the backward pass and the more interesting value to
                # dump. This way, the dump named `module.fwd` is the output
                # of the forward pass (i.e. txact), and `module.bwd` is the
                # output of the backward pass (i.e. txdelta) for the
                # corresponding kernel
                output = input

            name = get_name(module, counter_increment)
            save_tensors([name, key], output)

        self._forward_hook = functools.partial(save_output, "fwd")
        self._full_backward_hook = functools.partial(save_output, "bwd")

        # Add hook capturing parameter gradients
        def param_grad_hook(module, input):
            module_name = get_name(module)
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and not hasattr(param, "dump_context"):
                    param.dump_context = True
                    scope = [module_name, "bwd", name]
                    param.register_hook(functools.partial(save_tensors, scope))

        self._forward_pre_hook = param_grad_hook

    def enable_collection(self):
        """
        Install the hooks defined during `setup_hooks`, enabling the
        collection of the dumps.
        """

        def install_if_set(hook):
            hook_fn = getattr(self, f"_{hook}_hook")
            if hook_fn:
                register_fn = f"register_module_{hook}_hook"
                return getattr(torch.nn.modules.module, register_fn)(hook_fn)
            return None

        hooks = ("forward_pre", "forward", "backward", "full_backward")
        self._module_hooks = [install_if_set(hook) for hook in hooks]
        # Clear call counters
        self._call_counter = {}

    def disable_collection(self):
        """
        Uninstall the hooks installed during `enable_collection`, disabling
        further dump collection.
        """
        for hook in self._module_hooks:
            if hook:
                hook.remove()
        self._module_hooks = []

    def flush(self):
        """
        Write all dump buffers out to disk.
        """
        if not self._buffer:
            return

        if self._flush_count:
            outfile = f"act_dumps_{self._flush_count}.npz"
        else:
            outfile = "act_dumps.npz"
        np.savez(
            os.path.join(self._outdir, outfile),
            **{key: np.stack(values) for key, values in self._buffer.items()},
        )
        self._buffer.clear()
        self._flush_count += 1


class NaNChecker(Callback):
    """Callback to enable the Cerebras NaN Checker for a given run.

    The NaN Checker is a debugging tool specific to the CSX hardware that injects breakpoints in
    the kernels where NaN values are encountered. This tool should only be enabled to determine
    the source of NaN values, after these have already been detected for a given run, i.e. if
    the run raises an `ApplianceNanError` exception. Thus, this callback should purely be used as
    as debugging utility; enabling the NaN Checker in general will result in performance degradation
    on CSX.

    Note that this tool is different from the PyTorch operation `torch.isnan` that simply checks
    for NaN values in a tensor.
    """

    def __init__(self, num_steps: Optional[int] = None):
        """Callback to enable the Cerebras NaN Checker for a given run.

        Args:
            num_steps: The number of steps for the NaN Checker-enabled run. If specified,
                the run will early exit after this many steps.
        """
        self.num_steps = num_steps

        self.dls_enabled = False
        self.original_inis = {}
        self._nan_checker_inis = [
            "ws_rt_run_nancheck",
            "ws_rt_enable_stall_check",
        ]
        self._missing = object()

    def setup(self, trainer):
        self.dls_enabled = (
            trainer.precision is not None
            and isinstance(trainer.precision, MixedPrecision)
            and trainer.precision.loss_scaling_factor == "dynamic"
        )
        if self.dls_enabled:
            self._nan_checker_inis.append("ws_rt_skip_bwd_pass_nancheck")

        if self.num_steps is not None:

            def early_exit_fn():
                return (
                    trainer.schedule.batch_idx is not None
                    and trainer.schedule.batch_idx + 1 >= self.num_steps
                )

            trainer.early_exit.register(early_exit_fn)

    def _set_nan_checker_flags(self):
        """Set the debug flags for NaN checker."""
        for ini in self._nan_checker_inis:
            self.original_inis[ini] = getattr(
                cstorch.backends.csx.debug.ini, ini, self._missing
            )

            setattr(cstorch.backends.csx.debug.ini, ini, True)

    def _unset_nan_checker_flags(self):
        """Unset the debug flags for NaN checker."""
        for ini, value in self.original_inis.items():
            if value is self._missing:
                delattr(cstorch.backends.csx.debug.ini, ini)
            else:
                setattr(cstorch.backends.csx.debug.ini, ini, value)

        self.original_inis.clear()

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        self._set_nan_checker_flags()

    def on_train_end(self, trainer, model, loop, loop_idx):
        self._unset_nan_checker_flags()

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        self._set_nan_checker_flags()

    def on_validate_end(self, trainer, model, loop):
        self._unset_nan_checker_flags()

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
