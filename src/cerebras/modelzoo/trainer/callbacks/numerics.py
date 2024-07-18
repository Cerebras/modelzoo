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

from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from warnings import warn

from cerebras.modelzoo.common.dump_context import DumpContext
from cerebras.modelzoo.trainer.callbacks import Callback


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
