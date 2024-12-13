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
This file contains the GradientAccumulationCallback class which is used to accumulate gradients.
"""

from cerebras.modelzoo.trainer.callbacks import CoreCallback


class GradientAccumulationCallback(CoreCallback):
    """
    Callback class to accumulate gradients.
    """

    def __init__(self):
        """
        Attributes:
            grad_accum_steps: The number of steps to accumulate gradients for
                before stepping the optimizer.
            should_run_optimizer_step: If True, run the optimizer step in the current step.

        """
        self.grad_accum_steps = None
        self.should_run_optimizer_step = True

    def setup(self, trainer):
        # Get the number of gradient accumulation steps from the trainer's loop
        # callback
        self.grad_accum_steps = getattr(trainer.loop, "grad_accum_steps", 1)
        if trainer.backend.is_csx:
            if self.grad_accum_steps != 1:
                trainer.logger.info(
                    "`grad_accum_steps` param has no effect when running on the CSX. "
                    "Consider setting `micro_batch_size` to \"auto\" or \"disable\" to enable or "
                    "disable micro batch tiling on CSX."
                )
            self.grad_accum_steps = 1
        else:
            trainer.logger.info(
                f"Gradient accumulation steps is {self.grad_accum_steps}"
            )

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if self.grad_accum_steps > 1:
            self.should_run_optimizer_step = (
                batch_idx + 1
            ) % self.grad_accum_steps == 0

    def on_after_forward(self, trainer, model, outputs, batch):
        if self.grad_accum_steps > 1 and "loss" in outputs:
            # Purposefully avoid inplace operation on loss
            # as it complicates the backward pass unnecessarily
            outputs["loss"] = outputs["loss"] / self.grad_accum_steps
