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

"""Contains the SparsityCallback class that applies sparsity to the model."""

from typing import Optional

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback, CoreCallback


class SparsityCallback(CoreCallback):
    """Callback class that applies sparsity to the model and optimizer."""

    def __init__(
        self, sparsity: Optional[cstorch.sparse.SparsityAlgorithm] = None
    ):
        """
        Args:
            sparsity: Sparsity algorithm instance.
        """
        self.sparsity = sparsity

    def setup(self, trainer):
        if self.sparsity is None:
            return
        elif isinstance(self.sparsity, cstorch.sparse.SparsityAlgorithm):
            pass
        else:
            self.sparsity = self.sparsity()
            if self.sparsity is None:
                return

        trainer.model.apply(self.sparsity)

        if trainer.optimizer is not None:
            trainer.optimizer.apply(self.sparsity)

    def on_save_checkpoint(self, trainer, state_dict):
        if self.sparsity:
            state_dict["sparsity"] = self.sparsity.state_dict()

    def on_load_checkpoint(self, trainer, state_dict):
        if self.sparsity:
            if (
                "model" in state_dict
                and not trainer.checkpoint.disable_strict_checkpoint_loading
                and not any(
                    k.endswith("_mask")
                    and k[: -len("_mask")] in state_dict["model"]
                    for k in state_dict["model"].keys()
                )
            ):
                raise RuntimeError(
                    "Did not find any sparsity masks in the model checkpoint."
                    " Please ensure that you're using a checkpoint with the"
                    " correct sparsity config from a previous run."
                )

            if "sparsity" in state_dict:
                self.sparsity.load_state_dict(state_dict["sparsity"])

                trainer.logger.info(
                    f"Sparsity state found in checkpoint and loaded successfully."
                )
            else:
                trainer.logger.info(
                    "Sparsity state not found in the checkpoint. "
                    "Using default initialized state."
                )


class LogSparsity(Callback):
    """Log target and actual sparsity levels."""

    def setup(self, trainer):
        if trainer.optimizer is not None:
            sparsity = trainer.callbacks["sparsity"].sparsity
            if sparsity is None:
                return

            sparsity.register_target_sparsity_hook(
                lambda _, name, target: trainer.log_metrics(
                    **{f"sparsity/{name}/target": target}
                )
            )
            sparsity.register_computed_sparsity_hook(
                lambda _, name, actual: trainer.log_metrics(
                    **{f"sparsity/{name}/actual": actual}
                )
            )

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
