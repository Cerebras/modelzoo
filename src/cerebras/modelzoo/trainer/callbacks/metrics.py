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

"""This module contains the callback class that logs all metrics attached to the model."""

from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.pytorch.metrics.metric import Metric


class ModelEvalMetrics(Callback):
    """Callback class that logs all metrics attached to the model."""

    def on_validate_end(self, trainer, model, loop):
        if trainer.backend.is_e2e_execution:
            # Print all metrics that are attached to the model
            metrics = {}

            for metric in model.modules():
                if isinstance(metric, Metric):
                    if metric.num_updates == 0:
                        trainer.logger.warning(
                            f"Skipping logging unused metric `{metric.name}` "
                            f"To remove this warning, either remove it from "
                            f"the model or step the metric every step."
                        )
                    else:
                        metrics[metric.name] = float(metric)
                        metric.reset()

            trainer.logger.info("Evaluation metrics:")
            for name, metric in metrics.items():
                trainer.logger.info(f"  - {name} = {float(metric)}")
            trainer.log_metrics(**metrics)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
