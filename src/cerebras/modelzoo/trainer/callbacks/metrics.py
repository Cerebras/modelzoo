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

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback


class ModelEvalMetrics(Callback):
    """Callback class that logs all metrics attached to the model."""

    def on_validate_end(self, trainer, model, loop):
        if trainer.backend.is_e2e_execution:
            # Print all metrics that are attached to the model
            metrics = {}

            for metric in model.modules():
                if isinstance(metric, cstorch.metrics.Metric):
                    metrics[metric.name] = float(metric)
                    metric.reset()

            trainer.log_metrics(**metrics)
