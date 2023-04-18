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
Accuracy metric for PyTorch.
"""
import warnings

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.metrics.cb_metric import CBMetric, DeviceOutputs


class _PipelineAccuracyMetric(CBMetric):
    def init_state(self):
        self.reset_state()

    def update_on_device(self, labels, predictions, weights=None, dtype=None):
        return DeviceOutputs(args=[labels, predictions, weights])

    def update_on_host(self, labels, predictions, weights=None):
        labels = labels.detach().flatten()
        predictions = predictions.detach().flatten()
        correct_predictions = (labels == predictions).float()
        if weights is None:
            num_tokens = float(correct_predictions.numel())
        else:
            weights = weights.detach().flatten()
            correct_predictions = correct_predictions * weights
            num_tokens = float(weights.sum())

        self.total_correct_predictions += correct_predictions.sum()
        self.total_num_tokens += num_tokens

    def compute(self):
        """Returns the computed accuracy as a float."""
        return float(self.total_correct_predictions / self.total_num_tokens)

    def reset_state(self):
        self.total_correct_predictions = 0.0
        self.total_num_tokens = 0.0


class _WSAccuracyMetric(CBMetric):
    def init_state(self):
        self.reset_state()

    def update_on_device(self, labels, predictions, weights=None, dtype=None):
        if labels.shape != predictions.shape:
            warnings.warn(
                "Shapes mismatch in accuracy metric"
                f"\n    labels: {labels.shape}"
                f"\n    predictions {predictions.shape}"
            )
            predictions = predictions.reshape(labels.shape)

        correct_predictions = (labels == predictions).float()
        if weights is None:
            num_correct_predictions = correct_predictions.sum()
            num_tokens = torch.tensor(
                correct_predictions.numel(),
                dtype=torch.float32,
                device=predictions.device,
            )
        else:
            correct_predictions = correct_predictions * weights
            num_correct_predictions = correct_predictions.sum()
            num_tokens = (weights > 0).float().sum()
        self.total_correct_predictions.add_(num_correct_predictions)
        self.total_num_tokens.add_(num_tokens)
        result = self.total_correct_predictions / self.total_num_tokens
        # TODO(SW-82959): We need to remove the whole half dtype everywhere in metrics once we do for BERT model
        if dtype is None:
            from modelzoo.common.pytorch.run_utils import half_dtype_instance

            return DeviceOutputs(
                args=[result.to(half_dtype_instance.half_dtype)]
            )
        else:
            return DeviceOutputs(args=[result.to(dtype)])

    def update_on_host(self, result):
        self.result = result

    def compute(self):
        return float(self.result)

    def reset_state(self):
        self.total_correct_predictions = torch.tensor(
            0, dtype=torch.float32
        ).to(cm.device())
        self.total_num_tokens = torch.tensor(0, dtype=torch.float32).to(
            cm.device()
        )

    def on_device_state_dict(self):
        return {
            "total_correct_predictions": self.total_correct_predictions,
            "total_num_tokens": self.total_num_tokens,
        }


# Create a factory for creating a metric depending on execution strategy
AccuracyMetric = CBMetric.create_metric_impl_factory(
    pipeline_metric_cls=_PipelineAccuracyMetric,
    ws_metric_cls=_WSAccuracyMetric,
)
