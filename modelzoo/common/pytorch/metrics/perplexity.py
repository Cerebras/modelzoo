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
Perplexity metric for PyTorch.
"""
import math

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch
from modelzoo.common.pytorch.metrics.cb_metric import CBMetric, DeviceOutputs


def PerplexityMetric(*args, **kwargs):
    """Calculates LM perplexity, which is the exp(loss per predicted token)."""
    if cm.use_cs() and cbtorch.env().weight_streaming_mode:
        return WSPerplexityMetric(*args, **kwargs)
    else:
        return PipelinePerplexityMetric(*args, **kwargs)


class PipelinePerplexityMetric(CBMetric):
    def init_state(self):
        self.reset()

    def update_on_device(self, labels, loss, weights=None, dtype=None):
        return DeviceOutputs(args=[labels, loss, weights])

    def update_on_host(self, labels, loss, weights=None):
        """Host calculation of LM perplexity.

        Args:
            labels: Tensor of shape (batch, sequence) and type int32.
            loss: Tensor of shape (1) and type float.
            weights: Optional float Tensor of shape (batch, sequence).
        """
        if weights is None:
            num_tokens = float(labels.numel())
        else:
            num_tokens = float(weights.detach().sum())

        self.total_loss += loss
        self.total_num_tokens += num_tokens

    def compute(self):
        """Returns the computed metric value over all updates."""
        try:
            perplexity = math.exp(self.total_loss / self.total_num_tokens)
        except OverflowError:
            perplexity = math.inf
        return perplexity

    def reset_state(self):
        self.total_loss = 0.0
        self.total_num_tokens = 0.0


class WSPerplexityMetric(CBMetric):
    def init_state(self):
        self.reset_state()

    def update_on_device(self, labels, loss, weights=None, dtype=None):
        if weights is None:
            num_tokens = torch.tensor(
                labels.numel(), dtype=torch.float32, device=labels.device
            )
        else:
            num_tokens = (weights > 0).float().sum()

        self.total_loss.add_(loss)
        self.total_num_tokens.add_(num_tokens)

        result = torch.exp(self.total_loss / self.total_num_tokens)
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
        self.total_loss = torch.tensor(
            0, dtype=torch.float32, device=cm.device()
        )
        self.total_num_tokens = torch.tensor(
            0, dtype=torch.float32, device=cm.device()
        )

    def on_device_state_dict(self):
        return {
            "total_loss": self.total_loss,
            "total_num_tokens": self.total_num_tokens,
        }
