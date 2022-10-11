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

from modelzoo.common.pytorch.metrics.cb_metric import CBMetric


class PerplexityMetric(CBMetric):
    """Calculates LM perplexity, which is the exp(loss per predicted token)."""

    def init_state(self):
        self.reset()

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
