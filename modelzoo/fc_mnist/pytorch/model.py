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

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebras_pytorch.metrics import AccuracyMetric
from modelzoo.common.pytorch.run_utils import half_dtype_instance


class MNIST(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.fc_layers = []
        use_bias = model_params.get("use_bias", True)
        input_size = 784
        # Set the default or None
        if "hidden_sizes" in model_params:
            # Depth is len(hidden_sizes)
            model_params["depth"] = len(model_params["hidden_sizes"])
        else:
            # same hidden size across dense layers
            model_params["hidden_sizes"] = [
                model_params["hidden_size"]
            ] * model_params["depth"]

        for hidden_size in model_params["hidden_sizes"]:
            fc_layer = nn.Linear(input_size, hidden_size, bias=use_bias)
            self.fc_layers.append(fc_layer)
            input_size = hidden_size
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.last_layer = nn.Linear(input_size, 10, bias=use_bias)

        self.nonlin = self._get_nonlinear(model_params)

        self.dropout = nn.Dropout(model_params["dropout"])

    def forward(self, inputs):
        x = torch.flatten(inputs, 1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            if self.nonlin:
                x = self.nonlin(x)
            x = self.dropout(x)

        pred_logits = self.last_layer(x)
        return pred_logits

    def _get_nonlinear(self, model_params):
        if model_params["activation_fn"] == "relu":
            return nn.ReLU()
        elif model_params["activation_fn"] is None:
            return None
        else:
            raise ValueError("supports activation_fn: 'relu' or null")


class MNISTModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Disable eval metrics in non-eval modes
        if params["runconfig"]["mode"] != "eval" and params["model"].get(
            "compute_eval_metrics", []
        ):
            params["model"]["compute_eval_metrics"] = []

        model_params = deepcopy(params["model"])
        self.model = self.build_model(model_params)
        self.loss_fn = nn.NLLLoss()
        self.disable_softmax = params["model"].get("disable_softmax", False)

        compute_eval_metrics = model_params.get("compute_eval_metrics", [])
        if isinstance(compute_eval_metrics, bool) and compute_eval_metrics:
            compute_eval_metrics = ["accuracy"]  # All metrics

        self.accuracy_metric = None
        for name in compute_eval_metrics:
            if "accuracy" in name:
                self.accuracy_metric = AccuracyMetric(name=name)
            else:
                raise ValueError(f"Unknown metric: {name}")

    def build_model(self, model_params):
        dtype = (
            half_dtype_instance.half_dtype
            if model_params["to_float16"]
            else torch.float32
        )
        model = MNIST(model_params)
        model.to(dtype)
        return model

    def forward(self, data):
        inputs, labels = data
        pred_logits = self.model(inputs)
        if self.accuracy_metric:
            labels = labels.clone()
            predictions = pred_logits.argmax(-1).int()
            self.accuracy_metric(labels=labels, predictions=predictions)
        if not self.disable_softmax:
            pred_logits = F.log_softmax(pred_logits, dim=1)
        loss = self.loss_fn(pred_logits, labels)
        return loss
