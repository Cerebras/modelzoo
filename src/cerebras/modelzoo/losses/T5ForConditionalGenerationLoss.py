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

# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


## Add label smoothing to loss function, this is a workaround method of label smoothing in our system
def smooth_loss(prediction_scores, loss, label_smoothing, classes):
    logits = prediction_scores.view(-1, classes)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    smooth_loss = -1.0 * logprobs.mean(dim=-1)
    loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth_loss

    return loss


class T5ForConditionalGenerationLoss(nn.Module):
    def __init__(self, lm_loss_weight, mlm_loss_scaling, label_smoothing=0.0):
        super(T5ForConditionalGenerationLoss, self).__init__()
        self.global_loss_weight = lm_loss_weight
        self.label_smoothing = label_smoothing
        self.mlm_loss_scaling = mlm_loss_scaling

    def forward(
        self, lm_logits, labels, decoder_attention_mask, loss_weight=None
    ):
        """
        Per-token loss is averaged across the batch by
            1) Summing across all tokens in the batch
            2) Dividing by the batch size
            3) Multiplying by the provided loss weight (expected to be roughly
                equal to `batch_size / num_tokens_in_batch`)
        The user has the option to specify this loss weight once and use the
        same weight for every batch (by setting `self.global_loss_weight` and not
        passing in `loss_weight` to the forward function) or use a different
        weight for every batch (by passing `loss_weight` to the forward function).
        """

        decoder_attention_mask = decoder_attention_mask.to(
            dtype=lm_logits.dtype
        )
        if self.mlm_loss_scaling == "precomputed_num_masked":
            if loss_weight is not None:
                loss_weight = loss_weight.to(dtype=lm_logits.dtype)
                decoder_attention_mask *= loss_weight
            else:
                self.mlm_loss_scaling = "batch_size"
                self.global_loss_weight = self.global_loss_weight or 1
                self.global_loss_weight = self.global_loss_weight
                print(
                    "mlm_loss_scaling is precomputed_num_masked and loss_weight is not provide, "
                    f"changing to mlm_loss_scaling to batch_size and set global_loss_weight to {self.global_loss_weight}. "
                    "Please ignore the warning if is eval mode."
                )
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1).long()
        )
        if self.label_smoothing > 0.0 and self.training:
            # Calculate loss correction for label smoothing
            loss = smooth_loss(
                lm_logits, loss, self.label_smoothing, lm_logits.size(-1)
            )
        loss *= decoder_attention_mask.view(-1)
        batch_size = labels.shape[0]
        if self.mlm_loss_scaling == "num_masked":
            loss = torch.sum(loss) / torch.sum(decoder_attention_mask)
        elif self.mlm_loss_scaling == "precomputed_num_masked":
            loss = torch.sum(loss) / batch_size
        elif self.mlm_loss_scaling == "batch_size":
            loss = torch.sum(loss) / batch_size * self.global_loss_weight
        else:
            raise ValueError(
                f"{self.mlm_loss_scaling} is not supported. Choose between `num_masked`, `precomputed_num_masked`, `batch_size`."
            )

        return loss
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
