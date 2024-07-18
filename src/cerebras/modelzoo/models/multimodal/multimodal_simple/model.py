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

import torch
import torch.nn as nn

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.models.multimodal.multimodal_base_model import (
    ModalityType,
)
from cerebras.modelzoo.models.multimodal.multimodal_simple.modeling_mmsimple import (
    MultimodalSimple,
)


@registry.register_model("multimodal_simple")
class MMSimpleModel(nn.Module):
    """Multimodal Models that combine image and text"""

    def __init__(self, params):
        super().__init__()

        model_params = params.model
        self.model = MultimodalSimple.build_model(params)
        text_model = getattr(params.model, ModalityType.TEXT.value)
        self.loss_fn = GPTLMHeadModelLoss(
            text_model.vocab_size,
            model_params.loss_scaling,
            model_params.loss_weight,
        )

    def forward(
        self,
        data,
        reduce_batch=True,
        output_logits=False,
    ):
        """The forward pass on the input data. This method
        returns the loss tensor if `output_logits` is False.
        If `output_logits` is True, the model call will also
        return the output logits tensor in addition to the
        loss as a (loss, lm_logits) tuple.

        This may be useful for performing post processing on
        the model's output logits.
        """

        tgt_key_padding_mask = data.get("key_padding_mask", None)
        if tgt_key_padding_mask is not None:
            dtype = (
                data["image_data"].dtype
                if data["image_data"] is not None
                else data["image_embeddings"].dtype
            )
            tgt_key_padding_mask = tgt_key_padding_mask.to(dtype)
            tgt_key_padding_mask = (
                tgt_key_padding_mask
                * torch.finfo(tgt_key_padding_mask.dtype).min
            )

        model_outputs = self.model(
            image_data=data.get("image_data", None),
            text_input_ids=data.get("text_input_ids", None),
            image_embeddings=data.get("image_embeddings", None),
            text_embeddings=data.get("text_embeddings", None),
            tgt_key_padding_mask=tgt_key_padding_mask,
            attention_span=data.get("attention_span", None),
            image_data_loc=data.get("image_data_loc", None),
            token_modality_idx=data.get("token_modality_idx", None),
            position_ids=data.get("position_ids", None),
        )

        if isinstance(model_outputs, tuple):
            logits, routing_weights, expert_mask = model_outputs
        else:
            logits = model_outputs

        loss = self.loss_fn(
            logits,
            labels=data["labels"],
            attention_mask=data["loss_mask"],  # acts as a loss mask
            reduce_batch=reduce_batch,
        )

        if output_logits:
            return loss, logits
        else:
            return loss
