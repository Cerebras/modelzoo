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

from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
from pydantic import Field
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.models.multimodal.multimodal_base_model import (
    ModalityType,
)
from cerebras.modelzoo.models.multimodal.multimodal_simple.modeling_mmsimple import (
    MultimodalImageModelList,
    MultimodalSimple,
)
from cerebras.modelzoo.models.nlp.llama.model import LlamaLMHeadModelConfig
from cerebras.modelzoo.models.nlp.t5.model import (
    T5ForConditionalGenerationModelConfig,
)

MultimodalTextModel = Annotated[
    Union[
        LlamaLMHeadModelConfig,
        T5ForConditionalGenerationModelConfig,
    ],
    Field(discriminator="name"),
]


class MultimodalDecoderModelConfig(ModelConfig):
    name: Literal["multimodal_simple"]
    "The name of the model. Must be set to `multimodal_simple`."

    freeze: Optional[List[str]] = None
    """
    List of regex patterns that match layers of the model to be frozen.
    Frozen layers will not have their weights updated during training.
    Note that regex patterns should be specified as single quotes in the yaml for escape codes.
    Perl style regex expected and are parsed by `re` python package.
    Ex: freeze: ['^image_model.image_model_list', '^text_model']
    freezes all parameters whose names start with 
    `image_model.image_model_list` and `text_model`.
    """

    image_model_list: MultimodalImageModelList = ...
    "Allows the model to instantiate a list of image models that run same image through multiple image encoders."

    text_model: MultimodalTextModel = ...
    "Decoder-only LLM model that processes all the modalities together through the backbone and produces output."

    output_list: Optional[
        List[Literal["image", "image_encoder_out", "projector_out"]]
    ] = None
    "List of intermediate values that should be returned. Options include: image, image_encoder_out, projector_out. Model always returns output of VLLMModel."

    loss_scaling: Literal["batch_size", "num_tokens"] = "num_tokens"
    """The scaling type used to calculate the loss. Accepts - `batch_size`, `num_tokens`.
    See [more](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html).
    **Note:** It is recommended to set this to `num_tokens` for convenience."""

    loss_weight: float = 0.0
    """The weight for the loss scaling when `loss_scaling = 'batch_size'`, generally set to
    '1/max_sequence_length`.
    """


class MMSimpleModel(nn.Module):
    """Multimodal Models that combine image and text."""

    def __init__(self, config: MultimodalDecoderModelConfig):
        if isinstance(config, dict):
            config = MultimodalDecoderModelConfig(**config)

        super().__init__()

        self.model = MultimodalSimple.build_model(config)
        text_model = getattr(config, ModalityType.TEXT.value)
        self.loss_fn = GPTLMHeadModelLoss(
            text_model.vocab_size,
            config.loss_scaling,
            config.loss_weight,
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
