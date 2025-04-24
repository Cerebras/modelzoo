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

import math
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers.FeedForwardNetwork import (
    FeedForwardNetworkConfig,
    MoEConfig,
)
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.losses.LoadBalancingLoss import LoadBalancingLoss
from cerebras.modelzoo.models.multimodal.llava.modeling_llava import Llava
from cerebras.modelzoo.models.multimodal.multimodal_base_model import (
    ModalityType,
    MultimodalBaseModelWrapper,
)
from cerebras.modelzoo.models.nlp.llama.model import LlamaLMHeadModelConfig
from cerebras.modelzoo.models.nlp.t5.t5_model import (
    T5ForConditionalGenerationModelConfig,
)
from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import (
    ViTModelConfig,
)
from cerebras.modelzoo.trainer import summarize_scalar

MultiModalModelType = Annotated[
    Union[
        ViTModelConfig,
        LlamaLMHeadModelConfig,
        T5ForConditionalGenerationModelConfig,
    ],
    Field(discriminator="name"),
]

ProjectorType = Annotated[
    Union[FeedForwardNetworkConfig,],
    Field(discriminator="name"),
]


class LlavaModelConfig(ModelConfig):
    name: Literal["llava"]
    "The name of the model. Must be set to `llava`."

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
    image_feature_select_layer_idx: Optional[int] = -1
    """
    Zero based index that indicates the decoder layer whose output, 
    should be extracted and used as image features. 
    For example: If the image_model consists of 
    20 decoder layers denoted as layer_0,layer_1... layer_19, 
    then setting `image_feature_select_layer_idx` = -1
    would extract the output of layer_19 in forward pass. 
    Similarly setting `image_feature_select_layer_idx` = 2, 
    would extract the output from layer_2 i.e 
    third layer among the 20 layers present in the decoder stack.
    """
    image_start_idx: int = 1
    "The position in sequence where the image tokens start."

    image_feature_select_mode: Literal["patch", "cls_patch"] = "patch"
    """
    If `patch`, only consider output at image patch tokens and ignore CLS token
    If `cls_patch`, consider both CLS token and patch features from image_model.
    """

    loss_scaling: Literal["num_tokens", "batch_size"] = "num_tokens"
    """The scaling type used to calculate the loss. Accepts - `batch_size`, `num_tokens`.
    See [more](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html).
    **Note:** It is recommended to set this to `num_tokens` for convenience."""

    loss_weight: float = 1.0
    """The weight for the loss scaling when `loss_scaling = 'batch_size'`, generally set to
    '1/max_sequence_length`.
    """
    image_model: MultiModalModelType = ...
    "The underlying image model being used."
    text_model: MultiModalModelType = ...
    "The underlying text model being used."
    projector: Optional[Dict[str, ProjectorType]] = None
    "The underlying projector module that connects image_model and text_model."
    moe_params: MoEConfig = Field(default_factory=MoEConfig, alias="moe")
    "A dict of MoE params including num_experts, top_k and load_balancing_loss_coef."

    @field_validator("projector")
    @classmethod
    def update_projector(cls, projector: Optional[Dict[str, ProjectorType]]):
        if projector is not None:
            return {
                key: (
                    value.copy(
                        update=dict(
                            # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
                            # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
                            # https://github.com/pytorch/pytorch/issues/57109
                            kernel_initializer={
                                "name": "kaiming_uniform",
                                "a": math.sqrt(5),
                            },
                            # Note: Using Kaiming_uniform directly on bias tensor
                            # results in PyTorch error:`ValueError: Fan in and fan out
                            # can not be computed for tensor with fewer than 2 dimensions`
                            # While this mismatches the src code, since we load from
                            # HF -> CS converted checkpoint, this is initialized in the
                            # checkpoint correctly
                            bias_initializer="zeros",
                        )
                    )
                    if isinstance(value, FeedForwardNetworkConfig)
                    else value
                )
                for key, value in projector.items()
            }

        return projector

    @model_validator(mode="after")
    def validate_projector(self):
        if (
            self.image_model.hidden_size != self.text_model.hidden_size
            and self.projector is None
        ):
            raise ValueError(
                f"The model should have a projector when the image model "
                f"and text model do not have the same `hidden_size`."
            )

        return self

    def post_init(self, context):
        # convert negative index and positive index representing layer_id of
        # encoder to positive index. All indices are zero-based.
        image_feature_select_layer_idx = (
            self.image_feature_select_layer_idx
            % self.image_model.num_hidden_layers
        )
        if (
            self.image_feature_select_layer_idx
            != image_feature_select_layer_idx
        ):
            self.image_feature_select_layer_idx = image_feature_select_layer_idx


class LlavaModel(MultimodalBaseModelWrapper, nn.Module):
    def __init__(self, config: LlavaModelConfig):
        super(LlavaModel, self).__init__()
        self._modalities = [ModalityType.IMAGE, ModalityType.TEXT]
        self._config = config
        self.freeze = config.freeze
        self.image_feature_select_layer_idx = (
            config.image_feature_select_layer_idx
        )
        self.image_start_idx = config.image_start_idx
        self.image_feature_select_mode = config.image_feature_select_mode

        text_config = config.text_model
        if not hasattr(text_config, 'moe_params'):
            self.moe_enabled = False
        else:
            total_experts = text_config.moe_params.num_experts
            if text_config.moe_params.num_shared_experts:
                total_experts += text_config.moe_params.num_shared_experts
            self.moe_enabled = total_experts > 1

        if self.moe_enabled:
            self.moe_params = text_config.moe_params
            self.load_balancing_loss_fn = LoadBalancingLoss(
                text_config.moe_params.num_experts,
                text_config.moe_params.top_k,
            )

        self.model = self.build_model(config)
        vocab_size = getattr(config, ModalityType.TEXT.value).vocab_size
        self.loss_fn = GPTLMHeadModelLoss(
            vocab_size,
            config.loss_scaling,
            config.loss_weight,
        )

    @property
    def modalities(self):
        return self._modalities

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, config):
        # Build individual model towers for each modalities
        modality_models = self.build_modality_models(config)

        # Build projectors for each of the modalities
        projectors = self.build_projectors(config)

        # Initialize LLaVA model
        model = Llava(
            **modality_models,
            **projectors,
            freeze=self.freeze,
            image_feature_select_layer_idx=self.image_feature_select_layer_idx,
            image_start_idx=self.image_start_idx,
            image_feature_select_mode=self.image_feature_select_mode,
        )

        return model

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

        # Note: attention_mask is a misnomer in this model and actually acts as
        # a loss mask. In the model computation its contents are ignored and
        # only its shape is used.
        _msg = (
            f"LLaVA model expects these data fields: \n"
            f"1. Either `image_data` (or) `image_embeddings` \n"
            f"2. Either `text_input_ids` (or) `text_embeddings` \n"
            f"3. `labels` \n"
            f"4. `loss_mask` \n"
            f"5. `key_padding_mask` \n"
        )
        assert (
            ("image_data" in data or "image_embeddings" in data)
            and ("text_input_ids" in data or "text_embeddings" in data)
            and "loss_mask" in data
            and "labels" in data
            and "key_padding_mask" in data
        ), _msg

        _msg = (
            f"The dtype for `text_input_ids`, "
            f"`loss_mask`, `labels`, `key_padding_mask "
            f"should be torch.int32"
        )
        assert (
            data["text_input_ids"].dtype == torch.int32
            and data["loss_mask"].dtype == torch.int32
            and data["labels"].dtype == torch.int32
            and data["key_padding_mask"].dtype == torch.int32
        ), _msg

        tgt_key_padding_mask = data["key_padding_mask"]
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(
                data["image_data"].dtype
            )
            tgt_key_padding_mask = (
                tgt_key_padding_mask
                * torch.finfo(tgt_key_padding_mask.dtype).min
            )

        model_outputs = self.model(
            image_data=data.get("image_data"),
            text_input_ids=data.get("text_input_ids"),
            image_embeddings=data.get("image_embeddings"),
            text_embeddings=data.get("text_embeddings"),
            tgt_key_padding_mask=tgt_key_padding_mask,
            attention_span=data.get("attention_span"),
            position_ids=data.get("position_ids"),
            img_start_idx=data.get("img_start_idx"),
        )

        if self.moe_enabled:
            logits, routing_weights, expert_mask = model_outputs
        else:
            logits = model_outputs

        loss = self.loss_fn(
            logits,
            labels=data["labels"],
            attention_mask=data["loss_mask"],  # acts as a loss mask
            reduce_batch=reduce_batch,
        )

        if (
            self.moe_enabled
            and self.moe_params.load_balancing_loss_coef > 0.0
            and self.training
        ):
            load_balance_loss = (
                self.moe_params.load_balancing_loss_coef
                * self.load_balancing_loss_fn(
                    routing_weights,
                    expert_mask,
                    attention_mask=data["loss_mask"],  # acts as a loss mask
                )
            )
            summarize_scalar("load_balance_loss", load_balance_loss)
            loss = loss + load_balance_loss

        if output_logits:
            return loss, logits
        else:
            return loss
