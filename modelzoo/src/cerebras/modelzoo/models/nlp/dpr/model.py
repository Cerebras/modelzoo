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

import copy
from typing import Any, Literal, Optional

import torch
from pydantic import Field, model_validator

from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.losses.dpr_loss import DPRLoss
from cerebras.modelzoo.models.nlp.bert.bert_model import (
    BertModel,
    BertModelConfig,
)
from cerebras.modelzoo.models.nlp.dpr.dpr_model import DPRModel
from cerebras.pytorch.metrics import AccuracyMetric


class DPREncoderConfig(BertModelConfig):
    # Includes the same Bert model params + the following:
    add_pooling_layer: bool = False

    disable_nsp: bool = False
    "Disables Next Sentence Prediction (NSP) objective."

    num_segments: Optional[int] = None
    """Number of segments (token types) in embedding. When not specified
    (and NSP objective is enabled), num_segments will default to 2"""

    pad_token_id: int = 0
    "The embedding vector at pad_token_id is not updated during training."

    encoder_nonlinearity: ActivationType = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block.
    See list of non-linearity functions [here](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
    """

    # The following fields are deprecated and unused.
    # They will be removed in the future once all configs have been fixed
    # These are required because the checkpoint converter doesn't distinguish between bert model types
    mlm_loss_weight: Optional[Any] = Field(default=None, deprecated=True)
    mlm_nonlinearity: Optional[Any] = Field(default=None, deprecated=True)
    share_embedding_weights: Optional[Any] = Field(
        default=None, deprecated=True
    )

    def post_init(self, context):
        super().post_init(context)

        if self.num_segments is None:
            self.num_segments = None if self.disable_nsp else 2

        self.embedding_dropout_rate = self.dropout_rate
        self.embedding_pad_token_id = self.pad_token_id
        self.nonlinearity = self.encoder_nonlinearity


class DPRModelConfig(ModelConfig):
    name: Literal["dpr"]

    q_encoder: Optional[DPREncoderConfig] = None
    "Encoder for question in biencoder model (e.g., DPR)"
    ctx_encoder: Optional[DPREncoderConfig] = None
    "Encoder for context in biencoder model (e.g., DPR)"
    encoder: Optional[DPREncoderConfig] = None
    """
    Encoder for both question and context model.
    - If `encoder` is already provided, users should not provide
    `q_encoder` and `ctx_encoder` in the same config file.
    - Simply providing `encoder` doesn't automatically make the architecture a uni-encoder model;
    instead, the users should explicitly set `use_biencoder` to be False. Otherwise, a bi-encoder
    model will be instantiated with question & context encoders have the same config.
    """
    softmax_temperature: float = 1.0
    "Divide the score matrix by temperature before softmax computation"
    mutual_information: bool = False
    "Whether to add context-to-question loss in addition to question-to-context loss"
    use_biencoder: bool = True
    "Use uniencoder or biencoder architecture"
    pooler_type: Literal["mean", "cls", "ffn_pooler"] = "cls"
    """Pooler method for generating sequence embedding out of output token embeddings.
    Can be one of -
    `mean` -  average all token embeddings as the final sequence embedding,
    `fixed` - use the token embedding of the [CLS] token as the final sequence embedding",
    `ffn_pooler` -  apply an additional linear layer on top of the token embedding of the [CLS] token as the final sequence embedding
    """
    compute_eval_metrics: bool = False
    "Computes accuracy metrics in addition to loss"
    selected_encoder: Optional[
        Literal["q_encoder", "ctx_encoder", "encoder"]
    ] = None
    "Select which encoder to use in embedding_generation. This field is only used in embedding_generation."

    @model_validator(mode="after")
    def validate_encoders(self):
        valid_biencoder_config = (
            self.q_encoder and self.ctx_encoder and not self.encoder
        )
        valid_uniencoder_config = (
            not self.q_encoder and not self.ctx_encoder and self.encoder
        )
        assert (
            valid_uniencoder_config or valid_biencoder_config
        ), "Either provide both q_encoder and ctx_encoder, or only encoder in config"

        if not self.use_biencoder:
            assert (
                valid_uniencoder_config
            ), "If uniencoder is used, only provide encoder attribute in config"

        return self

    def post_init(self, context):
        super().post_init(context)

        add_pooling_layer = self.pooler_type == "ffn_pooler"
        for name in ["q_encoder", "ctx_encoder", "encoder"]:
            if (encoder := getattr(self, name)) is not None:
                setattr(
                    self,
                    name,
                    encoder.copy(
                        update=dict(add_pooling_layer=add_pooling_layer)
                    ),
                )


class DPRWrapperModel(torch.nn.Module):
    def __init__(self, config: DPRModelConfig):
        super().__init__()
        self.use_biencoder = config.use_biencoder
        # Adding context_to_question loss when mutual_information is ON
        self.mutual_information = config.mutual_information
        self.softmax_temperature = config.softmax_temperature
        self.compute_eval_metrics = config.compute_eval_metrics
        self.pooler_type = config.pooler_type
        self.model = self.build_model(config)
        self.loss_fn = DPRLoss(
            self.mutual_information, self.softmax_temperature
        )
        if self.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

    def build_bert(self, config: DPREncoderConfig):
        return BertModel(config)

    def build_model(self, config: DPRModelConfig):
        if self.use_biencoder:
            return self.build_model_biencoder(config)
        else:
            return self.build_model_uniencoder(config)

    def build_model_uniencoder(self, config: DPRModelConfig):
        encoder_params = config.encoder

        encoder = self.build_bert(encoder_params)

        return DPRModel(
            encoder,
            None,
            self.pooler_type,
            self.mutual_information,
        )

    def build_model_biencoder(self, config: DPRModelConfig):
        if config.q_encoder is not None and config.ctx_encoder is not None:
            q_encoder_params = config.q_encoder
            ctx_encoder_params = config.ctx_encoder
        else:
            q_encoder_params = config.encoder
            ctx_encoder_params = copy.deepcopy(q_encoder_params)

        question_encoder = self.build_bert(q_encoder_params)
        ctx_encoder = self.build_bert(ctx_encoder_params)

        return DPRModel(
            question_encoder,
            ctx_encoder,
            self.pooler_type,
            self.mutual_information,
        )

    def forward(self, data):
        labels = data.pop("labels")
        context_labels = data.pop("context_labels", None)
        (
            q2c_scores,
            c2q_scores,
            q_embds,
            ctx_embds,
        ) = self.model(**data)
        total_loss = self.loss_fn(
            q2c_scores, labels, c2q_scores, context_labels
        )

        if not self.model.training and self.compute_eval_metrics:
            embedding_labels = labels.clone()
            embedding_preds = q2c_scores.argmax(-1).int()

            self.accuracy_metric(
                labels=embedding_labels,
                predictions=embedding_preds,
            )

        return total_loss
