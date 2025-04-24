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

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.mup_utils import (
    LRAdjustmentGroup,
    is_mup,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    get_embedding_dtype,
)
from cerebras.modelzoo.losses.T5ForConditionalGenerationLoss import (
    T5ForConditionalGenerationLoss,
)
from cerebras.modelzoo.models.nlp.t5.t5_model import (
    T5ForConditionalGeneration,
    T5ForConditionalGenerationModelConfig,
)
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


class T5ModelConfig(T5ForConditionalGenerationModelConfig):
    name: Literal["t5"]

    # Loss:
    mlm_loss_scaling: Literal[
        "num_masked", "precomputed_num_masked", "batch_size"
    ] = "batch_size"
    """A string specifying the scaling factor type used for the language modeling loss.
    Accepts one of - `"num_masked"` - uses the off-the shelf loss scaling by number
    of valid (non-padding) tokens the cross entropy loss function,
    `"precomputed_num_masked"` - uses loss scaling from the computed num valid
    masks in the data loader, when enabling
    """

    lm_loss_weight: float = 1.0
    """Value that scales loss by the mean number of predictions per sequence in the dataset.
    This number varies per dataset and can be calculated by getting the reciprocal of
    average number of tokens per sequence in the training dataset. This is only needed
    when setting loss scaling to `"batch_size"`."""

    # Misc:
    compute_eval_metrics: Optional[bool] = True
    "Computes perplexity & accuracy metrics in addition to loss"

    @property
    def __model_cls__(self):
        from cerebras.modelzoo.models.nlp.t5.model import (
            T5ForConditionalGenerationModel,
        )

        return T5ForConditionalGenerationModel


class T5ForConditionalGenerationModel(nn.Module):
    """
    T5 models.
    """

    def __init__(self, config: T5ModelConfig):
        if isinstance(config, dict):
            config = T5ModelConfig(**config)

        super().__init__()

        pol = cstorch.backends.csx.precision.optimization_level
        if pol == 2 or (
            pol == 1 and cstorch.amp.get_half_dtype_str() == "cbfloat16"
        ):
            self.attention_softmax_fp32 = False
        else:
            self.attention_softmax_fp32 = config.attention_softmax_fp32

        self.lr_adjustment_groups = self.create_default_lr_adjustment_groups()
        self.model = self.build_model(config)
        self.loss_fn = T5ForConditionalGenerationLoss(
            config.lm_loss_weight,
            config.mlm_loss_scaling,
            config.label_smoothing,
        )

        self.compute_eval_metrics = config.compute_eval_metrics
        if self.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy_lm")
            self.perplexity_metric = PerplexityMetric(name="eval/perplexity_lm")

    def _post_device_transfer(self):
        self.model.tie_weights()

    def model_class(self):
        return T5ForConditionalGeneration

    def build_model(self, config):
        mup_config = is_mup(config)

        scale_encoder_qk_dot_by_d = config.scale_encoder_qk_dot_by_d
        if scale_encoder_qk_dot_by_d is None:
            if mup_config:
                scale_encoder_qk_dot_by_d = True
                logging.warning(
                    "Found muP params but no scale_encoder_qk_dot_by_d was provided, "
                    "so it will be automatically set to 'True' as the muP "
                    "default."
                )
            else:
                scale_encoder_qk_dot_by_d = False

        scale_decoder_qk_dot_by_d = config.scale_decoder_qk_dot_by_d
        if scale_decoder_qk_dot_by_d is None:
            if mup_config:
                scale_decoder_qk_dot_by_d = True
                logging.warning(
                    "Found muP params but no scale_decoder_qk_dot_by_d was provided, "
                    "so it will be automatically set to 'True' as the muP "
                    "default."
                )
            else:
                scale_decoder_qk_dot_by_d = False

        update = {
            # Updating input and model params to account extra ids
            # for T5 Language Modeling task.
            "src_vocab_size": config.src_vocab_size + config.extra_ids,
            "tgt_vocab_size": (
                config.tgt_vocab_size + config.extra_ids
                if config.tgt_vocab_size
                else None
            ),
            # muP (maximal update parameterization)  parameters
            "lr_adjustment_groups": self.lr_adjustment_groups,
            "scale_encoder_qk_dot_by_d": scale_encoder_qk_dot_by_d,
            "scale_decoder_qk_dot_by_d": scale_decoder_qk_dot_by_d,
            "attention_softmax_fp32": self.attention_softmax_fp32,
            "dtype": get_embedding_dtype(
                dtype=cstorch.amp.get_floating_point_dtype_str(),
            ),
        }

        cls = self.model_class()
        return cls(config.copy(update=update))

    def _xentropy_loss(self, labels, logits, weights=None):
        """
        Calculates MLM Cross-Entropy (to be used for Perplexity calculation).

        Args:
            labels: Tensor of shape (batch, sequence) and type int32.
            logits: Tensor of shape (batch, sequence, vocab) and type float.
            weights: Optional float Tensor of shape (batch, sequence).
        Returns:
            The loss tensor

        """
        labels = labels.detach()
        logits = logits.detach()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        vocab_size = logits.shape[2]
        loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1).long(),
        )
        if weights is not None:
            weights = weights.detach()
            loss = loss * weights.view(-1)
        return loss.sum()

    def forward(self, data):
        kwargs = {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "decoder_input_ids": data["decoder_input_ids"],
            "decoder_attention_mask": data["decoder_attention_mask"],
            "labels": data["labels"],
        }
        logits = self.model(**kwargs)
        loss = None
        if data["labels"] is not None:
            loss = self.loss_fn(
                logits,
                data["labels"],
                data["decoder_attention_mask"],
                data.get("loss_weight", None),
            )

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            labels = data["labels"].clone()
            decoder_mask = (
                data["decoder_attention_mask"].clone().to(logits.dtype)
            )
            predictions = logits.argmax(-1).int()

            metric_dtype = (
                torch.float32
                if cstorch.amp.is_cbfloat16_tensor(logits)
                else logits.dtype
            )

            self.accuracy_metric(
                labels=labels,
                predictions=predictions,
                weights=decoder_mask,
                dtype=metric_dtype,
            )

            # eval/perplexity_lm
            cross_entropy_loss = self._xentropy_loss(
                labels, logits, decoder_mask
            )
            self.perplexity_metric(
                labels=labels,
                loss=cross_entropy_loss,
                weights=decoder_mask,
                dtype=metric_dtype,
            )
        return loss

    def create_default_lr_adjustment_groups(self):
        return {
            "embedding": LRAdjustmentGroup("*embedding*weight"),
            "decoder_qkv_projection": LRAdjustmentGroup(
                [
                    "*decoder*attn.proj_q*dense*weight",
                    "*decoder*attn.proj_k*dense*weight",
                    "*decoder*attn.proj_v*dense*weight",
                ]
            ),
            "decoder_output_projection": LRAdjustmentGroup(
                "*decoder*attn.proj_output*dense*weight"
            ),
            "decoder_input_ffn": LRAdjustmentGroup(
                "*decoder*ffn.ffn.[!1]*weight"
            ),
            "decoder_output_ffn": LRAdjustmentGroup(
                "*decoder*ffn.ffn.[1]*weight"
            ),
            "encoder_qkv_projection": LRAdjustmentGroup(
                [
                    "*encoder*attn.proj_q*dense*weight",
                    "*encoder*attn.proj_k*dense*weight",
                    "*encoder*attn.proj_v*dense*weight",
                ]
            ),
            "encoder_output_projection": LRAdjustmentGroup(
                "*encoder*attn.proj_output*dense*weight"
            ),
            "encoder_input_ffn": LRAdjustmentGroup(
                "*encoder*ffn.ffn.[!1]*weight"
            ),
            "encoder_output_ffn": LRAdjustmentGroup(
                "*encoder*ffn.ffn.[1]*weight"
            ),
            "lm_head": LRAdjustmentGroup("*lm_head*weight"),
        }
