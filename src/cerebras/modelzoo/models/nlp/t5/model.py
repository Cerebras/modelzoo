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

import torch.nn as nn

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.registry import registry
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
from cerebras.modelzoo.models.nlp.t5.t5_model import T5ForConditionalGeneration
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


@registry.register_model(
    [
        "t5",
    ],
    datasetprocessor=["T5HDF5DataProcessor", "T5DynamicDataProcessor"],
)
@registry.register_model(
    "transformer", datasetprocessor=["TransformerDynamicDataProcessor"]
)
class T5ForConditionalGenerationModel(nn.Module):
    """
    T5 models
    """

    def __init__(self, params):
        super().__init__()

        pol = cstorch.backends.csx.precision.optimization_level
        if pol == 2 or (pol == 1 and params.model.fp16_type == "cbfloat16"):
            params.model.attention_softmax_fp32 = False

        self._model_params = params.model
        self.lr_adjustment_groups = self.create_default_lr_adjustment_groups()
        self.model = self.build_model()
        self.loss_fn = T5ForConditionalGenerationLoss(
            self._model_params.lm_loss_weight,
            self._model_params.mlm_loss_scaling,
            self._model_params.label_smoothing,
        )

        if self._model_params.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy_lm")
            self.perplexity_metric = PerplexityMetric(name="eval/perplexity_lm")

    def _post_device_transfer(self):
        self.model.tie_weights()

    def model_class(self):
        return T5ForConditionalGeneration

    def build_model(self):
        model_params = self._model_params

        mup_config = is_mup(model_params)
        if model_params.scale_encoder_qk_dot_by_d is None:
            if mup_config:
                model_params.scale_encoder_qk_dot_by_d = True
                logging.warning(
                    "Found muP params but no scale_encoder_qk_dot_by_d was provided, "
                    "so it will be automatically set to 'True' as the muP "
                    "default."
                )
            else:
                model_params.scale_encoder_qk_dot_by_d = False
        if model_params.scale_decoder_qk_dot_by_d is None:
            if mup_config:
                model_params.scale_decoder_qk_dot_by_d = True
                logging.warning(
                    "Found muP params but no scale_decoder_qk_dot_by_d was provided, "
                    "so it will be automatically set to 'True' as the muP "
                    "default."
                )
            else:
                model_params.scale_decoder_qk_dot_by_d = False

        kwargs = {
            "src_vocab_size": model_params.src_vocab_size,
            "tgt_vocab_size": model_params.tgt_vocab_size,
            "mlm_loss_scaling": model_params.mlm_loss_scaling,
            "label_smoothing": model_params.label_smoothing,
            "extra_ids": model_params.extra_ids,
            "d_model": model_params.d_model,
            "d_kv": model_params.d_kv,
            "d_ff": model_params.d_ff,
            "encoder_num_hidden_layers": model_params.encoder_num_hidden_layers,
            "decoder_num_hidden_layers": model_params.decoder_num_hidden_layers,
            "num_heads": model_params.num_heads,
            "use_projection_bias_in_attention": model_params.use_projection_bias_in_attention,
            "relative_attention_num_buckets": model_params.relative_attention_num_buckets,
            # This param ties weights between lm_head and
            # decoder.embed_tokens layers.
            "share_embedding_weights": model_params.share_embedding_weights,
            "norm_type": model_params.norm_type,
            "dropout_rate": model_params.dropout_rate,
            "layer_norm_epsilon": model_params.layer_norm_epsilon,
            "encoder_nonlinearity": model_params.encoder_nonlinearity,
            "decoder_nonlinearity": model_params.decoder_nonlinearity,
            "position_embedding_type": model_params.position_embedding_type,
            "src_max_position_embeddings": model_params.src_max_position_embeddings,
            "tgt_max_position_embeddings": model_params.tgt_max_position_embeddings,
            "use_dropout_outside_residual_path": model_params.use_dropout_outside_residual_path,
            # This param ties weights between encoder.embed_tokens and
            # decoder.embed_tokens layers.
            "share_encoder_decoder_embedding": model_params.share_encoder_decoder_embedding,
            "relu_dropout_rate": model_params.relu_dropout_rate,
            "use_pre_encoder_decoder_dropout": model_params.use_pre_encoder_decoder_dropout,
            "use_pre_encoder_decoder_layer_norm": model_params.use_pre_encoder_decoder_layer_norm,
            "use_ffn_bias": model_params.use_ffn_bias,
            "lm_loss_weight": model_params.lm_loss_weight,
            "use_transformer_initialization": model_params.use_transformer_initialization,
            # muP (maximal update parameterization)  parameters
            "lr_adjustment_groups": self.lr_adjustment_groups,
            "mup_base_d_model": model_params.mup_base_d_model,
            "mup_base_d_ff": model_params.mup_base_d_ff,
            "mup_base_d_kv": model_params.mup_base_d_kv,
            "embeddings_alpha": model_params.embeddings_alpha,
            "encoder_attention_logits_alpha": model_params.encoder_attention_logits_alpha,
            "decoder_attention_logits_alpha": model_params.decoder_attention_logits_alpha,
            "scale_encoder_qk_dot_by_d": model_params.scale_encoder_qk_dot_by_d,
            "scale_decoder_qk_dot_by_d": model_params.scale_decoder_qk_dot_by_d,
            "scale_output_logits_by_d": model_params.scale_output_logits_by_d,
            "output_logits_alpha": model_params.output_logits_alpha,
            "attention_softmax_fp32": model_params.attention_softmax_fp32,
            "attention_kernel": model_params.attention_kernel,
            "dtype": get_embedding_dtype(
                model_params.mixed_precision, model_params.fp16_type
            ),
        }

        # Updating input and model params to account extra ids
        # for T5 Language Modeling task.
        extra_ids = kwargs.pop("extra_ids", 0)
        kwargs["src_vocab_size"] += extra_ids

        # T5 model has the same vocabulary size for source and target
        # sequences.
        if kwargs["tgt_vocab_size"] is None:
            kwargs["tgt_vocab_size"] = kwargs["src_vocab_size"]
        else:
            kwargs["tgt_vocab_size"] += extra_ids

        # T5 model does not distinguish dropout rate for
        # after relu computations, and utilizes the common dropout rate
        # across the whole model. Transformer, however, is using `0`
        # dropout rate there.
        if kwargs["relu_dropout_rate"] is None:
            kwargs["relu_dropout_rate"] = kwargs["dropout_rate"]

        cls = self.model_class()
        model = cls(**kwargs)

        return model

    def _xentropy_loss(self, labels, logits, weights=None):
        """
        Calculates MLM Cross-Entropy (to be used for Perplexity calculation)

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
        if not self.model.training and self._model_params.compute_eval_metrics:
            labels = data["labels"].clone()
            decoder_mask = (
                data["decoder_attention_mask"].clone().to(logits.dtype)
            )
            predictions = logits.argmax(-1).int()

            self.accuracy_metric(
                labels=labels,
                predictions=predictions,
                weights=decoder_mask,
                dtype=logits.dtype,
            )

            # eval/perplexity_lm
            cross_entropy_loss = self._xentropy_loss(
                labels, logits, decoder_mask
            )
            self.perplexity_metric(
                labels=labels,
                loss=cross_entropy_loss,
                weights=decoder_mask,
                dtype=logits.dtype,
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
