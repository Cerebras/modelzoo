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

import torch
from torch.nn import CrossEntropyLoss

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.common.utils.model.mup_utils import (
    LRAdjustmentGroup,
    is_mup,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    get_embedding_dtype,
)
from cerebras.modelzoo.losses.BertPretrainModelLoss import BertPretrainModelLoss
from cerebras.modelzoo.models.nlp.bert.bert_pretrain_models import (
    BertPretrainModel,
)
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


@registry.register_model(
    [
        "bert",
    ],
    datasetprocessor=[
        "BertCSVDataProcessor",
        "BertCSVDynamicMaskDataProcessor",
        "BertSumCSVDataProcessor",
    ],
)
class BertForPreTrainingModel(torch.nn.Module):
    """
    BERT-based models
    """

    def __init__(self, params):
        super().__init__()

        pol = cstorch.backends.csx.precision.optimization_level
        if pol == 2 or (pol == 1 and params.model.fp16_type == "cbfloat16"):
            params.model.attention_softmax_fp32 = False

        self.lr_adjustment_groups = self.create_default_lr_adjustment_groups()
        self._model_params = params.model
        self.model = self.build_model()
        self.loss_fn = BertPretrainModelLoss(
            disable_nsp=self._model_params.disable_nsp,
            mlm_loss_weight=self._model_params.mlm_loss_weight,
            label_smoothing=self._model_params.label_smoothing,
        )

        if self._model_params.compute_eval_metrics:
            if not self._model_params.disable_nsp:
                self.accuracy_metric_cls = AccuracyMetric(
                    name="eval/accuracy_cls"
                )
            self.accuracy_metric_mlm = AccuracyMetric(
                name="eval/accuracy_masked_lm"
            )
            self.perplexity_metric = PerplexityMetric(
                name="eval/mlm_perplexity"
            )

    def _post_device_transfer(self):
        self.model.tie_weights()

    def model_class(self):
        return BertPretrainModel

    def build_model(self):
        cls = self.model_class()
        args = self.build_model_args()
        return cls(**args)

    def build_model_args(self):
        mup_config = is_mup(self._model_params)
        if self._model_params.scale_qk_dot_by_d is None:
            if mup_config:
                self._model_params.scale_qk_dot_by_d = True
                logging.warning(
                    "Found muP params but no scale_qk_dot_by_d was provided, "
                    "so it will be automatically set to 'True' as the muP "
                    "default."
                )
            else:
                self._model_params.scale_qk_dot_by_d = False

        rotary_dim = None
        if self._model_params.position_embedding_type == "rotary":
            # rotary_dim defaults to 25% of head dim (hidden_size / num_heads)
            # similar to other models that use RoPE like GPT-NeoX
            rotary_dim = self._model_params.rotary_dim
            if rotary_dim is None:
                rotary_dim = int(
                    self._model_params.hidden_size
                    // self._model_params.num_heads
                    * 0.25
                )
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3
            assert (
                rotary_dim
                <= self._model_params.hidden_size / self._model_params.num_heads
            ), "Rotary dimensions should be <= hidden size divided by number of attention heads."
            assert (
                rotary_dim % 2 == 0
            ), "Rotary dimension must be an even number."
        return {
            "disable_nsp": self._model_params.disable_nsp,
            "num_classes": self._model_params.num_classes,
            "vocab_size": self._model_params.vocab_size,
            "max_position_embeddings": self._model_params.max_position_embeddings,
            "position_embedding_type": self._model_params.position_embedding_type,
            "embedding_pad_token_id": self._model_params.pad_token_id,
            "mask_padding_in_positional_embed": self._model_params.mask_padding_in_positional_embed,
            "rotary_dim": rotary_dim,
            "rope_theta": self._model_params.rope_theta,
            "pad_rope": self._model_params.pad_rope,
            "num_relative_attention_buckets": self._model_params.num_relative_attention_buckets,
            "alibi_trainable_slopes": self._model_params.alibi_trainable_slopes,
            "pos_scaling_factor": self._model_params.pos_scaling_factor,
            "hidden_size": self._model_params.hidden_size,
            "share_embedding_weights": self._model_params.share_embedding_weights,
            "num_hidden_layers": self._model_params.num_hidden_layers,
            "layer_norm_epsilon": self._model_params.layer_norm_epsilon,
            # Encoder Attn
            "num_heads": self._model_params.num_heads,
            "attention_module": self._model_params.attention_module,
            "extra_attention_params": self._model_params.extra_attention_params,
            "attention_type": self._model_params.attention_type,
            "dropout_rate": self._model_params.dropout_rate,
            "nonlinearity": self._model_params.encoder_nonlinearity,
            "mlm_nonlinearity": self._model_params.mlm_nonlinearity,
            "pooler_nonlinearity": self._model_params.pooler_nonlinearity,
            "attention_dropout_rate": self._model_params.attention_dropout_rate,
            "attention_softmax_fp32": self._model_params.attention_softmax_fp32,
            "attention_kernel": self._model_params.attention_kernel,
            "use_projection_bias_in_attention": self._model_params.use_projection_bias_in_attention,
            "use_ffn_bias_in_attention": self._model_params.use_ffn_bias_in_attention,
            "filter_size": self._model_params.filter_size,
            "use_ffn_bias": self._model_params.use_ffn_bias,
            "use_ffn_bias_in_mlm": self._model_params.use_ffn_bias_in_mlm,
            "use_output_bias_in_mlm": self._model_params.use_output_bias_in_mlm,
            "initializer_range": self._model_params.initializer_range,
            "num_segments": self._model_params.num_segments,
            "dtype": get_embedding_dtype(
                self._model_params.mixed_precision, self._model_params.fp16_type
            ),
            # muP (maximal update parameterization)  parameters
            "lr_adjustment_groups": self.lr_adjustment_groups,
            "embeddings_scale": self._model_params.embeddings_scale,
            "scale_qk_dot_by_d": self._model_params.scale_qk_dot_by_d,
            "attention_logits_alpha": self._model_params.attention_logits_alpha,
            "scale_output_logits_by_d": self._model_params.scale_output_logits_by_d,
            "mup_base_hidden_size": self._model_params.mup_base_hidden_size,
            "mup_base_filter_size": self._model_params.mup_base_filter_size,
            "output_logits_alpha": self._model_params.output_logits_alpha,
        }

    def mlm_xentropy_loss(self, labels, logits, weights=None):
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
        loss_fct = CrossEntropyLoss(reduction="none")
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
        next_sentence_label = data.pop("next_sentence_label", None)
        should_calc_loss = data.pop("should_calc_loss", True)
        mlm_loss_scale = data.pop("mlm_loss_scale", None)
        labels = data.pop("labels")

        _, len_labels = list(labels.size())
        batch_size, seq_len = data["input_ids"].shape[:2]
        should_gather_mlm_labels = len_labels != seq_len
        data["should_gather_mlm_labels"] = should_gather_mlm_labels

        # MLM Needs a half precision "weights" tensor; use binary mask for now.
        if should_gather_mlm_labels:
            masked_lm_mask = data.pop("masked_lm_mask")
        else:
            masked_lm_mask = torch.ones(
                batch_size,
                seq_len,
                device=labels.device,
            )

        mlm_logits, nsp_logits, _, _ = self.model(**data)

        if mlm_loss_scale is not None:
            mlm_loss_scale = mlm_loss_scale.to(mlm_logits.dtype)

        masked_lm_mask = masked_lm_mask.to(mlm_logits.dtype)

        total_loss = None
        if should_calc_loss:
            total_loss = self.loss_fn(
                mlm_logits,
                self._model_params.vocab_size,
                labels,
                nsp_logits,
                next_sentence_label,
                masked_lm_mask,
                mlm_loss_scale,
            )
        if not self.model.training and self._model_params.compute_eval_metrics:
            if not self._model_params.disable_nsp:
                nsp_label = next_sentence_label.clone()
                nsp_pred = nsp_logits.argmax(-1).int()
                # eval/accuracy_cls
                self.accuracy_metric_cls(
                    labels=nsp_label,
                    predictions=nsp_pred,
                    dtype=mlm_logits.dtype,
                )

            mlm_preds = mlm_logits.argmax(-1).int()

            mlm_labels = labels.clone()
            mlm_weights = masked_lm_mask.clone()
            mlm_xentr = self.mlm_xentropy_loss(
                mlm_labels, mlm_logits, mlm_weights
            )

            # eval/accuracy_masked_lm
            self.accuracy_metric_mlm(
                labels=mlm_labels,
                predictions=mlm_preds,
                weights=mlm_weights,
                dtype=mlm_logits.dtype,
            )
            # eval/mlm_perplexity
            self.perplexity_metric(
                labels=mlm_labels,
                loss=mlm_xentr,
                weights=mlm_weights,
                dtype=mlm_logits.dtype,
            )

        return total_loss

    def create_default_lr_adjustment_groups(self):
        return {
            "embedding": LRAdjustmentGroup("*embedding*weight"),
            "encoder_attention": LRAdjustmentGroup(
                "*transformer_encoder*attn*dense*weight"
            ),
            "encoder_input_ffn": LRAdjustmentGroup(
                "*transformer_encoder*ffn.ffn.[!1]*weight"
            ),
            "encoder_output_ffn": LRAdjustmentGroup(
                "*transformer_encoder*ffn.ffn.[1]*weight"
            ),
            "pooler": LRAdjustmentGroup("*pooler*weight"),
        }
