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
from torch.nn import CrossEntropyLoss

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    get_embedding_dtype,
)
from cerebras.modelzoo.losses.BertPretrainModelLoss import BertPretrainModelLoss
from cerebras.modelzoo.models.nlp.bert.bert_pretrain_models import (
    BertForPreTrainingModelConfig,
    BertPretrainModel,
)
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


class BertForPreTrainingModel(torch.nn.Module):
    """
    BERT-based models.
    """

    def __init__(self, config: BertForPreTrainingModelConfig):
        super().__init__()

        pol = cstorch.backends.csx.precision.optimization_level
        if pol == 2 or (
            pol == 1 and cstorch.amp.get_half_dtype_str() == "cbfloat16"
        ):
            attention_softmax_fp32 = False
        else:
            attention_softmax_fp32 = config.attention_softmax_fp32

        config = config.copy(
            update=dict(
                attention_softmax_fp32=attention_softmax_fp32,
                dtype=get_embedding_dtype(
                    dtype=cstorch.amp.get_floating_point_dtype_str(),
                ),
            )
        )

        self.disable_nsp = config.disable_nsp
        self.compute_eval_metrics = config.compute_eval_metrics
        self.vocab_size = config.vocab_size

        self.model = self.build_model(config)
        self.loss_fn = BertPretrainModelLoss(
            disable_nsp=self.disable_nsp,
            mlm_loss_weight=config.mlm_loss_weight,
            label_smoothing=config.label_smoothing,
        )

        if self.compute_eval_metrics:
            if not self.disable_nsp:
                self.accuracy_metric_cls = AccuracyMetric(
                    name="eval/accuracy_cls"
                )
            self.accuracy_metric_mlm = AccuracyMetric(
                name="eval/accuracy_masked_lm"
            )
            self.perplexity_metric = PerplexityMetric(
                name="eval/mlm_perplexity"
            )

    @property
    def lr_adjustment_groups(self):
        return getattr(self.model, "get_lr_adjustment_groups", lambda: {})()

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, config: BertForPreTrainingModelConfig):
        return BertPretrainModel(config)

    def mlm_xentropy_loss(self, labels, logits, weights=None):
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
        mlm_loss_scale = data.pop("mlm_loss_scale", None)
        labels = data.pop("labels")

        _, len_labels = list(labels.size())
        batch_size, seq_len = data["input_ids"].shape[:2]
        should_gather_mlm_labels = len_labels != seq_len
        data["should_gather_mlm_labels"] = should_gather_mlm_labels

        # MLM Needs a half precision "weights" tensor; use binary mask for now.
        masked_lm_mask = data.pop("masked_lm_mask", None)
        if not should_gather_mlm_labels:
            masked_lm_mask = torch.ones(
                batch_size,
                seq_len,
                device=labels.device,
            )

        mlm_logits, nsp_logits, _, _ = self.model(**data)

        if mlm_loss_scale is not None:
            mlm_loss_scale = mlm_loss_scale.to(mlm_logits.dtype)

        masked_lm_mask = masked_lm_mask.to(mlm_logits.dtype)

        total_loss = self.loss_fn(
            mlm_logits,
            self.vocab_size,
            labels,
            nsp_logits,
            next_sentence_label,
            masked_lm_mask,
            mlm_loss_scale,
        )

        if not self.model.training and self.compute_eval_metrics:
            metric_dtype = (
                torch.float32
                if cstorch.amp.is_cbfloat16_tensor(mlm_logits)
                else mlm_logits.dtype
            )
            if not self.disable_nsp:
                nsp_label = next_sentence_label.clone()
                nsp_pred = nsp_logits.argmax(-1).int()
                # eval/accuracy_cls
                self.accuracy_metric_cls(
                    labels=nsp_label,
                    predictions=nsp_pred,
                    dtype=metric_dtype,
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
                dtype=metric_dtype,
            )
            # eval/mlm_perplexity
            self.perplexity_metric(
                labels=mlm_labels,
                loss=mlm_xentr,
                weights=mlm_weights,
                dtype=metric_dtype,
            )

        return total_loss
