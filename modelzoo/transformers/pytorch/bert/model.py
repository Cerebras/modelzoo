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

from torch.nn import CrossEntropyLoss

from modelzoo.common.pytorch.metrics import AccuracyMetric, PerplexityMetric
from modelzoo.common.pytorch.model_utils.BertPretrainModelLoss import (
    BertPretrainModelLoss,
)
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.bert.bert_pretrain_models import (
    BertPretrainModel,
)
from modelzoo.transformers.pytorch.bert.utils import (
    check_unused_model_params,
    set_custom_stack_params,
)


class BertForPreTrainingModel(PyTorchBaseModel):
    """
    BERT-based models
    """

    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = BertPretrainModelLoss(
            disable_nsp=self.disable_nsp,
            mlm_loss_weight=self.mlm_loss_weight,
            label_smoothing=self.label_smoothing,
        )
        super().__init__(params=params, model=self.model, device=device)

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
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

        # Add stack flags for performance runs
        set_custom_stack_params(params)

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, model_params):
        self.disable_nsp = model_params.pop("disable_nsp", False)
        self.mlm_loss_weight = model_params.pop("mlm_loss_weight", 1.0)
        self.label_smoothing = model_params.pop("label_smoothing", 0.0)
        self.vocab_size = model_params.pop("vocab_size")

        model = BertPretrainModel(
            disable_nsp=self.disable_nsp,
            mlm_loss_weight=self.mlm_loss_weight,
            label_smoothing=self.label_smoothing,
            vocab_size=self.vocab_size,
            max_position_embeddings=model_params.pop("max_position_embeddings"),
            position_embedding_type=model_params.pop(
                "position_embedding_type", "learned"
            ).lower(),
            embedding_pad_token_id=model_params.pop("pad_token_id", 0),
            hidden_size=model_params.pop("hidden_size"),
            share_embedding_weights=model_params.pop(
                "share_embedding_weights", True
            ),
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            layer_norm_epsilon=float(model_params.pop("layer_norm_epsilon")),
            # Encoder Attn
            num_heads=model_params.pop("num_heads"),
            attention_type=model_params.pop(
                "attention_type", "scaled_dot_product"
            ),
            dropout_rate=model_params.pop("dropout_rate"),
            nonlinearity=model_params.pop("encoder_nonlinearity", "gelu"),
            attention_dropout_rate=model_params.pop("attention_dropout_rate"),
            use_projection_bias_in_attention=model_params.pop(
                "use_projection_bias_in_attention", True
            ),
            use_ffn_bias_in_attention=model_params.pop(
                "use_ffn_bias_in_attention", True
            ),
            filter_size=model_params.pop("filter_size"),
            use_ffn_bias=model_params.pop("use_ffn_bias", True),
            use_ffn_bias_in_mlm=model_params.pop("use_ffn_bias_in_mlm", True),
            use_output_bias_in_mlm=model_params.pop(
                "use_output_bias_in_mlm", True
            ),
            initializer_range=model_params.pop("initializer_range", 0.02),
            num_segments=None if self.disable_nsp else 2,
        )

        enable_vts = model_params.pop("enable_vts")
        if enable_vts:
            from modelzoo.common.pytorch import cbtorch

            self.vts = cbtorch.nn.StripPadding()
        else:
            self.vts = None

        check_unused_model_params(model_params)

        return model

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
        loss_fct = CrossEntropyLoss(reduction="none",)
        vocab_size = logits.shape[2]
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1).long(),)
        if weights is not None:
            weights = weights.detach()
            loss = loss * weights.view(-1)
        return loss.sum()

    def __call__(self, data):

        if self.vts and not self.model.training:
            self.vts = None
            logging.info(
                "VTS is only supported in train mode. Disabling for the "
                "current run."
            )

        if self.vts:
            # always mask the main input
            masks = {
                "attention_mask": ["input_ids"],
            }
            # if NSP is enabled, mask it with main mask
            if "token_type_ids" in data:
                masks["attention_mask"].append("token_type_ids")

            # Check if we're "gathering" MLM predictions:
            if "masked_lm_positions" in data:
                # Yes, so use the masked_lm_mask for MLM tensors
                masks["masked_lm_mask"] = ["masked_lm_positions", "labels"]
            else:
                # No, so use main mask for MLM label
                masks["attention_mask"].append("labels")

            for mask, inputs in masks.items():
                mask_tensor = data[mask]
                for name in inputs:
                    data[name] = self.vts(data[name], mask_tensor)

        # MLM Needs a half precision "weights" tensor; use binary mask for now.
        data["masked_lm_weights"] = data.pop("masked_lm_mask").half()
        mlm_logits, nsp_logits, _, _ = self.model(**data)
        total_loss = None
        if data.get("should_calc_loss", True):
            total_loss = self.loss_fn(
                mlm_logits,
                self.vocab_size,
                data["labels"],
                nsp_logits,
                data.get("next_sentence_label", None),
                data["masked_lm_weights"],
                data.get("mlm_loss_scale", None),
            )
        if not self.model.training and self.compute_eval_metrics:
            if not self.disable_nsp:

                nsp_label = data["next_sentence_label"].clone()
                nsp_pred = nsp_logits.argmax(-1).int()
                # eval/accuracy_cls
                self.accuracy_metric_cls(labels=nsp_label, predictions=nsp_pred)

            mlm_preds = mlm_logits.argmax(-1).int()

            mlm_labels = data["labels"].clone()
            mlm_weights = data["masked_lm_weights"].clone()
            mlm_xentr = self.mlm_xentropy_loss(
                mlm_labels, mlm_logits, mlm_weights
            )

            # eval/accuracy_masked_lm
            self.accuracy_metric_mlm(
                labels=mlm_labels, predictions=mlm_preds, weights=mlm_weights,
            )
            # eval/mlm_perplexity
            self.perplexity_metric(
                labels=mlm_labels, loss=mlm_xentr, weights=mlm_weights,
            )

        return total_loss
