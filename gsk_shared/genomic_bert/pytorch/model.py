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

from gsk_shared.genomic_bert.pytorch.configuration_bert import GenomicBertConfig
from gsk_shared.genomic_bert.pytorch.modeling_bert import (
    GenomicBertForPreTraining,
)

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.metrics.accuracy import AccuracyMetric
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.bert.utils import check_unused_model_params


class GenomicBertForPreTrainingModel(PyTorchBaseModel):
    """
    The genomic BERT model.

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        self.model = self.build_model(model_params)

        super(GenomicBertForPreTrainingModel, self).__init__(
            params=params, model=self.model, device=device
        )

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.accuracy_metric_dna = AccuracyMetric(
                name="eval/accuracy_masked_lm_dna"
            )
            self.accuracy_metric_ideas = AccuracyMetric(
                name="eval/accuracy_masked_lm_ideas"
            )

        # Add custom Cerebras stack flags
        if cm.use_cs():
            import cerebras.framework.torch as cbtorch

            state = cbtorch.state()
            state.full_config.placement.optimize_buses.deltat_relative_margin = (
                0.5
            )
            state.full_config.matching.kernel.no_dcache_spill_splits = True

    def _post_device_transfer(self):
        self.model.tie_weights()
        self.model.cls.predictions.decoder_dna.bias = (
            self.model.cls.predictions.bias_dna
        )
        self.model.cls.predictions.decoder_ideas.bias = (
            self.model.cls.predictions.bias_ideas
        )

    def build_model(self, model_params):
        self.mlm_loss_weight = model_params.pop("mlm_loss_weight")

        kwargs = {
            "vocab_size_dna": model_params.pop("vocab_size_dna"),
            "vocab_size_ideas": model_params.pop("vocab_size_ideas"),
            "hidden_size": model_params.pop("hidden_size"),
            "num_hidden_layers": model_params.pop("num_hidden_layers"),
            "num_attention_heads": model_params.pop("num_heads"),
            "intermediate_size": model_params.pop("filter_size"),
            "hidden_act": model_params.pop("encoder_nonlinearity"),
            "hidden_dropout_prob": model_params.pop("dropout_rate"),
            "attention_probs_dropout_prob": model_params.pop(
                "attention_dropout_rate"
            ),
            "max_position_embeddings": model_params.pop(
                "max_position_embeddings"
            ),
            "tie_word_embeddings": model_params.pop(
                "share_embedding_weights", True,
            ),
            "layer_norm_eps": float(model_params.pop("layer_norm_epsilon")),
            "use_projection_bias_in_attention": model_params.pop(
                "use_projection_bias_in_attention", False
            ),
            "use_ffn_bias_in_attention": model_params.pop(
                "use_ffn_bias_in_attention", False
            ),
            "use_output_bias_in_mlm": model_params.pop(
                "use_output_bias_in_mlm", True
            ),
            "use_ffn_bias_in_mlm": model_params.pop(
                "use_ffn_bias_in_mlm", True
            ),
            "mlm_nonlinearity": model_params.pop("mlm_nonlinearity", "gelu"),
            "use_ffn_bias": model_params.pop("use_ffn_bias", True),
        }

        check_unused_model_params(model_params)

        model = GenomicBertForPreTraining(
            GenomicBertConfig(**kwargs), mlm_loss_weight=self.mlm_loss_weight,
        )
        self.loss_fn = model.loss_fn
        return model

    def __call__(self, data):
        output = self.model(**data)
        loss = output.loss

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            mlm_labels_dna = data["labels_dna"].clone()
            mlm_labels_ideas = data["labels_ideas"].clone()
            mlm_preds_dna = output.prediction_logits_dna.argmax(-1).int()
            mlm_preds_ideas = output.prediction_logits_ideas.argmax(-1).int()

            # eval/accuracy_dna
            self.accuracy_metric_dna(
                labels=mlm_labels_dna, predictions=mlm_preds_dna,
            )

            # eval/accuracy_ideas
            self.accuracy_metric_ideas(
                labels=mlm_labels_ideas, predictions=mlm_preds_ideas,
            )

        return loss
