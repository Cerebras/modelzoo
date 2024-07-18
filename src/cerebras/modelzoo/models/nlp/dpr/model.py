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

import torch

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.losses.dpr_loss import DPRLoss
from cerebras.modelzoo.models.nlp.bert.bert_model import BertModel
from cerebras.modelzoo.models.nlp.dpr.dpr_model import DPRModel
from cerebras.pytorch.metrics import AccuracyMetric


@registry.register_model("dpr", datasetprocessor=["DPRHDF5DataProcessor"])
class DPRWrapperModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        model_params = params.model
        self.use_biencoder = model_params.use_biencoder
        # Adding context_to_question loss when mutual_information is ON
        self.mutual_information = model_params.mutual_information
        self.softmax_temperature = model_params.softmax_temperature
        self.compute_eval_metrics = model_params.compute_eval_metrics
        self.pooler_type = model_params.pooler_type
        self.model = self.build_model(model_params)
        self.loss_fn = DPRLoss(
            self.mutual_information, self.softmax_temperature
        )
        if self.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

    def build_bert(self, model_params):
        disable_nsp = model_params.disable_nsp
        vocab_size = model_params.vocab_size
        dropout_rate = model_params.dropout_rate

        position_embedding_type = (
            model_params.position_embedding_type
            if model_params.position_embedding_type
            else "learned"
        ).lower()

        encoder = BertModel(
            vocab_size=vocab_size,
            max_position_embeddings=model_params.max_position_embeddings,
            position_embedding_type=position_embedding_type,
            hidden_size=model_params.hidden_size,
            embedding_dropout_rate=dropout_rate,
            embedding_pad_token_id=(
                model_params.pad_token_id if model_params.pad_token_id else 0
            ),
            mask_padding_in_positional_embed=(
                model_params.mask_padding_in_positional_embed
                if model_params.mask_padding_in_positional_embed
                else False
            ),
            num_hidden_layers=model_params.num_hidden_layers,
            layer_norm_epsilon=model_params.layer_norm_epsilon,
            num_heads=model_params.num_heads,
            attention_module=(
                model_params.attention_module
                if model_params.attention_module
                else "aiayn_attention"
            ),
            extra_attention_params=(
                model_params.extra_attention_params
                if model_params.extra_attention_params
                else {}
            ),
            attention_type=(
                model_params.attention_type
                if model_params.attention_type
                else "scaled_dot_product"
            ),
            attention_softmax_fp32=(
                model_params.attention_softmax_fp32
                if model_params.attention_softmax_fp32
                else True
            ),
            dropout_rate=dropout_rate,
            nonlinearity=(
                model_params.encoder_nonlinearity
                if model_params.encoder_nonlinearity
                else "gelu"
            ),
            pooler_nonlinearity=(
                model_params.pooler_nonlinearity
                if model_params.pooler_nonlinearity
                else None
            ),
            attention_dropout_rate=model_params.attention_dropout_rate,
            use_projection_bias_in_attention=(
                model_params.use_projection_bias_in_attention
                if model_params.use_projection_bias_in_attention
                else True
            ),
            use_ffn_bias_in_attention=(
                model_params.use_ffn_bias_in_attention
                if model_params.use_ffn_bias_in_attention
                else True
            ),
            filter_size=model_params.filter_size,
            use_ffn_bias=(
                model_params.use_ffn_bias if model_params.use_ffn_bias else True
            ),
            initializer_range=(
                model_params.initializer_range
                if model_params.initializer_range
                else 0.02
            ),
            num_segments=(
                model_params.num_segments
                if model_params.num_segments
                else None if disable_nsp else 2
            ),
            add_pooling_layer=self.pooler_type == "ffn_pooler",
            freeze_ffn_bias_in_glu=model_params.freeze_ffn_bias_in_glu,
        )

        return encoder

    def build_model(self, model_params):
        if self.use_biencoder:
            return self.build_model_biencoder(model_params)
        else:
            return self.build_model_uniencoder(model_params)

    def build_model_uniencoder(self, model_params):
        encoder_params = model_params.encoder

        encoder = self.build_bert(encoder_params)

        return DPRModel(
            encoder,
            None,
            self.pooler_type,
            self.mutual_information,
        )

    def build_model_biencoder(self, model_params):
        if (
            model_params.q_encoder is not None
            and model_params.ctx_encoder is not None
        ):
            q_encoder_params = model_params.q_encoder
            ctx_encoder_params = model_params.ctx_encoder
        else:
            q_encoder_params = model_params.encoder
            ctx_encoder_params = copy.deepcopy(q_encoder_params)

        question_encoder = self.build_bert(q_encoder_params)
        ctx_encoder = self.build_bert(ctx_encoder_params)

        return DPRModel(
            question_encoder,
            ctx_encoder,
            self.pooler_type,
            self.mutual_information,
        )

    def __call__(self, data):
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
