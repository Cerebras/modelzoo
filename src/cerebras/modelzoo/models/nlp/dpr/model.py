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

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.losses.dpr_loss import DPRLoss
from cerebras.modelzoo.models.nlp.bert.bert_model import BertModel
from cerebras.modelzoo.models.nlp.bert.utils import check_unused_model_params
from cerebras.modelzoo.models.nlp.dpr.dpr_model import DPRModel


@registry.register_model("dpr", datasetprocessor=["DPRHDF5DataProcessor"])
class DPRWrapperModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        model_params = params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = DPRLoss()

    def build_bert(self, model_params):
        disable_nsp = model_params.pop("disable_nsp", False)
        vocab_size = model_params.pop("vocab_size")
        dropout_rate = model_params.pop("dropout_rate")

        position_embedding_type = model_params.pop(
            "position_embedding_type", "learned"
        ).lower()

        assert (
            position_embedding_type == "fixed"
            or position_embedding_type == "learned"
        ), f"Only fixed or learned position embedding is supported by Bert for now, but got {position_embedding_type}"

        encoder = BertModel(
            vocab_size=vocab_size,
            max_position_embeddings=model_params.pop("max_position_embeddings"),
            position_embedding_type=position_embedding_type,
            hidden_size=model_params.pop("hidden_size"),
            embedding_dropout_rate=dropout_rate,
            embedding_pad_token_id=model_params.pop("pad_token_id", 0),
            mask_padding_in_positional_embed=model_params.pop(
                "mask_padding_in_positional_embed", False
            ),
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            layer_norm_epsilon=float(model_params.pop("layer_norm_epsilon")),
            num_heads=model_params.pop("num_heads"),
            attention_module=model_params.pop(
                "attention_module", "aiayn_attention"
            ),
            extra_attention_params=model_params.pop(
                "extra_attention_params", {}
            ),
            attention_type=model_params.pop(
                "attention_type", "scaled_dot_product"
            ),
            attention_softmax_fp32=model_params.pop(
                "attention_softmax_fp32", True
            ),
            dropout_rate=dropout_rate,
            nonlinearity=model_params.pop("encoder_nonlinearity", "gelu"),
            pooler_nonlinearity=model_params.pop("pooler_nonlinearity", None),
            attention_dropout_rate=model_params.pop("attention_dropout_rate"),
            use_projection_bias_in_attention=model_params.pop(
                "use_projection_bias_in_attention", True
            ),
            use_ffn_bias_in_attention=model_params.pop(
                "use_ffn_bias_in_attention", True
            ),
            filter_size=model_params.pop("filter_size"),
            use_ffn_bias=model_params.pop("use_ffn_bias", True),
            initializer_range=model_params.pop("initializer_range", 0.02),
            num_segments=model_params.pop(
                "num_segments", None if disable_nsp else 2
            ),
            add_pooling_layer=model_params.pop("add_pooling_layer", False),
        )

        check_unused_model_params(model_params)
        return encoder

    def build_model(self, model_params):
        hidden_size = model_params["q_encoder"].get("hidden_size")
        q_encoder_params = model_params.pop("q_encoder")
        ctx_encoder_params = model_params.pop("ctx_encoder")
        scale_similarity = model_params.pop("scale_similarity", False)
        question_encoder = self.build_bert(q_encoder_params)
        ctx_encoder = self.build_bert(ctx_encoder_params)
        model = DPRModel(
            question_encoder, ctx_encoder, hidden_size, scale_similarity
        )
        return model

    def __call__(self, data):
        labels = data.pop("labels")
        (
            scores,
            q_embds,
            ctx_embds,
        ) = self.model(**data)
        total_loss = self.loss_fn(scores, labels)
        return total_loss
