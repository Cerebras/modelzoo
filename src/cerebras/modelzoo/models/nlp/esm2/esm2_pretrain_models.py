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

from cerebras.modelzoo.models.nlp.bert.bert_pretrain_models import (
    BertPretrainModel,
)
from cerebras.modelzoo.models.nlp.esm2.esm2_model import Esm2Model


class Esm2PretrainModel(BertPretrainModel):
    """
    Esm-2 Model pretrain model
    """

    def __init__(
        self,
        token_dropout=False,
        mask_token_id=None,
        use_final_layer_norm=False,
        embedding_layer_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.bert_encoder = Esm2Model(
            # ESM-2 params:
            token_dropout=token_dropout,
            mask_token_id=mask_token_id,
            # Regular BERT params:
            # Embedding
            vocab_size=kwargs.get("vocab_size"),
            max_position_embeddings=kwargs.get("max_position_embeddings"),
            position_embedding_type=kwargs.get("position_embedding_type"),
            hidden_size=kwargs.get("hidden_size"),
            embedding_dropout_rate=kwargs.get("dropout_rate"),
            embedding_pad_token_id=kwargs.get("embedding_pad_token_id"),
            mask_padding_in_positional_embed=kwargs.get(
                "mask_padding_in_positional_embed"
            ),
            rotary_dim=kwargs.get("rotary_dim"),
            rope_theta=kwargs.get("rope_theta"),
            pad_rope=kwargs.get("pad_rope"),
            num_relative_attention_buckets=kwargs.get(
                "num_relative_attention_buckets"
            ),
            alibi_trainable_slopes=kwargs.get("alibi_trainable_slopes"),
            pos_scaling_factor=kwargs.get("pos_scaling_factor"),
            # Encoder
            num_hidden_layers=kwargs.get("num_hidden_layers"),
            layer_norm_epsilon=kwargs.get("layer_norm_epsilon"),
            embedding_layer_norm=embedding_layer_norm,
            # Encoder Attn
            num_heads=kwargs.get("num_heads"),
            attention_module=kwargs.get("attention_module"),
            extra_attention_params=kwargs.get("extra_attention_params"),
            attention_type=kwargs.get("attention_type"),
            attention_softmax_fp32=kwargs.get("attention_softmax_fp32"),
            dropout_rate=kwargs.get("dropout_rate"),
            nonlinearity=kwargs.get("nonlinearity"),
            pooler_nonlinearity=kwargs.get("pooler_nonlinearity"),
            attention_dropout_rate=kwargs.get("attention_dropout_rate"),
            use_projection_bias_in_attention=kwargs.get(
                "use_projection_bias_in_attention"
            ),
            use_ffn_bias_in_attention=kwargs.get("use_ffn_bias_in_attention"),
            # Encoder ffn
            filter_size=kwargs.get("filter_size"),
            use_ffn_bias=kwargs.get("use_ffn_bias"),
            # Task-specific
            use_final_layer_norm=use_final_layer_norm,
            initializer_range=kwargs.get("initializer_range"),
            num_segments=kwargs.get("num_segments"),
            add_pooling_layer=(not self.disable_nsp),
        )
