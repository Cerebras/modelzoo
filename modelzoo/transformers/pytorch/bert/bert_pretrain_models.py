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
import torch.nn as nn

from modelzoo.common.pytorch.layers import FeedForwardNetwork
from modelzoo.common.pytorch.model_utils.BertPretrainModelLoss import (
    BertPretrainModelLoss,
)
from modelzoo.transformers.pytorch.bert.bert_model import BertModel
from modelzoo.transformers.pytorch.transformer_utils import (
    make_key_padding_mask_broadcastable,
)


class BertClassifierHead(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_classes=2,
        use_bias=True,
        kernel_initializer="xavier_uniform",
    ):
        super().__init__()

        self.classifier = FeedForwardNetwork(
            input_unit=hidden_size,
            layers_units=[num_classes],
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def forward(self, pooled_output):
        cls_logits = self.classifier(pooled_output)
        return cls_logits


class BertMLMHeadTransform(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        embedding_size=None,
        use_ffn_bias_in_mlm=True,
        activation="gelu",
        layer_norm_epsilon=1.0e-5,
        dropout=None,
        kernel_initializer="xavier_uniform",
    ):
        super().__init__()
        if embedding_size is None:
            embedding_size = hidden_size

        self.ffn = FeedForwardNetwork(
            input_unit=hidden_size,
            layers_units=[embedding_size],
            layers_activation=[activation],
            layers_dropout_rates=[dropout],
            use_bias=use_ffn_bias_in_mlm,
            kernel_initializer=kernel_initializer,
        )

        self.ln = nn.LayerNorm(embedding_size, eps=layer_norm_epsilon)

        self.__reset_parameters()

    def reset_parameters(self):
        self.ffn.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        self.ln.bias.data.zero_()
        self.ln.weight.data.fill_(1.0)

    def forward(self, bert_output):
        mlm_embeddings = self.ffn(bert_output)
        mlm_embeddings = self.ln(mlm_embeddings)
        return mlm_embeddings


class BertMLMHead(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        vocab_size=30522,
        embedding_size=None,
        use_ffn_bias_in_mlm=True,
        use_output_bias_in_mlm=True,
        activation="gelu",
        layer_norm_epsilon=1.0e-5,
        dropout=None,
        kernel_initializer="xavier_uniform",
    ):
        super().__init__()
        if embedding_size is None:
            embedding_size = hidden_size

        self.mlm_transform = BertMLMHeadTransform(
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            use_ffn_bias_in_mlm=use_ffn_bias_in_mlm,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
        )
        self.classifier = FeedForwardNetwork(
            input_unit=embedding_size,
            layers_units=[vocab_size],
            use_bias=use_output_bias_in_mlm,
            kernel_initializer=kernel_initializer,
        )

    def reset_parameters(self):
        self.mlm_transform.reset_parameters()
        self.classifier.reset_parameters()

    def get_output_embeddings(self):
        return self.classifier.ffn[0].linear_layer

    def forward(self, bert_output):
        mlm_embeddings = self.mlm_transform(bert_output)
        mlm_logits = self.classifier(mlm_embeddings)
        return mlm_logits


class BertPretrainModel(nn.Module):
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head. Following the paper: https://arxiv.org/abs/1810.04805.
    """

    def __init__(
        self,
        disable_nsp=False,
        mlm_loss_weight=1.0,
        label_smoothing=0.0,
        num_classes=2,
        mlm_nonlinearity="gelu",
        # Embedding
        vocab_size=50257,
        max_position_embeddings=1024,
        position_embedding_type="learned",
        embedding_pad_token_id=0,
        hidden_size=768,
        share_embedding_weights=True,
        # Encoder
        num_hidden_layers=12,
        layer_norm_epsilon=1.0e-5,
        # Encoder Attn
        num_heads=12,
        attention_module_str="aiayn_attention",
        extra_attention_params={},
        attention_type="scaled_dot_product",
        dropout_rate=0.1,
        nonlinearity="gelu",
        attention_dropout_rate=0.1,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        # Encoder ffn
        filter_size=3072,
        use_ffn_bias=True,
        use_ffn_bias_in_mlm=True,
        use_output_bias_in_mlm=True,
        # Task-specific
        initializer_range=0.02,
        num_segments=2,
    ):
        """
        Args:
            disable_nsp (:obj:`bool` `optional`, defaults to False):
                Whether to disable next-sentence-prediction and only use masked-language-model.
            mlm_loss_weight (:obj:`float` `optional`, defaults to 1.0):
                The scaling factor for masked-language-model loss.
            label_smoothing (:obj:`float` `optional`, defaults to 0.0):
                The label smoothing factor used during training.
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.disable_nsp = disable_nsp
        self.share_embedding_weights = share_embedding_weights
        self.initializer_range = initializer_range

        self.bert_encoder = BertModel(
            # Embedding
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            hidden_size=hidden_size,
            embedding_dropout_rate=dropout_rate,
            embedding_pad_token_id=embedding_pad_token_id,
            # Encoder
            num_hidden_layers=num_hidden_layers,
            layer_norm_epsilon=layer_norm_epsilon,
            # Encoder Attn
            num_heads=num_heads,
            attention_module_str=attention_module_str,
            extra_attention_params=extra_attention_params,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            nonlinearity=nonlinearity,
            attention_dropout_rate=attention_dropout_rate,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            # Encoder ffn
            filter_size=filter_size,
            use_ffn_bias=use_ffn_bias,
            # Task-specific
            initializer_range=initializer_range,
            num_segments=num_segments,
            add_pooling_layer=(not self.disable_nsp),
        )

        kernel_initializer = {
            "name": "truncated_normal",
            "std": self.initializer_range,
            "mean": 0.0,
            "a": self.initializer_range * -2.0,
            "b": self.initializer_range * 2.0,
        }

        if not self.disable_nsp:
            self.bert_cls_head = BertClassifierHead(
                hidden_size=hidden_size,
                num_classes=num_classes,
                use_bias=use_ffn_bias,
                kernel_initializer=kernel_initializer,
            )

        self.bert_mlm_head = BertMLMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            use_ffn_bias_in_mlm=use_ffn_bias_in_mlm,
            use_output_bias_in_mlm=use_output_bias_in_mlm,
            activation=mlm_nonlinearity,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=None,
            kernel_initializer=kernel_initializer,
        )

        self.loss_fn = BertPretrainModelLoss(
            disable_nsp=self.disable_nsp,
            mlm_loss_weight=mlm_loss_weight,
            label_smoothing=label_smoothing,
        )

    def reset_parameters(self):
        self.bert_encoder.reset_parameters()
        if not self.disable_nsp:
            self.bert_cls_head.reset_parameters()
        self.bert_mlm_head.reset_parameters()

    def tie_weights(self):
        if not self.share_embedding_weights:
            return

        output_embedding = self.get_output_embeddings()
        input_embedding = self.get_input_embeddings()
        output_embedding.weight = input_embedding.weight

        if getattr(output_embedding, "bias", None) is not None:
            output_embedding.bias.data = nn.functional.pad(
                output_embedding.bias.data,
                (
                    0,
                    output_embedding.weight.shape[0]
                    - output_embedding.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embedding, "out_features") and hasattr(
            input_embedding, "num_embeddings"
        ):
            output_embedding.out_features = input_embedding.num_embeddings

    def get_output_embeddings(self):
        return self.bert_mlm_head.get_output_embeddings()

    def get_input_embeddings(self):
        return self.bert_encoder.embedding_layer.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        next_sentence_label=None,
        masked_lm_weights=None,
        masked_lm_positions=None,
        mlm_loss_scale=None,
        should_calc_loss=True,
    ):
        """
        Args:
            input_ids (Tensor):
                The id of input tokens. Can be of shape ``[batch_size, seq_length]``
            attention_mask (Tensor):
                Can be 2D of shape ``[batch_size, seq_length]``,
                or 3D of shape ``[batch, query_length, seq_length]``,
                or 4D of shape ``[batch, num_heads, query_length, seq_length]``.
            token_type_ids (Tensor):
                The segment id of input tokens, indicating which sequence the token belongs to.
                Can be of shape ```[batch_size, seq_length]`
            labels (Tensor):
                Labels for computing the masked language modeling loss. Shape ``[batch_size, max_predictions_per_seq]``
            next_sentence_label (Tensor):
                Labels for computing the next sentence prediction loss. Shape ``[batch_size, 1]``
            masked_lm_weights (Tensor):
                Weights for mlm logits. Shape ``[batch_size, max_predictions_per_seq]``
            masked_lm_positions (Tensor):
                Position ids of mlm tokens. Shape ``[batch_size, max_predictions_per_seq]``
        """
        attention_mask = make_key_padding_mask_broadcastable(attention_mask)
        mlm_hidden_states, pooled_hidden_states = self.bert_encoder(
            input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        batch_size, seq_len, hidden_size = list(mlm_hidden_states.size())
        _, len_labels = list(labels.size())
        should_gather_mlm_labels = len_labels != seq_len

        focused_mlm_hidden_states = mlm_hidden_states
        if should_gather_mlm_labels:
            batch_size, max_num_pred = list(masked_lm_positions.size())
            index = torch.broadcast_to(
                masked_lm_positions.unsqueeze(-1),
                (batch_size, max_num_pred, hidden_size),
            ).long()
            focused_mlm_hidden_states = torch.gather(
                mlm_hidden_states, dim=1, index=index
            )

        mlm_logits = self.bert_mlm_head(focused_mlm_hidden_states)

        # nsp_logits
        nsp_logits = None
        if not self.disable_nsp:
            nsp_logits = self.bert_cls_head(pooled_hidden_states)
        return mlm_logits, nsp_logits, mlm_hidden_states, pooled_hidden_states
