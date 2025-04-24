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

from typing import Literal, Optional

import torch
import torch.nn as nn

from cerebras.modelzoo.layers import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
)
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.layers.init import TruncatedNormalInitializer
from cerebras.modelzoo.models.nlp.bert.bert_model import (
    BertModel,
    BertModelConfig,
)


class BertForPreTrainingModelConfig(BertModelConfig):
    name: Literal["bert"]

    share_embedding_weights: bool = True
    "Whether to share the embedding weights between the input and output embedding."

    num_segments: Optional[int] = None
    """Number of segments (token types) in embedding. When not specified
    (and NSP objective is enabled), num_segments will default to 2"""

    pad_token_id: int = 0
    "The embedding vector at pad_token_id is not updated during training."

    alibi_trainable_slopes: bool = False
    "Replaces alibi's fixed slopes with trainable slopes."

    encoder_nonlinearity: ActivationType = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block.
    See list of non-linearity functions [here](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
    """

    mlm_nonlinearity: Optional[ActivationType] = None
    """The non-linear activation function used in the MLM head. If not
    specified, defaults to encoder_nonlinearity."""

    use_ffn_bias_in_mlm: bool = True
    "Whether to use bias in MLM head's FFN layer."

    use_output_bias_in_mlm: bool = True
    "Whether to use bias in MLM head's output (classifier) layer."

    # Loss:
    mlm_loss_weight: float = 1.0
    "Value that scales the Masked Language Modelling (MLM) loss."

    label_smoothing: float = 0.0
    "The label smoothing factor used during training."

    # Task-specific:
    disable_nsp: bool = False
    "Disables Next Sentence Prediction (NSP) objective."

    num_classes: int = 2
    "Number of classes used by the classifier head (NSP)."

    # Misc:
    compute_eval_metrics: bool = True
    "Computes perplexity & accuracy metrics in addition to loss."

    def post_init(self, context):
        super().post_init(context)

        if self.num_segments is None:
            self.num_segments = None if self.disable_nsp else 2

        self.nonlinearity = self.encoder_nonlinearity
        self.embedding_dropout_rate = self.dropout_rate
        self.embedding_pad_token_id = self.pad_token_id
        self.add_pooling_layer = not self.disable_nsp

        if self.mlm_nonlinearity is None:
            self.mlm_nonlinearity = self.nonlinearity


class BertPretrainModel(nn.Module):
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head. Following the paper: https://arxiv.org/abs/1810.04805.
    """

    def __init__(self, config: BertForPreTrainingModelConfig):
        """
        Args:
            config: Settings for the model.
        """

        super().__init__()

        self.disable_nsp = config.disable_nsp
        self.share_embedding_weights = config.share_embedding_weights

        self.bert_encoder = self.build_encoder_model(config)

        # Handle muP scaling
        self.output_logits_scale = None
        if config.mup_base_hidden_size:
            hidden_size_width_mult = (
                config.hidden_size / config.mup_base_hidden_size
            )
            if config.scale_output_logits_by_d:
                self.output_logits_scale = (
                    config.output_logits_alpha / hidden_size_width_mult
                )
            else:
                self.output_logits_scale = (
                    config.output_logits_alpha / hidden_size_width_mult**0.5
                )

        if not self.disable_nsp:
            self.bert_cls_head = BertClassifierHead(
                hidden_size=config.hidden_size,
                num_classes=config.num_classes,
                use_bias=config.use_ffn_bias,
                kernel_initializer=TruncatedNormalInitializer(
                    std=config.initializer_range,
                    mean=0.0,
                    a=config.initializer_range * -2.0,
                    b=config.initializer_range * 2.0,
                ),
            )

        self.bert_mlm_head = self.build_mlm_head(config)

        self.tie_weights()

    def get_lr_adjustment_groups(self):
        return self.bert_encoder.get_lr_adjustment_groups()

    def build_encoder_model(self, config: BertForPreTrainingModelConfig):
        return BertModel(config)

    def build_mlm_head(self, config: BertForPreTrainingModelConfig):
        return BertMLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            use_ffn_bias_in_mlm=config.use_ffn_bias_in_mlm,
            use_output_bias_in_mlm=config.use_output_bias_in_mlm,
            activation=config.mlm_nonlinearity,
            layer_norm_epsilon=config.layer_norm_epsilon,
            dropout=None,
            kernel_initializer=TruncatedNormalInitializer(
                std=config.initializer_range,
                mean=0.0,
                a=config.initializer_range * -2.0,
                b=config.initializer_range * 2.0,
            ),
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
        position_ids=None,
        token_type_ids=None,
        masked_lm_positions=None,
        should_gather_mlm_labels=False,
        attention_span=None,
    ):
        """
        Args:
            input_ids (Tensor):
                The id of input tokens. Can be of shape ``[batch_size, seq_length]``
            attention_mask (Tensor):
                Can be 2D of shape ``[batch_size, seq_length]``,
                or 3D of shape ``[batch, query_length, seq_length]``,
                or 4D of shape ``[batch, num_heads, query_length, seq_length]``.
            position_ids (Tensor):
                The position id of input tokens. Can be of shape ``[batch_size, seq_length]``
            token_type_ids (Tensor):
                The segment id of input tokens, indicating which sequence the token belongs to.
                Can be of shape ```[batch_size, seq_length]`
            masked_lm_positions (Tensor):
                Position ids of mlm tokens. Shape ``[batch_size, max_predictions_per_seq]``
            attention_span (Tensor):
                The attention span of input tokens for creating VSL mask. Can be of shape ```[batch_size, seq_length]```.
        """
        mlm_hidden_states, pooled_hidden_states = self.bert_encoder(
            input_ids,
            position_ids=position_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
            attention_span=attention_span,
        )
        batch_size, seq_len, hidden_size = list(mlm_hidden_states.size())

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

        # scale mlm_logits for muP transfer
        if self.output_logits_scale:
            mlm_logits = mlm_logits * torch.tensor(
                float(self.output_logits_scale),
                dtype=mlm_logits.dtype,
            )

        # nsp_logits
        nsp_logits = None
        if not self.disable_nsp:
            nsp_logits = self.bert_cls_head(pooled_hidden_states)
            # scale nsp_logits for muP transfer
            if self.output_logits_scale:
                nsp_logits = nsp_logits * torch.tensor(
                    float(self.output_logits_scale),
                    dtype=nsp_logits.dtype,
                )

        return mlm_logits, nsp_logits, mlm_hidden_states, pooled_hidden_states


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
            FeedForwardNetworkConfig(
                input_unit=hidden_size,
                layers_units=[num_classes],
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
            )
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
            FeedForwardNetworkConfig(
                input_unit=hidden_size,
                layers_units=[embedding_size],
                layers_activation=[activation],
                layers_dropout_rates=[dropout],
                use_bias=use_ffn_bias_in_mlm,
                kernel_initializer=kernel_initializer,
            )
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
            FeedForwardNetworkConfig(
                input_unit=embedding_size,
                layers_units=[vocab_size],
                use_bias=use_output_bias_in_mlm,
                kernel_initializer=kernel_initializer,
            )
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
