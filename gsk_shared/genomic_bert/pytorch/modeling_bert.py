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

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from modelzoo.transformers.pytorch.huggingface_common.modeling_bert import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _TOKENIZER_FOR_DOC,
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertEncoder,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


@dataclass
class GenomicBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`GenomicBertForPreTraining`.
    Args:
        loss (`optional`, returned when ``labels_dna`` and `labels_ideas` are provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss for dna nad ideas tasks.
        prediction_logits_dna (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size_dna)`):
            Prediction scores of the dna language modeling head (scores for each vocabulary token before SoftMax).
        prediction_logits_ideas (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size_ideas)`):
            Prediction scores of the ideas language modeling head (scores for each vocabulary token before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits_dna: torch.FloatTensor = None
    prediction_logits_ideas: torch.FloatTensor = None


class GenomicBertForPreTrainingLoss(nn.Module):
    # Adopted for EBERT based on BERT model
    # See: https://github.com/Cerebras/monolith/blob/master/src/models/transformers/pytorch/huggingface_common/modeling_bert.py#L1164
    def __init__(self, mlm_loss_weight=1.0):
        super(GenomicBertForPreTrainingLoss, self).__init__()
        self.mlm_loss_weight = mlm_loss_weight

    def forward(
        self,
        vocab_size_dna,
        vocab_size_ideas,
        prediction_scores_dna,
        prediction_scores_ideas,
        labels_dna,
        labels_ideas,
        masked_lm_weights_dna,
        masked_lm_weights_ideas,
    ):
        # implement the 'batch_size' MLM Scaling mode, which does mean
        # reduction over the batch, but also scales by the mean number
        # of predictions per sequence in the dataset (given by config)
        loss_fct = CrossEntropyLoss(reduction="none")
        masked_lm_loss_dna = loss_fct(
            prediction_scores_dna.view(-1, vocab_size_dna),
            labels_dna.view(-1).long(),
        ) * masked_lm_weights_dna.view(-1)

        masked_lm_loss_ideas = loss_fct(
            prediction_scores_ideas.view(-1, vocab_size_ideas),
            labels_ideas.view(-1).long(),
        ) * masked_lm_weights_ideas.view(-1)

        masked_lm_loss_dna = (
            torch.sum(masked_lm_loss_dna)
            / labels_dna.shape[0]
            * self.mlm_loss_weight
        ).half()

        masked_lm_loss_ideas = (
            torch.sum(masked_lm_loss_ideas)
            / labels_ideas.shape[0]
            * (1.0 - self.mlm_loss_weight)
        ).half()

        total_loss = masked_lm_loss_dna + masked_lm_loss_ideas
        return total_loss


class GenomicBertLMPredictionHead(nn.Module):
    # Adopted for EBERT based on BERT model
    # See: https://github.com/Cerebras/monolith/blob/master/src/models/transformers/pytorch/huggingface_common/modeling_bert.py#L782
    def __init__(self, config):
        super().__init__()
        self.transform_dna = BertPredictionHeadTransform(config)
        self.transform_ideas = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dna = nn.Linear(
            config.hidden_size,
            config.vocab_size_dna,
            bias=config.use_output_bias_in_mlm,
        )
        self.decoder_ideas = nn.Linear(
            config.hidden_size,
            config.vocab_size_ideas,
            bias=config.use_output_bias_in_mlm,
        )

        self.bias_dna = self.decoder_dna.bias
        self.bias_ideas = self.decoder_ideas.bias

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder_dna.bias = self.bias_dna
        self.decoder_ideas.bias = self.bias_ideas

    def forward(self, hidden_states):
        hidden_states_dna = self.transform_dna(hidden_states)
        hidden_states_ideas = self.transform_ideas(hidden_states)
        hidden_states_dna = self.decoder_dna(hidden_states_dna)
        hidden_states_ideas = self.decoder_ideas(hidden_states_ideas)
        return hidden_states_dna, hidden_states_ideas


class GenomicBertPreTrainingHeads(nn.Module):
    # Adopted for EBERT based on BERT model
    # See: https://github.com/Cerebras/monolith/blob/master/src/models/transformers/pytorch/huggingface_common/modeling_bert.py#L824
    def __init__(self, config):
        super().__init__()
        self.predictions = GenomicBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores_dna, prediction_scores_ideas = self.predictions(
            sequence_output
        )
        return prediction_scores_dna, prediction_scores_ideas


class GenomicBertEmbeddings(nn.Module):
    # Adopted for EBERT based on BERT model
    # See: https://github.com/Cerebras/monolith/blob/master/src/models/transformers/pytorch/huggingface_common/modeling_bert.py#L187
    def __init__(self, config):
        super().__init__()
        self.word_embeddings_dna = nn.Embedding(
            config.vocab_size_dna,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.word_embeddings_ideas = nn.Embedding(
            config.vocab_size_ideas,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids_dna=None, input_ids_ideas=None):
        input_shape = input_ids_dna.size()

        seq_length = input_shape[1]

        inputs_embeds_dna = self.word_embeddings_dna(input_ids_dna.long())
        inputs_embeds_ideas = self.word_embeddings_ideas(input_ids_ideas.long())

        embeddings = inputs_embeds_dna + inputs_embeds_ideas

        if self.position_embedding_type == "absolute":
            # Ensure the device can see the constant `torch.arange`.
            position_ids = torch.arange(
                seq_length, device=self.position_ids.device
            ).expand((1, -1))
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@add_start_docstrings(
    "The Bert Model transformer with two embedding layers for dna and ideas outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class GenomicBertModel(BertPreTrainedModel):
    # Adopted for EBERT based on BERT model
    # See: https://github.com/Cerebras/monolith/blob/master/src/models/transformers/pytorch/huggingface_common/modeling_bert.py#L961
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = GenomicBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def get_input_embeddings(self):
        return [
            self.embeddings.word_embeddings_dna,
            self.embeddings.word_embeddings_ideas,
        ]

    def set_input_embeddings(self, value_dna, value_ideas):
        self.embeddings.word_embeddings_dna = value_dna
        self.embeddings.word_embeddings_ideas = value_ideas

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids_dna=None,
        input_ids_ideas=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = (
                use_cache if use_cache is not None else self.config.use_cache
            )
        else:
            use_cache = False

        input_shape = input_ids_dna.size()
        batch_size, seq_length = input_shape

        device = input_ids_dna.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device,
                dtype=self.dtype,
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids_dna=input_ids_dna, input_ids_ideas=input_ids_ideas
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling dna` head and 
    `masked language modeling ideas`.
    """,
    BERT_START_DOCSTRING,
)
class GenomicBertForPreTraining(BertPreTrainedModel):
    # Adopted for EBERT based on BERT model
    # See: https://github.com/Cerebras/monolith/blob/master/src/models/transformers/pytorch/huggingface_common/modeling_bert.py#L1212
    def __init__(self, config, mlm_loss_weight=1.0):
        super().__init__(config)
        self.bert = GenomicBertModel(config)
        self.cls = GenomicBertPreTrainingHeads(config)
        self.loss_fn = GenomicBertForPreTrainingLoss(
            mlm_loss_weight=mlm_loss_weight
        )
        self.init_weights()

    def get_output_embeddings(self):
        return [
            self.cls.predictions.decoder_dna,
            self.cls.predictions.decoder_ideas,
        ]

    def set_output_embeddings(self, new_embeddings_dna, new_embeddings_ideas):
        self.cls.predictions_dna.decoder_dna = new_embeddings_dna
        self.cls.predictions_ideas.decoder_ideas = new_embeddings_ideas

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=GenomicBertForPreTrainingOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids_dna=None,
        input_ids_ideas=None,
        attention_mask=None,
        head_mask=None,
        labels_dna=None,
        labels_ideas=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        masked_lm_weights_dna=None,
        masked_lm_weights_ideas=None,
        masked_lm_positions=None,
    ):
        """
        Returns:
         `GenomicBertForPreTrainingOutput` with loss, prediction_logits_dna and prediction_logits_ideas.
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids_dna=input_ids_dna,
            input_ids_ideas=input_ids_ideas,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        batch_size, length, hidden_size = list(sequence_output.size())
        len_labels = list(labels_dna.size())[1]
        gather_mlm_labels = len_labels != length

        if gather_mlm_labels:
            batch_size, max_pred = list(masked_lm_positions.size())
            index = torch.broadcast_to(
                masked_lm_positions.unsqueeze(2),
                (batch_size, max_pred, hidden_size),
            ).long()
            masked_output = torch.gather(sequence_output, dim=1, index=index,)
            sequence_output = masked_output

        prediction_scores_dna, prediction_scores_ideas = self.cls(
            sequence_output
        )

        total_loss = self.loss_fn(
            self.config.vocab_size_dna,
            self.config.vocab_size_ideas,
            prediction_scores_dna,
            prediction_scores_ideas,
            labels_dna,
            labels_ideas,
            masked_lm_weights_dna,
            masked_lm_weights_ideas,
        )

        if not return_dict:
            output = (
                prediction_scores_dna,
                prediction_scores_ideas,
            ) + outputs[2:]
            return (
                ((total_loss,) + output) if total_loss is not None else output
            )

        return GenomicBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits_dna=prediction_scores_dna,
            prediction_logits_ideas=prediction_scores_ideas,
        )
