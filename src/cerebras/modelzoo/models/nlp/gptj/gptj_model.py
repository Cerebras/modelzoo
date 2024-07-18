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

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.mup_utils import (
    scale_initializers_by_dimension,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    create_broadcasted_autoregressive_mask,
)
from cerebras.modelzoo.layers import (
    EmbeddingLayer,
    GPTJDecoderLayer,
    TransformerDecoder,
)
from cerebras.modelzoo.layers.norms import get_norm


class GPTJModel(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        # Embedding params
        vocab_size=50257,
        max_position_embeddings=1024,
        embd_pdrop=0.1,
        share_embedding_weights=True,
        position_embedding_type="rotary",
        rotary_dim=None,
        rope_theta=10000,
        pad_rope=False,
        num_relative_attention_buckets=32,
        # Decoder params
        num_hidden_layers=12,
        filter_size=3072,
        dropout_rate=0.1,
        nonlinearity="gelu",
        norm_type="layernorm",
        layer_norm_epsilon=1.0e-5,
        use_ffn_bias=True,
        use_untied_layer_norm=False,
        # Attention params
        num_heads=12,
        attention_module="aiayn_attention",
        attention_sliding_window_length=None,
        extra_attention_params={},
        attention_type="scaled_dot_product",
        attention_dropout_rate=0.1,
        attention_softmax_fp32=True,
        attention_kernel=None,
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        # Task-specific
        initializer_range=0.02,
        use_bias_in_output=False,
        embedding_initializer=None,
        attention_initializer=None,
        output_layer_initializer=None,
        ffn_initializer=None,
        ffn_output_layer_initializer=None,
        # muP (maximal update parameterization)  parameters
        lr_adjustment_groups=None,
        mup_base_hidden_size=None,
        mup_base_filter_size=None,
        embeddings_scale=1.0,
        scale_qk_dot_by_d=False,
        attention_logits_alpha=1.0,
        scale_output_logits_by_d=True,
        output_logits_alpha=None,
        alibi_trainable_slopes=False,
        pos_scaling_factor=1.0,
        moe_params=dict(num_experts=1),
        dtype=None,
    ):
        super(GPTJModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.share_embedding_weights = share_embedding_weights
        self.initializer_range = initializer_range
        self.num_heads = num_heads
        self.moe_enabled = moe_params["num_experts"] > 1
        self.moe_num_experts = moe_params["num_experts"]
        self.moe_routing_algorithm = moe_params.get(
            "routing_algorithm", "learned"
        )

        default_initializer = {
            "name": "truncated_normal",
            "std": self.initializer_range,
            "mean": 0.0,
            "a": self.initializer_range * -2.0,
            "b": self.initializer_range * 2.0,
        }
        if embedding_initializer is None:
            embedding_initializer = default_initializer.copy()
        if attention_initializer is None:
            attention_initializer = default_initializer.copy()
        if output_layer_initializer is None:
            output_layer_initializer = default_initializer.copy()
        if moe_params.get("gate_initializer") is None:
            moe_params["gate_initializer"] = default_initializer
        if ffn_initializer is None:
            ffn_initializer = output_layer_initializer.copy()
        if ffn_output_layer_initializer is None:
            ffn_output_layer_initializer = output_layer_initializer.copy()

        # Handle muP scaling
        self.embeddings_scale = embeddings_scale
        self.output_logits_scale = None
        if mup_base_hidden_size:
            hidden_size_width_mult = hidden_size / mup_base_hidden_size
            scale_initializers_by_dimension(
                [attention_initializer, ffn_initializer],
                width_scale=hidden_size_width_mult**-0.5,
            )
            scale_initializers_by_dimension(
                output_layer_initializer,
                width_scale=hidden_size_width_mult**-0.5,
                depth_scale=(2 * num_hidden_layers) ** -0.5,
            )
            if not output_logits_alpha:
                output_logits_alpha = 1.0
            if scale_output_logits_by_d:
                self.output_logits_scale = (
                    output_logits_alpha / hidden_size_width_mult
                )
            else:
                self.output_logits_scale = (
                    output_logits_alpha / hidden_size_width_mult**0.5
                )
            for lr_adjustment_group in [
                "decoder_attention",
                "decoder_input_ffn",
            ]:
                lr_adjustment_groups[lr_adjustment_group].set_scale(
                    1 / hidden_size_width_mult
                )

        if mup_base_filter_size:
            filter_size_width_mult = filter_size / mup_base_filter_size
            scale_initializers_by_dimension(
                ffn_output_layer_initializer,
                width_scale=filter_size_width_mult**-0.5,
                depth_scale=(2 * num_hidden_layers) ** -0.5,
            )
            lr_adjustment_groups["decoder_output_ffn"].set_scale(
                1 / filter_size_width_mult
            )

        # embedding layer that only contains token embeddings
        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            embeddings_initializer=embedding_initializer,
            position_embedding_type=position_embedding_type,
            position_embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
            num_heads=num_heads,
            num_relative_attention_buckets=num_relative_attention_buckets,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            pad_rope=pad_rope,
            pos_scaling_factor=pos_scaling_factor,
            dtype=dtype,
        )

        self.drop_embd = nn.Dropout(embd_pdrop)

        norm_class = get_norm(norm_type)

        extra_attention_params["attention_kernel"] = attention_kernel
        decoder_layer = GPTJDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            use_untied_layer_norm=use_untied_layer_norm,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_layer=norm_class,
            attention_module=attention_module,
            extra_attention_params=extra_attention_params,
            add_cross_attention=False,
            attention_type=attention_type,
            scale_qk_dot_by_d=scale_qk_dot_by_d,
            attention_logits_alpha=attention_logits_alpha,
            attention_dropout_rate=attention_dropout_rate,
            attention_softmax_fp32=attention_softmax_fp32,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            attention_output_layer_initializer=output_layer_initializer,
            ffn_initializer=ffn_initializer,
            ffn_output_layer_initializer=ffn_output_layer_initializer,
            use_ff_layer1_dropout=False,
            norm_first=True,
            moe_params=moe_params,
        )

        self.ln_f = norm_class(hidden_size, eps=layer_norm_epsilon)

        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_hidden_layers, norm=self.ln_f
        )

        self.attention_sliding_window_length = attention_sliding_window_length

        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=use_bias_in_output
        )

        self.tie_weights()

        self.__reset_parameters()

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.transformer_decoder.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Init final norm layer
        if hasattr(self.ln_f, "bias") and hasattr(self.ln_f.bias, 'data'):
            self.ln_f.bias.data.zero_()
        if hasattr(self.ln_f, "weight") and hasattr(self.ln_f.weight, 'data'):
            self.ln_f.weight.data.fill_(1.0)

        # Initialize LM head
        if not self.share_embedding_weights:
            self.lm_head.weight.data.normal_(
                mean=0.0, std=self.initializer_range
            )
        if self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()

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

    def get_input_embeddings(self):
        return self.embedding_layer.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def compute_input_embeddings(self, input_ids, position_ids=None):
        hidden_states = self.embedding_layer(
            input_ids, position_ids=position_ids
        )
        hidden_states = hidden_states * torch.tensor(
            float(self.embeddings_scale), dtype=hidden_states.dtype
        )
        hidden_states = self.drop_embd(hidden_states)
        return hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        attention_span=None,
        tgt_key_padding_mask=None,
        position_ids=None,
        input_embeddings=None,
        inference_loop_index=None,
    ):
        if input_ids is not None and input_embeddings is not None:
            raise ValueError(
                f"Only one of `input_ids` or `input_embeddings` "
                f"should be passed to model.forward"
            )
        elif input_ids is None and input_embeddings is None:
            raise ValueError(
                f"Both `input_ids` and `input_embeddings` are None, "
                f"either one of them should be passed to model.forward"
            )

        if input_embeddings is None:
            hidden_states = self.compute_input_embeddings(
                input_ids, position_ids
            )
        else:
            hidden_states = input_embeddings

        expert_hash_idx = None
        if self.moe_enabled and self.moe_routing_algorithm == "hash":
            expert_hash_idx = input_ids.to(torch.float) % self.moe_num_experts
            expert_hash_idx = expert_hash_idx.to(input_ids.dtype)

        decoder_outputs = self.apply_decoder(
            hidden_states,
            attention_mask=attention_mask,
            attention_span=attention_span,
            position_ids=position_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
            expert_hash_idx=expert_hash_idx,
        )

        if self.moe_enabled:
            hidden_states, routing_weights, expert_masks = decoder_outputs
        else:
            hidden_states = decoder_outputs

        if inference_loop_index is not None:
            # When running an implicit autoregressive loop for generation, this
            # tensor holds the "current token" index. We can pull out only the
            # hidden states from that token to avoid unnecessary work in the
            # lm_head matmul.
            hidden_states = cstorch.experimental.get_loop_iteration_slice(
                hidden_states, inference_loop_index
            )

        if (
            cstorch.use_cs()
            and cstorch.backends.csx.precision.optimization_level == 1
        ):
            lm_logits = cstorch.pol(bwd_level=0)(self.lm_head)(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        # scale lm_logits for muP transfer
        if self.output_logits_scale:
            lm_logits = lm_logits * torch.tensor(
                float(self.output_logits_scale),
                dtype=lm_logits.dtype,
            )

        if self.moe_enabled:
            return lm_logits, routing_weights, expert_masks
        else:
            return lm_logits

    def apply_decoder(
        self,
        input_embeddings,
        attention_mask=None,
        attention_span=None,
        tgt_key_padding_mask=None,
        position_ids=None,
        extract_layer_idx=None,
        expert_hash_idx=None,
    ):
        # `extract_layer_idx` is used only multimodal use case
        # input_embeddings : shape (bsz, MSL, H)
        causal_attention_mask = create_broadcasted_autoregressive_mask(
            batch_size=input_embeddings.shape[0],
            num_heads=self.num_heads,
            tgt_seq_length=input_embeddings.shape[1],
            attention_span=attention_span,
            sliding_window_length=self.attention_sliding_window_length,
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )

        # Helpers on alibi/relative position embeddings bias
        length = input_embeddings.shape[1]
        self_attn_position_bias = self.embedding_layer.compute_position_bias(
            length, length
        )

        return self.transformer_decoder(
            input_embeddings,
            tgt_mask=causal_attention_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            rotary_position_embedding_helper=self.embedding_layer.get_rope_helper(),
            self_attn_position_bias=self_attn_position_bias,
            extract_layer_idx=extract_layer_idx,
            expert_hash_idx=expert_hash_idx,
        )

    def extract_features(
        self,
        input_embeddings,
        extract_layer_idx,
        attention_mask=None,
        attention_span=None,
        tgt_key_padding_mask=None,
        position_ids=None,
    ):
        """
        Extract features of input_embeddings from `extract_layer_idx` of decoder
        extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
            Applies decoder layers up to (and including) `extract_layer_idx`
            instead of all decoder layers.
            For ex: extract_layer_idx=3 would run fwd pass from decoder_block_0 to decoder_block_3
            and return outputs from decoder_block_3.
            If `extract_layer_idx` = None and `norm` != None, then
            the output returned would be decoder_block_{self.num_layers-1} -> norm -> output (return)

        This function is added for multimodal use case.
        """
        hidden_states = self.apply_decoder(
            input_embeddings,
            extract_layer_idx=extract_layer_idx,
            attention_mask=attention_mask,
            attention_span=attention_span,
            tgt_key_padding_mask=tgt_key_padding_mask,
            position_ids=position_ids,
        )

        if isinstance(hidden_states, tuple):
            return hidden_states[0]
        else:
            return hidden_states
