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

from typing import Any, Literal, Optional

import torch
from annotated_types import Ge, Le
from pydantic import Field
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.models.nlp.layers_api_demo.cb_transformer import (
    TransformerModel,
)
from cerebras.modelzoo.models.nlp.layers_api_demo.pytorch_transformer import (
    generate_square_subsequent_mask,
)


class TransformerBaseModelConfig(ModelConfig):
    name: Literal["layers_api_demo"]

    vocab_size: int = ...
    embedding_size: int = ...
    hidden_size: int = ...
    num_hidden_layers: int = ...
    dropout: Annotated[float, Ge(0), Le(1)] = None
    nonlinearity: ActivationType = "gelu"
    seq_len: int = 10
    num_heads: int = ...

    mixed_precision: Optional[Any] = Field(default=None, deprecated=True)
    fp16_type: Optional[Any] = Field(default=None, deprecated=True)


class TransformerBaseModel(torch.nn.Module):
    def __init__(self, config: TransformerBaseModelConfig):
        super().__init__()

        self.model = self.build_model(config)

    def build_model(self, config):
        self.ntokens = config.vocab_size
        emsize = config.embedding_size
        d_hid = config.hidden_size
        nlayers = config.num_hidden_layers
        dropout = config.dropout
        activation = config.nonlinearity

        self.seq_len = config.seq_len
        self.num_head = config.num_heads

        model = TransformerModel(
            self.ntokens,
            emsize,
            self.num_head,
            d_hid,
            nlayers,
            dropout,
            activation,
            self.seq_len,
        )

        self.loss_fn = GPTLMHeadModelLoss(
            vocab_size=self.ntokens,
            loss_scaling="batch_size",
            loss_weight=1.0 / self.ntokens,
        )
        return model

    def forward(self, data):
        input_ids = data["input_ids"]
        target_ids = data["target_ids"]
        attention_mask = data["attention_mask"]

        src_mask = generate_square_subsequent_mask(
            input_ids.shape[0],
            self.num_head,
            self.seq_len,
            cstorch.amp.get_half_dtype(),
            input_ids.device,
        )

        """ alternatively, you can use helper functions to create the masks from transformers/pytorch/transformer_utils.py
        # from cerebras.modelzoo.common.utils.model.transformer_utils import (create_2D_autoregressive_mask, NEGATIVE_INFINITY,)
        # src_mask = create_2D_autoregressive_mask(
        #     self.seq_len,
        #     self.seq_len,
        #     device=input_ids.device,
        # ) * NEGATIVE_INFINITY
        """

        output = self.model(input_ids, src_mask)

        loss = self.loss_fn(output, target_ids, attention_mask)

        return loss
