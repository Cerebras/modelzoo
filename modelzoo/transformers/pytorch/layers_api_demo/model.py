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

from modelzoo.common.pytorch.model_utils.GPTLMHeadModelLoss import (
    GPTLMHeadModelLoss,
)
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.layers_api_demo.cb_transformer import (
    TransformerModel,
)
from modelzoo.transformers.pytorch.layers_api_demo.pytorch_transformer import (
    generate_square_subsequent_mask,
)


class TransformerBaseModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()

        self.model = self.build_model(model_params)

        super(TransformerBaseModel, self).__init__(
            params=params, model=self.model, device=device
        )

    def build_model(self, model_params):
        self.ntokens = model_params.pop("vocab_size")
        emsize = model_params.pop("embedding_size")
        nhead = model_params.pop("num_heads")
        d_hid = model_params.pop("hidden_size")
        nlayers = model_params.pop("num_hidden_layers")
        dropout = model_params.pop("dropout")
        activation = model_params.pop("nonlinearity")

        self.seq_len = model_params.pop("seq_len")

        model = TransformerModel(
            self.ntokens,
            emsize,
            nhead,
            d_hid,
            nlayers,
            dropout,
            activation,
            self.seq_len,
        )

        self.loss_fn = GPTLMHeadModelLoss(self.ntokens, 1.0 / self.ntokens,)
        return model

    def __call__(self, data):
        input_ids = data["input_ids"]
        target_ids = data["target_ids"]
        attention_mask = data["attention_mask"]

        src_mask = generate_square_subsequent_mask(
            self.seq_len, input_ids.device
        )

        """ alternatively, you can use helper functions to create the masks from transformers/pytorch/transformer_utils.py
        # from modelzoo.transformers.pytorch.transformer_utils import (create_2D_autoregressive_mask,)
        # src_mask = create_2D_autoregressive_mask(
        #     self.seq_len,
        #     self.seq_len,
        #     device=input_ids.device,
        # ) * -1.0e4
        """

        output = self.model(input_ids, src_mask)

        loss = self.loss_fn(output, target_ids, attention_mask)

        return loss
