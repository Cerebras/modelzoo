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

from functools import partial
from types import MethodType

import torch

from cerebras.modelzoo.models.nlp.dpr.embedding_generation.saver import (
    DPREmbeddingSaver,
)
from cerebras.modelzoo.models.nlp.dpr.model import DPRWrapperModel
from cerebras.pytorch.utils.step_closures import step_closure


def custom_load_state_dict(
    self, state_dict, encoder_key, strict=True, assign=False
):
    if strict:
        unexpected_keys = list(
            filter(
                lambda k: k.find("q_encoder") == -1
                and k.find("ctx_encoder") == -1,
                state_dict.keys(),
            ),
        )
        if len(unexpected_keys) > 0:
            raise RuntimeError(
                f'Unexpected key(s) in state_dict: {unexpected_keys}.'
            )
    encoder_state_dict = {
        k.replace(encoder_key + ".", ""): v
        for k, v in state_dict.items()
        if k.find(encoder_key) >= 0
    }
    return torch.nn.Module.load_state_dict(
        self, encoder_state_dict, strict=strict, assign=assign
    )


class DPRWrapperModelForEmbeddingGeneration(DPRWrapperModel):
    def __init__(
        self, embedding_saver: DPREmbeddingSaver, selected_encoder, params
    ):
        self.embedding_saver = embedding_saver
        self.selected_encoder = selected_encoder
        super().__init__(params)
        self.loss_fn = None

    def build_model(self, model_params):
        q_encoder_params = model_params.pop("q_encoder")
        ctx_encoder_params = model_params.pop("ctx_encoder")

        if self.selected_encoder == "q_encoder":
            encoder = self.build_bert(q_encoder_params)
        elif self.selected_encoder == "ctx_encoder":
            encoder = self.build_bert(ctx_encoder_params)
        else:
            raise RuntimeError(
                "The 'selected_encoder' model parameter should either be "
                "'q_encoder' or 'ctx_encoder' for embedding generation."
            )

        encoder_key = (
            "q_encoder"
            if self.selected_encoder == "q_encoder"
            else self.selected_encoder
        )

        custom_load_state_dict_fn = partial(
            custom_load_state_dict, encoder_key=encoder_key
        )
        # We apply the custom function twice to handle checkpoints with
        # and without model. prefix, as they're handled differently in
        # run_cstorch_flow
        self.load_state_dict = MethodType(custom_load_state_dict_fn, self)
        encoder.load_state_dict = MethodType(custom_load_state_dict_fn, encoder)

        return encoder

    def __call__(self, data):
        model_forward_args = {
            key: data.pop(key, None)
            for key in [
                "input_ids",
                "position_ids",
                "segment_ids",
                "attention_mask",
            ]
        }

        hidden_states, pooled_output = self.model(**model_forward_args)

        if not self.model.pooler:
            pooled_output = hidden_states[:, 0]

        self.save_embeddings(pooled_output, data.pop("id"))

        return torch.tensor([0])

    @step_closure
    def save_embeddings(self, embeddings, ids):
        self.embedding_saver.add_embeddings(
            embeddings.cpu().detach().numpy(), ids.cpu().detach().numpy()
        )
