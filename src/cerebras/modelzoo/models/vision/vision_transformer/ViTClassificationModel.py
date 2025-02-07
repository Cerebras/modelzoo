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

from typing import List, Literal, Optional

import torch
from annotated_types import Ge, Le
from torch import nn
from typing_extensions import Annotated

from cerebras.modelzoo.models.multimodal.multimodal_utils import freeze_modules
from cerebras.modelzoo.models.nlp.bert.bert_pretrain_models import (
    BertClassifierHead,
)
from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import (
    ViTModel,
    ViTModelConfig,
)


class ViTClassificationModelConfig(ViTModelConfig):
    name: Literal["vit_classification"]

    num_classes: int = 2
    "Number of possible classes."

    # Encoder Attn
    attention_dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.1

    # Task-specific
    use_bias_in_output: bool = True

    use_dinov2_classifier: bool = False

    freeze: Optional[List[str]] = None
    "List of regex strings used to freeze specific layers."


class ViTClassificationModel(nn.Module):
    def __init__(self, config: ViTClassificationModelConfig):
        if isinstance(config, dict):
            config = ViTClassificationModelConfig(**config)

        super().__init__()

        self.vit_model = ViTModel(config)

        classifier_hidden_size = config.hidden_size
        self.use_dinov2_classifier = config.use_dinov2_classifier
        if self.use_dinov2_classifier:
            if not config.prepend_cls_token:
                raise ValueError(
                    f"prepend_cls_token must be set to True for Dinov2."
                )
            classifier_hidden_size *= 2

        self.classifier = BertClassifierHead(
            hidden_size=classifier_hidden_size,
            num_classes=config.num_classes,
            use_bias=config.use_bias_in_output,
            kernel_initializer=config.default_initializer,
        )

        # Freeze specified parameters
        if config.freeze is not None:
            freeze_modules(self, config.freeze)

    def reset_parameters(self):
        self.vit_model.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, input_image):
        hidden_states, pooled_states = self.vit_model(input_image)
        if self.use_dinov2_classifier:
            cls_tokens = hidden_states[:, 0]
            patch_tokens = hidden_states[:, 1:]
            pooled_states = torch.cat(
                [cls_tokens, patch_tokens.mean(dim=1)], dim=1
            )
        logits = self.classifier(pooled_states)
        return logits

    def compute_input_embeddings(self, input_image):
        input_image_embeddings = self.vit_model.embedding_layer(input_image)
        return input_image_embeddings

    def get_input_embeddings(self):
        return self.vit_model.embedding_layer

    def get_output_embeddings(self):
        """
        Extract the final layer that produces logits.
        """
        return self.classifier.classifier

    def tie_weights(self):
        # weights not tied
        pass

    def extract_features(self, input_embeddings, extract_layer_idx):
        """
        Extract features from `extract_layer_idx` of encoder
        by passing input_tensor through encoder
        input_embeddings: Tensor with output from embeddings layer
        extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
                Applies encoder layers up to (and including) `extract_layer_idx`
                instead of all encoder layers.
                For ex: extract_layer_idx=3 would run fwd pass from encoder_block_0 to encoder_block_3
                and return outputs from encoder_block_3.
                If `extract_layer_idx` = None and `norm` != None, then
                the output returned would be encoder_block_{self.num_layers-1} -> norm -> output (return).
        """

        return self.vit_model.extract_features(
            input_embeddings, extract_layer_idx
        )
