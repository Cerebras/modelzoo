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

import math
from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseConfigConverter,
    BaseDictionaryConverter,
    ConversionRule,
    FormatVersions,
)


class ConfigConverter_sP_muP(BaseConfigConverter):
    """Transforms a CS muP config to a CS sP config."""

    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(["output_logits_scale"]),
            ConversionRule(["embeddings_scale"]),
            ConversionRule(["scale_qk_dot_by_d"]),
            ConversionRule(
                ["share_embedding_weights"],
                action=self.set_share_embedding_weights,
            ),
            ConversionRule(
                [r".*"], action=self.replaceKey
            ),  # Catch-all for everything else
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return ("sP", "muP")

    @staticmethod
    def file_formats() -> Tuple[str, str]:
        return ()

    @staticmethod
    def is_mup(config):
        return _is_mup(config)

    def set_share_embedding_weights(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 1 and (
            "output_logits_scale" in old_state_dict
            or "embeddings_scale" in old_state_dict
        ):
            new_state_dict[new_key] = False
        else:
            new_state_dict[new_key] = old_state_dict[old_key]


class Converter_sP_muP(BaseDictionaryConverter):
    """Transforms a CS muP checkpoints into a CS sP checkpoint.

    muP: Maximal Update Parametrization.
    sP: Standard Parametrization.
    """

    def __init__(self):
        super().__init__()

        self.rules = [
            ConversionRule(
                [r".+\.proj_k_dense_layer.*"],
                action=self.scale_k_projection,
            ),
            ConversionRule(
                [r"(?:model\.|)lm_head\.weight"],
                action=self.scale_lm_head,
            ),
            ConversionRule(
                [r"(?:model\.|)embedding_layer\.word_embeddings\.weight"],
                action=self.scale_embeddings,
            ),
            ConversionRule(
                [
                    r"(?:model\.|)embedding_layer\.position_embeddings(?:\.embed)?\.weight"
                ],
                action=self.scale_embeddings,
            ),
            ConversionRule(
                [r"(?:model\.|)embedding_ln_f\.(?:weight|bias)"],
                action=self.scale_embedding_layernorm,
            ),
            ConversionRule(
                [r".*"], action=self.replaceKey
            ),  # Catch-all for everything else
        ]

    def scale_k_projection(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]

        if config["model"].get('scale_qk_dot_by_d', False):
            d_model = config["model"]["hidden_size"]
            n_heads = config["model"]["num_heads"]
            d_sqrt = math.sqrt(d_model // n_heads)

            new_state_dict[new_key] = old_state_dict[old_key] / d_sqrt
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    def scale_lm_head(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]

        if "output_logits_scale" in config["model"]:
            output_scale = config["model"]["output_logits_scale"]

            new_state_dict[new_key] = old_state_dict[old_key] * output_scale
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    def scale_embeddings(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]

        # Fold embeddings_scale into word/position embeddings if embedding
        # layer norm *is not* enabled
        if "embeddings_scale" in config["model"] and not config["model"].get(
            "embedding_layer_norm", False
        ):
            emb_scale = config["model"]["embeddings_scale"]

            new_state_dict[new_key] = old_state_dict[old_key] * emb_scale
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    def scale_embedding_layernorm(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]

        # Fold embeddings_scale into embedding layer norm if embedding
        # layer norm *is* enabled
        if "embeddings_scale" in config["model"] and config["model"].get(
            "embedding_layer_norm", False
        ):
            emb_scale = config["model"]["embeddings_scale"]

            new_state_dict[new_key] = old_state_dict[old_key] * emb_scale
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    @staticmethod
    def is_mup(config):
        return _is_mup(config.get('model', {}))

    @staticmethod
    def formats():
        return ("sP", "muP")


def _is_mup(model_config):
    scale_qk_dot_by_d = model_config.get('scale_qk_dot_by_d', False)
    embeddings_scale = model_config.get('embeddings_scale', None)
    output_logits_scale = model_config.get('output_logits_scale', None)

    all_set = scale_qk_dot_by_d and embeddings_scale and output_logits_scale
    any_set = scale_qk_dot_by_d or embeddings_scale or output_logits_scale

    if any_set and not all_set:
        raise ValueError(
            "This looks like an incomplete muP config. Either all of or none of "
            "\"scale_qk_dot_by_d\", \"embeddings_scale\", \"output_logits_scale\" can be "
            "specified, but this config only has some that are specified."
        )

    return all_set
