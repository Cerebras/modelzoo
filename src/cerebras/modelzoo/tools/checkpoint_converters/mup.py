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
    EquivalentSubkey,
    FormatVersions,
)


def scale_initializers_by_dimension(
    initializers,
    width_scale=None,
    depth_scale=None,
):
    if not width_scale:
        width_scale = 1.0
    if not depth_scale:
        depth_scale = 1.0
    mup_scalar = width_scale * depth_scale

    if not isinstance(initializers, list):
        initializers = [initializers]

    for initializer in initializers:
        if type(initializer) == str:
            initializer = {"name": initializer}
        if "name" not in initializer:
            raise ValueError("Initializer name must be provided")
        initializer_name = initializer["name"].lower()

        if initializer_name == "normal":
            initializer["std"] = initializer.get("std", 1.0) * mup_scalar
        elif initializer_name == "truncated_normal":
            std = initializer.get("std", 1.0)
            initializer["std"] = std * mup_scalar
            initializer["a"] = initializer.get("a", -2 * std) * mup_scalar
            initializer["b"] = initializer.get("b", 2 * std) * mup_scalar
            std = None


class ConfigConverter_sP_muP_pre_CS23(BaseConfigConverter):
    """Transforms a CS 2.2 and before muP config to a CS sP config."""

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
        return _is_mup_CS22(config)

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


class ConfigConverter_sP_muP_post_CS23(ConfigConverter_sP_muP_pre_CS23):
    """Transforms a CS 2.3 and onwards muP config to a CS sP config."""

    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(["mup_base_hidden_size"]),
            ConversionRule(["mup_base_filter_size"]),
            ConversionRule(["attention_logits_alpha"]),
            ConversionRule(["output_logits_alpha"]),
            ConversionRule(["scale_output_logits_by_d"]),
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
            _is_mup(old_state_dict) or 'embeddings_scale' in old_state_dict
        ):
            new_state_dict[new_key] = False
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    @staticmethod
    def is_mup(config):
        return _is_mup(config)


class ConfigConverter_T5_sP_muP(ConfigConverter_sP_muP_post_CS23):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(["mup_base_d_model"]),
            ConversionRule(["mup_base_d_ff"]),
            ConversionRule(["mup_base_d_kv"]),
            ConversionRule(["encoder_attention_logits_alpha"]),
            ConversionRule(["decoder_attention_logits_alpha"]),
            ConversionRule(["output_logits_alpha"]),
            ConversionRule(["scale_output_logits_by_d"]),
            ConversionRule(["embeddings_alpha"]),
            ConversionRule(["scale_encoder_qk_dot_by_d"]),
            ConversionRule(["scale_decoder_qk_dot_by_d"]),
            ConversionRule(
                ["share_embedding_weights"],
                action=self.set_share_embedding_weights,
            ),
            ConversionRule(
                [r".*"], action=self.replaceKey
            ),  # Catch-all for everything else
        ]

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
            _is_mup(old_state_dict) or 'embeddings_alpha' in old_state_dict
        ):
            new_state_dict[new_key] = False
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    @staticmethod
    def is_mup(config):
        return _is_mup(config)


class ConfigConverter_muP_CS22_CS23(BaseConfigConverter):
    """Transforms a CS 2.2 muP config to a CS 2.3 muP config with the reworked
    params.
    """

    def __init__(self):
        super().__init__()
        self.hidden_size_width_mult = None
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("initializer", "initializer"),
                ],
                action=self.unscale_input_initializer,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "output_layer_initializer", "output_layer_initializer"
                    ),
                ],
                action=self.unscale_output_initializer,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "output_logits_scale", "output_logits_alpha"
                    ),
                ],
                action=self.replace_output_logits_scale,
            ),
            ConversionRule(
                [r".*"], action=self.replaceKey
            ),  # Catch-all for everything else
        ]

    @classmethod
    def convert(
        cls,
        model,
        config,
        converter_indices,
        drop_unmatched_keys=False,
        no_progress_bar=True,
        debug=False,
    ):
        instance = cls()
        lr_scale = config["optimizer"].pop("adjust_learning_rate")
        instance.hidden_size_width_mult = 1 / lr_scale["decoder_kernel"]
        return instance.convert_helper(
            model,
            config["model"],
            converter_indices,
            drop_unmatched_keys=drop_unmatched_keys,
            no_progress_bar=no_progress_bar,
            debug=debug,
        )

    def unscale_input_initializer(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # self.hidden_size_width_mult = hidden_size / mup_base_hidden_size
        # input_layer = input_layer' / sqrt(self.hidden_size_width_mult)
        initializer_config = old_state_dict[old_key]
        scale_initializers_by_dimension(
            initializer_config,
            width_scale=self.hidden_size_width_mult**0.5,
        )

        new_state_dict[new_key] = initializer_config

    def unscale_output_initializer(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # self.hidden_size_width_mult = hidden_size / mup_base_hidden_size
        num_hidden_layers = old_state_dict["num_hidden_layers"]
        # output_layer = output_layer' / sqrt(2 * num_hidden_layers * self.hidden_size_width_mult)
        initializer_config = old_state_dict[old_key]
        scale_initializers_by_dimension(
            initializer_config,
            width_scale=self.hidden_size_width_mult**0.5,
            depth_scale=(2 * num_hidden_layers) ** 0.5,
        )

        new_state_dict[new_key] = initializer_config

    def replace_output_logits_scale(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        hidden_size = old_state_dict["hidden_size"]
        filter_size = old_state_dict["filter_size"]
        # self.hidden_size_width_mult = hidden_size / mup_base_hidden_size
        mup_base_hidden_size = hidden_size / self.hidden_size_width_mult
        mup_base_filter_size = filter_size / self.hidden_size_width_mult

        # output_logits_scale = output_logits_alpha / self.hidden_size_width_mult
        output_logits_alpha = (
            old_state_dict[old_key] * self.hidden_size_width_mult
        )

        new_state_dict["mup_base_hidden_size"] = mup_base_hidden_size
        new_state_dict["mup_base_filter_size"] = mup_base_filter_size
        new_state_dict[new_key] = output_logits_alpha

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return ("cs-2.2", "cs-2.3")

    @staticmethod
    def file_formats() -> Tuple[str, str]:
        return ()

    @staticmethod
    def is_mup(config):
        return _is_mup_CS22(config)


class Converter_sP_muP_pre_CS23(BaseDictionaryConverter):
    """Transforms a CS 2.2 and older muP checkpoints into a CS sP checkpoint.

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
        return _is_mup_CS22(config.get('model', {}))

    @staticmethod
    def formats():
        return ("sP", "muP")


class Converter_sP_muP_post_CS23(Converter_sP_muP_pre_CS23):
    """Transforms a CS 2.3 and onwards muP checkpoint into a CS sP checkpoint.

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
                [r"(?:model\.|)mlm_head\.weight"],
                action=self.scale_lm_head,
            ),
            ConversionRule(
                [r"(?:model\.|)cls_head\.weight"],
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

        scale_qk_dot_by_d = config["model"].get("scale_qk_dot_by_d", None)
        if scale_qk_dot_by_d is None:
            scale_qk_dot_by_d = _is_mup(config["model"])

        attention_logits_alpha = config["model"].get(
            "attention_logits_alpha", None
        )
        if attention_logits_alpha is None:
            attention_logits_alpha = 1.0

        if scale_qk_dot_by_d:
            d_model = config["model"]["hidden_size"]
            n_heads = config["model"]["num_heads"]
            d_sqrt = math.sqrt(d_model // n_heads)

            new_state_dict[new_key] = (
                attention_logits_alpha * old_state_dict[old_key] / d_sqrt
            )
        else:
            new_state_dict[new_key] = (
                attention_logits_alpha * old_state_dict[old_key]
            )

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

        mup_base_hidden_size = config["model"].get("mup_base_hidden_size", None)

        if mup_base_hidden_size:
            output_logits_alpha = config["model"].get(
                "output_logits_alpha", None
            )
            if output_logits_alpha is None:
                output_logits_alpha = 1.0

            scale_output_logits_by_d = config["model"].get(
                "scale_output_logits_by_d", None
            )
            if scale_output_logits_by_d is None:
                scale_output_logits_by_d = True

            hidden_size = config["model"]["hidden_size"]
            hidden_size_width_mult = hidden_size / mup_base_hidden_size

            if scale_output_logits_by_d:
                output_logits_scale = (
                    output_logits_alpha / hidden_size_width_mult
                )
            else:
                output_logits_scale = (
                    output_logits_alpha / hidden_size_width_mult**0.5
                )

            new_state_dict[new_key] = (
                old_state_dict[old_key] * output_logits_scale
            )
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    @staticmethod
    def is_mup(config):
        return _is_mup(config["model"])


class Converter_T5_sP_muP(BaseDictionaryConverter):
    """Transforms a T5 CS muP checkpoint into a T5 CS sP checkpoint.

    muP: Maximal Update Parametrization.
    sP: Standard Parametrization.
    """

    def __init__(self):
        super().__init__()

        self.rules = [
            ConversionRule(
                [r".+\.proj_q_dense_layer.*"],
                action=self.scale_q_projection,
            ),
            ConversionRule(
                [r".*encoder.*proj_k_dense_layer.*"],
                action=self.scale_encoder_k_projection,
            ),
            ConversionRule(
                [r".*decoder.*proj_k_dense_layer.*"],
                action=self.scale_decoder_k_projection,
            ),
            ConversionRule(
                [r".+\.proj_v_dense_layer.*"],
                action=self.scale_v_projection,
            ),
            ConversionRule(
                [r".+\.proj_output_dense_layer.*"],
                action=self.scale_output_projection,
            ),
            ConversionRule(
                [r"(?:model\.|)lm_head\.weight"],
                action=self.scale_lm_head,
            ),
            ConversionRule(
                [r"(?:model\.|).*embeddings\.word_embeddings\.weight"],
                action=self.scale_embeddings,
            ),
            ConversionRule(
                [r".*"], action=self.replaceKey
            ),  # Catch-all for everything else
        ]

    def scale_q_projection(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]
        d_model = config["model"]["d_model"]

        mup_base_d_model = config["model"].get("mup_base_d_model", None)
        if mup_base_d_model is None:
            mup_base_d_model = d_model

        d_model_width_mult = d_model / mup_base_d_model
        d_sqrt = math.sqrt(config["model"]["d_kv"])
        projection_scale = d_model_width_mult**-0.5
        if config["model"].get("mup_base_d_kv", None):
            projection_scale = 1.0
        total_scale = projection_scale / d_sqrt
        new_state_dict[new_key] = old_state_dict[old_key] * total_scale

    def scale_encoder_k_projection(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]

        scale_qk_dot_by_d = config["model"].get(
            "scale_encoder_qk_dot_by_d", None
        )
        if scale_qk_dot_by_d is None:
            scale_qk_dot_by_d = _is_mup(config["model"])

        d_model = config["model"]["d_model"]

        mup_base_d_model = config["model"].get("mup_base_d_model", None)
        if mup_base_d_model is None:
            mup_base_d_model = d_model

        d_model_width_mult = d_model / mup_base_d_model

        attention_logits_alpha = config["model"].get(
            "encoder_attention_logits_alpha"
        )
        if attention_logits_alpha is None:
            attention_logits_alpha = 1.0

        projection_scale = d_model_width_mult**-0.5

        if config["model"].get("mup_base_d_kv", None):
            projection_scale = 1.0

        if scale_qk_dot_by_d:
            d_sqrt = math.sqrt(config["model"]["d_kv"])
            total_scale = attention_logits_alpha * projection_scale / d_sqrt
        else:
            total_scale = attention_logits_alpha * projection_scale

        new_state_dict[new_key] = old_state_dict[old_key] * total_scale

    def scale_decoder_k_projection(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]

        scale_qk_dot_by_d = config["model"].get(
            "scale_decoder_qk_dot_by_d", None
        )
        if scale_qk_dot_by_d is None:
            scale_qk_dot_by_d = _is_mup(config["model"])

        d_model = config["model"]["d_model"]

        mup_base_d_model = config["model"].get("mup_base_d_model", None)
        if mup_base_d_model is None:
            mup_base_d_model = d_model

        d_model_width_mult = d_model / mup_base_d_model

        attention_logits_alpha = config["model"].get(
            "decoder_attention_logits_alpha"
        )
        if attention_logits_alpha is None:
            attention_logits_alpha = 1.0

        projection_scale = d_model_width_mult**-0.5

        if config["model"].get("mup_base_d_kv", None):
            projection_scale = 1.0

        if scale_qk_dot_by_d:
            d_sqrt = math.sqrt(config["model"]["d_kv"])
            total_scale = attention_logits_alpha * projection_scale / d_sqrt
        else:
            total_scale = attention_logits_alpha * projection_scale

        new_state_dict[new_key] = old_state_dict[old_key] * total_scale

    def scale_v_projection(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]
        d_model = config["model"]["d_model"]

        mup_base_d_model = config["model"].get("mup_base_d_model", None)
        if mup_base_d_model is None:
            mup_base_d_model = d_model

        d_model_width_mult = d_model / mup_base_d_model
        projection_scale = d_model_width_mult**-0.5
        if config["model"].get("mup_base_d_kv", None):
            projection_scale = 1.0
        new_state_dict[new_key] = projection_scale * old_state_dict[old_key]

    def scale_output_projection(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        config = action_fn_args["configs"][1]
        d_model = config["model"]["d_model"]

        mup_base_d_model = config["model"].get("mup_base_d_model", None)
        if mup_base_d_model is None:
            mup_base_d_model = d_model

        d_model_width_mult = d_model / mup_base_d_model
        projection_scale = d_model_width_mult**0.5
        if config["model"].get("mup_base_d_kv", None):
            projection_scale = 1.0
        new_state_dict[new_key] = projection_scale * old_state_dict[old_key]

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
        if _is_mup(config["model"]):
            emb_alpha = config["model"].get("embeddings_alpha", None)
            d_model = config["model"]["d_model"]
            if not emb_alpha:
                emb_alpha = 1.0
            emb_scale = emb_alpha * d_model**0.5

            new_state_dict[new_key] = old_state_dict[old_key] * emb_scale

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

        mup_base_d_model = config["model"].get("mup_base_d_model", None)

        if mup_base_d_model:
            output_logits_alpha = config["model"].get(
                "output_logits_alpha", None
            )
            if output_logits_alpha is None:
                output_logits_alpha = 1.0

            scale_output_logits_by_d = config["model"].get(
                "scale_output_logits_by_d", None
            )
            if scale_output_logits_by_d is None:
                scale_output_logits_by_d = False

            d_model = config["model"]["d_model"]
            d_model_width_mult = d_model / mup_base_d_model

            if scale_output_logits_by_d:
                output_logits_scale = output_logits_alpha / d_model_width_mult
            else:
                output_logits_scale = (
                    output_logits_alpha / d_model_width_mult**0.5
                )

            new_state_dict[new_key] = (
                old_state_dict[old_key] * output_logits_scale
            )
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    @staticmethod
    def is_mup(config):
        return _is_mup(config["model"])

    @staticmethod
    def formats():
        return ("sP", "muP")


def _is_mup(model_config):
    return any(
        name.startswith('mup_base_') and model_config[name] is not None
        for name in model_config
    )


def _is_mup_CS22(model_config):
    scale_qk_dot_by_d = model_config.get('scale_qk_dot_by_d', False)
    embeddings_scale = model_config.get('embeddings_scale', None)
    output_logits_scale = model_config.get('output_logits_scale', None)

    all_set = scale_qk_dot_by_d and embeddings_scale and output_logits_scale
    # embeddings_scale defaults to 1.0 in many models so check for the other two
    any_set = scale_qk_dot_by_d or output_logits_scale

    if any_set and not all_set:
        raise ValueError(
            "This looks like an incomplete muP config. Either all of or none of "
            "\"scale_qk_dot_by_d\", \"embeddings_scale\", \"output_logits_scale\" can be "
            "specified, but this config only has some that are specified."
        )

    return all_set
