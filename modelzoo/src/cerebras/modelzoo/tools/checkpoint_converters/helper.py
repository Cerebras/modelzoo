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

from collections import OrderedDict
from typing import Optional, Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter,
    BaseConfigConverter,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)


# CS models may contain an extra 'model.' prefix. During HF -> CS conversion,
# we do not want to output checkpoints with this prefix. In CS -> HF conversion,
# we want to handle both the extra 'model.' prefix and no prefix cases.
def Build_HF_CS_Converter_WithOptionalModel(
    name,
    converter,
    derived_class,
    config_converter_class=None,
    formats=None,
    converter_note_fn=None,
):
    assert issubclass(
        derived_class, BaseCheckpointConverter
    ), "derived_class parameter must be a subclass of BaseCheckpointConverter"

    class ConverterWithOptionalModel(derived_class):
        def __init__(self) -> None:
            super().__init__()
            self.rules = [
                ConversionRule(
                    [
                        converter(),
                    ],
                    action=None,
                ),
                # If above did not match, try to apply conversion with stripped
                # 'model.' prefix
                ConversionRule(
                    [
                        EquivalentSubkey("", "model."),
                        converter(),
                    ],
                    action=None,
                ),
            ]

    ConverterWithOptionalModel.__name__ = name

    if config_converter_class:
        assert issubclass(
            config_converter_class, BaseConfigConverter
        ), "config_converter_class parameter must be a subclass of BaseConfigConverter"

        @staticmethod
        def _get_config_converter_class() -> BaseConfigConverter:
            return config_converter_class

        ConverterWithOptionalModel.get_config_converter_class = (
            _get_config_converter_class
        )
        ConverterWithOptionalModel.__abstractmethods__ = (
            ConverterWithOptionalModel.__abstractmethods__.difference(
                {"get_config_converter_class"}
            )
        )

    if formats:
        assert (
            isinstance(formats, tuple)
            and len(formats) == 2
            and all(isinstance(e, FormatVersions) for e in formats)
        ), "formats argument must be a tuple of two FormatVersions"

        @staticmethod
        def _formats_fn() -> Tuple[FormatVersions, FormatVersions]:
            return formats

        ConverterWithOptionalModel.formats = _formats_fn
        ConverterWithOptionalModel.__abstractmethods__ = (
            ConverterWithOptionalModel.__abstractmethods__.difference(
                {"formats"}
            )
        )

    if converter_note_fn:

        @classmethod
        def _converter_note(cls) -> str:
            return converter_note_fn(cls)

        ConverterWithOptionalModel.converter_note = _converter_note
        ConverterWithOptionalModel.__abstractmethods__ = (
            ConverterWithOptionalModel.__abstractmethods__.difference(
                {"converter_note"}
            )
        )

    return ConverterWithOptionalModel


def convert_use_rms_layer_norm_helper(
    self,
    old_key,
    new_key,
    old_state_dict,
    new_state_dict,
    from_index,
    action_fn_args,
):
    if from_index == 0:
        new_state_dict[new_key] = (
            "rmsnorm" if old_state_dict[old_key] else "layernorm"
        )
    else:
        if old_state_dict[old_key] == "rmsnorm":
            new_state_dict[new_key] = True
        elif old_state_dict[old_key] == "layernorm":
            new_state_dict[new_key] = False
        else:
            raise ConfigConversionError(
                "{} did not support {}".format(
                    self.formats()[0], old_state_dict[old_key]
                )
            )


def convert_use_biasless_layer_norm_helper(
    self,
    old_key,
    new_key,
    old_state_dict,
    new_state_dict,
    from_index,
    action_fn_args,
):
    if from_index == 0:
        new_state_dict[new_key] = (
            "biasless-layernorm" if old_state_dict[old_key] else "layernorm"
        )
    else:
        if old_state_dict[old_key] == "biasless-layernorm":
            new_state_dict[new_key] = True
        elif old_state_dict[old_key] == "layernorm":
            new_state_dict[new_key] = False
        else:
            raise ConfigConversionError(
                "{} did not support {}".format(
                    self.formats()[0], old_state_dict[old_key]
                )
            )


# Old cstorch checkpoints had a bug where aliased weights would show up as None
# This helper function fixes this by tying old_key and new_key together
# if either one doesn't exist or is None.
def tie_none_weights(
    old_key: str,
    new_key: str,
    old_state_dict: OrderedDict,
    new_state_dict: OrderedDict,
    from_index: int,
    action_fn_args: Optional[dict] = None,
) -> None:
    r"""
    Ties weights stored at old_key & new_key
    """
    if new_key not in old_state_dict or (
        old_state_dict[old_key] is not None and old_state_dict[new_key] is None
    ):
        new_state_dict[old_key] = old_state_dict[old_key]
        new_state_dict[new_key] = old_state_dict[old_key]
    elif (
        old_state_dict[old_key] is None and old_state_dict[new_key] is not None
    ):
        new_state_dict[old_key] = old_state_dict[new_key]
        new_state_dict[new_key] = old_state_dict[new_key]
    else:
        new_state_dict[old_key] = old_state_dict[old_key]


# Ties old_key and new_key if share_embedding_weights is enabled in the config
# (default is enabled)
def maybe_tie_lm_head(
    old_key: str,
    new_key: str,
    old_state_dict: OrderedDict,
    new_state_dict: OrderedDict,
    from_index: int,
    action_fn_args: Optional[dict] = None,
) -> None:
    cs_config = action_fn_args["configs"][1]
    if cs_config["model"].get("share_embedding_weights", True):
        tie_none_weights(
            old_key,
            new_key,
            old_state_dict,
            new_state_dict,
            from_index,
            action_fn_args,
        )
    else:
        new_state_dict[old_key] = old_state_dict[old_key]


def transpose_key_if_2D(
    old_key,
    new_key,
    old_state_dict,
    new_state_dict,
    from_index,
    action_fn_args,
):
    # HF checkpoint stores some layers as Conv2D instead of Linear.
    # In those cases, we need to transpose the weight matrix for the
    # dimensions to line up when converting.
    x = old_state_dict[old_key]
    if len(x.shape) == 2:
        x = x.transpose(0, 1)
    new_state_dict[new_key] = x
