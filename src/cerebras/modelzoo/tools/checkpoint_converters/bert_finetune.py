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

import logging
from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_CS_CS,
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatIndices,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.bert import (
    ConfigConverter_Bert_CS16_CS17,
    ConfigConverter_Bert_CS16_CS18,
    ConfigConverter_Bert_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
    Converter_BertModel_CS16_CS17,
    Converter_BertModel_WithoutOptionalModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)


class Converter_BertFinetuneModel_CS16_CS17(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    r"bert\.",
                    Converter_BertModel_CS16_CS17(),
                ],
            ),
            ConversionRule(
                [r"classifier\.(?:weight|bias)"],
                action=self.replaceKey,
            ),
        ]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Don't copy non model keys like optimizer state:
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        output_checkpoint["model"] = {}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "BertForSequenceClassification, BertForTokenClassification, "
            "BertForQuestionAnswering, and BertForSummarization classes"
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS17


class Converter_BertFinetuneModel_CS16_CS18(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertFinetuneModel_CS16_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertFinetuneModel_CS16_CS17(),
                ],
                action=None,
            ),
        ]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Don't copy non model keys like optimizer state:
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        output_checkpoint["model"] = {}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.6"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return (
            "BertForSequenceClassification, BertForTokenClassification, "
            "BertForQuestionAnswering, and BertForSummarization classes"
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS18


class Converter_BertForSequenceClassification_HF_CS17(
    Converter_BertFinetuneModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def pre_checkpoint_convert(self, *args):
        return BaseCheckpointConverter_HF_CS.pre_checkpoint_convert(
            self,
            *args,
        )

    def extract_model_dict(self, *args):
        return BaseCheckpointConverter_HF_CS.extract_model_dict(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForSequenceClassification".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForSequenceClassification_HF_CS17


class Converter_BertForSequenceClassification_HF_CS18(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertForSequenceClassification_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertForSequenceClassification_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForSequenceClassification".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForSequenceClassification_HF_CS18


class ConfigConverter_BertForSequenceClassification_HF_CS17(
    ConfigConverter_Bert_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Fine-tuning config params
            ConversionRule(
                [EquivalentSubkey("classifier_dropout", "task_dropout")],
                action=self.replaceKey,
            ),
            ConversionRule(["num_labels"], action=self.replaceKey),
            ConversionRule(["problem_type"], action=self.replaceKey),
            *self.rules,
        ]

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        # pylint: disable=line-too-long
        # From https://github.com/huggingface/transformers/blob/23c146c38b42d1193849fbd6f2943bf754b7c428/src/transformers/models/bert/modeling_bert.py#L1579
        if converter_indices.direction == 0:
            if "num_labels" not in config:
                if "id2label" in config:
                    config["num_labels"] = len(config["id2label"])
                else:
                    config["num_labels"] = 2

            if (
                "classifier_dropout" not in config
                or config["classifier_dropout"] is None
            ):
                config["classifier_dropout"] = config["hidden_dropout_prob"]
            if "problem_type" not in config or config["problem_type"] is None:
                if config["num_labels"] == 1:
                    config["problem_type"] = "regression"
                else:
                    raise ConfigConversionError(
                        "Cannot infer the problem_type (it is either single_label_classification "
                        "or multi_label_classification). Please explicitly include the "
                        "problem_type field before re-running."
                    )

        return config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_BertForSequenceClassification_HF_CS18(
    ConfigConverter_BertForSequenceClassification_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )


class Converter_BertForTokenClassification_HF_CS17(
    Converter_BertFinetuneModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def pre_checkpoint_convert(
        self,
        *args,
    ):
        return BaseCheckpointConverter_HF_CS.pre_checkpoint_convert(
            self,
            *args,
        )

    def extract_model_dict(self, *args):
        return BaseCheckpointConverter_HF_CS.extract_model_dict(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForTokenClassification".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForTokenClassification_HF_CS17


class Converter_BertForTokenClassification_HF_CS18(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertForTokenClassification_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertForTokenClassification_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForTokenClassification".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForTokenClassification_HF_CS18


class ConfigConverter_BertForTokenClassification_HF_CS17(
    ConfigConverter_Bert_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Fine-tuning config params
            ConversionRule(
                [
                    EquivalentSubkey(
                        "classifier_dropout", "encoder_output_dropout_rate"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_labels", "num_classes")],
                action=self.replaceKey,
            ),
            *self.rules,
        ]

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        # Additional Finetune specific defaults:
        if converter_indices.direction == 0:
            if "num_labels" not in config:
                if "id2label" in config:
                    config["num_labels"] = len(config["id2label"])
                else:
                    config["num_labels"] = 2
            if (
                "classifier_dropout" not in config
                or config["classifier_dropout"] is None
            ):
                config["classifier_dropout"] = config["hidden_dropout_prob"]

        return config

    def post_config_convert(
        self,
        model,
        original_config,
        old_config,
        new_config,
        converter_indices,
        drop_unmatched_keys,
    ):
        if converter_indices.direction == 0:
            if "loss_weight" not in new_config:
                new_config["loss_weight"] = 1.0
            if "include_padding_in_loss" not in new_config:
                new_config["include_padding_in_loss"] = False

        return super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_BertForTokenClassification_HF_CS18(
    ConfigConverter_BertForTokenClassification_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )


class Converter_BertForQuestionAnswering_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    r"bert\.",
                    Converter_BertModel_CS16_CS17(),
                ],
            ),
            ConversionRule(
                [
                    EquivalentSubkey("qa_outputs", "classifier"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForQuestionAnswering".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForQuestionAnswering_HF_CS17


class Converter_BertForQuestionAnswering_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertForQuestionAnswering_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertForQuestionAnswering_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForQuestionAnswering".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForQuestionAnswering_HF_CS18


class ConfigConverter_BertForQuestionAnswering_HF_CS17(
    ConfigConverter_Bert_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Fine-tuning config params
            ConversionRule(
                ["num_labels"],
                action=BaseConfigConverter.assert_factory_fn(0, 2),
            ),
            *self.rules,
        ]
        self.post_convert_defaults[0].update({"num_labels": 2})

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        # Additional Finetune specific defaults:
        if converter_indices.direction == 0:
            if "num_labels" not in config:
                if "id2label" in config:
                    config["num_labels"] = len(config["id2label"])
                else:
                    config["num_labels"] = 2

        return config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_BertForQuestionAnswering_HF_CS18(
    ConfigConverter_BertForQuestionAnswering_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################

# Converter_Bert_CS17_CS18


class ConfigConverter_BertForSequenceClassification_HF_CS21(
    ConfigConverter_BertForSequenceClassification_HF_CS18
):
    "CS 2.1 config is the same as CS 2.0."

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_BertForSequenceClassification_WithoutOptionalModel_HF_CS21(
    Converter_BertForSequenceClassification_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    "bert\.",
                    Converter_BertModel_WithoutOptionalModel_HF_CS21(),
                ],
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForSequenceClassification_HF_CS21


Converter_BertForSequenceClassification_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BertForSequenceClassification_HF_CS21",
    Converter_BertForSequenceClassification_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_BertForSequenceClassification_WithoutOptionalModel_HF_CS21,
)


class ConfigConverter_BertForTokenClassification_HF_CS21(
    ConfigConverter_BertForTokenClassification_HF_CS18
):
    "CS 2.1 config is the same as CS 2.0."

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_BertForTokenClassification_WithoutOptionalModel_HF_CS21(
    Converter_BertForTokenClassification_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    "bert\.",
                    Converter_BertModel_WithoutOptionalModel_HF_CS21(),
                ],
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForTokenClassification_HF_CS21


Converter_BertForTokenClassification_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BertForTokenClassification_HF_CS21",
    Converter_BertForTokenClassification_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_BertForTokenClassification_WithoutOptionalModel_HF_CS21,
)


class ConfigConverter_BertForQuestionAnswering_HF_CS21(
    ConfigConverter_BertForQuestionAnswering_HF_CS18
):
    "CS 2.1 config is the same as CS 2.0."

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_BertForQuestionAnswering_WithoutOptionalModel_HF_CS21(
    Converter_BertForQuestionAnswering_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    "bert\.",
                    Converter_BertModel_WithoutOptionalModel_HF_CS21(),
                ],
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForQuestionAnswering_HF_CS21


Converter_BertForQuestionAnswering_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BertForQuestionAnswering_HF_CS21",
    Converter_BertForQuestionAnswering_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_BertForQuestionAnswering_WithoutOptionalModel_HF_CS21,
)
