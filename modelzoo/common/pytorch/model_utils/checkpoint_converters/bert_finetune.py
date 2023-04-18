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

from modelzoo.common.pytorch.model_utils.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseCheckpointConverter_PT_PT,
    BaseConfigConverter,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    converter_notes,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.bert import (
    ConfigConverter_Bert_CS16_CS17,
    ConfigConverter_Bert_CS16_CS18,
    ConfigConverter_Bert_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
    Converter_BertModel_CS16_CS17,
)


class Converter_BertFinetuneModel_CS16_CS17(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(["bert\.", Converter_BertModel_CS16_CS17(),],),
            ConversionRule(
                ["classifier\.(?:weight|bias)"], action=self.replaceKey,
            ),
        ]

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        return {"model": checkpoint["model"]}

    @staticmethod
    @converter_notes(
        notes="""
    BertForSequenceClassification, BertForTokenClassification,
    BertForQuestionAnswering, and BertForSummarization classes"""
    )
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.7")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS17


class Converter_BertFinetuneModel_CS16_CS18(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_BertFinetuneModel_CS16_CS17(),], action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertFinetuneModel_CS16_CS17(),
                ],
                action=None,
            ),
        ]

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        return {"model": checkpoint["model"]}

    @staticmethod
    @converter_notes(
        notes="""
    BertForSequenceClassification, BertForTokenClassificationLoss,
    BertForQuestionAnswering, and BertForSummarization classes"""
    )
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS18


class Converter_BertForSequenceClassification_HF_CS17(
    Converter_BertFinetuneModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        return BaseCheckpointConverter_HF_CS.post_checkpoint_convert(
            self, checkpoint, from_index
        )

    @staticmethod
    @converter_notes(notes="HF <-> CS 1.7 for BertForSequenceClassification")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")

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
                [Converter_BertForSequenceClassification_HF_CS17(),],
                action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertForSequenceClassification_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    @converter_notes(notes="HF <-> CS 1.8 for BertForSequenceClassification")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForSequenceClassification_HF_CS18


class ConfigConverter_BertForSequenceClassification_HF_CS17(
    ConfigConverter_Bert_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Finetuning config params
            ConversionRule(
                [EquivalentSubkey("classifier_dropout", "task_dropout")],
                action=self.replaceKey,
            ),
            ConversionRule(["num_labels"], action=self.replaceKey),
            ConversionRule(["problem_type"], action=self.replaceKey),
            *self.rules,
        ]

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)

        # From https://github.com/huggingface/transformers/blob/23c146c38b42d1193849fbd6f2943bf754b7c428/src/transformers/models/bert/modeling_bert.py#L1579
        if from_index == 0:
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
                        "Cannot infer the problem_type (it is either single_label_classification or multi_label_classification). Please explcitly include the problem_type field before re-running."
                    )

        return config

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")


class ConfigConverter_BertForSequenceClassification_HF_CS18(
    ConfigConverter_BertForSequenceClassification_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")


class Converter_BertForTokenClassification_HF_CS17(
    Converter_BertFinetuneModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        return BaseCheckpointConverter_HF_CS.post_checkpoint_convert(
            self, checkpoint, from_index
        )

    @staticmethod
    @converter_notes(notes="HF <-> CS 1.7 for BertForTokenClassification")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")

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
                [Converter_BertForTokenClassification_HF_CS17(),], action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertForTokenClassification_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    @converter_notes(notes="HF <-> CS 1.8 for BertForTokenClassification")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForTokenClassification_HF_CS18


class ConfigConverter_BertForTokenClassification_HF_CS17(
    ConfigConverter_Bert_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Finetuning config params
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
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)

        # Additional Finetune specific defaults:
        if from_index == 0:
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
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            if "loss_weight" not in new_config:
                new_config["loss_weight"] = 1.0
            if "include_padding_in_loss" not in new_config:
                new_config["include_padding_in_loss"] = False

        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")


class ConfigConverter_BertForTokenClassification_HF_CS18(
    ConfigConverter_BertForTokenClassification_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")


class Converter_BertForQuestionAnswering_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(["bert\.", Converter_BertModel_CS16_CS17(),],),
            ConversionRule(
                [
                    EquivalentSubkey("qa_outputs", "classifier"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    @converter_notes(notes="HF <-> CS 1.7 for BertForQuestionAnswering")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForQuestionAnswering_HF_CS17


class Converter_BertForQuestionAnswering_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_BertForQuestionAnswering_HF_CS17(),], action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertForQuestionAnswering_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    @converter_notes(notes="HF <-> CS 1.8 for BertForQuestionAnswering")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertForQuestionAnswering_HF_CS18


class ConfigConverter_BertForQuestionAnswering_HF_CS17(
    ConfigConverter_Bert_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Finetuning config params
            ConversionRule(
                ["num_labels"],
                action=BaseConfigConverter.assert_factory_fn(0, 2),
            ),
            *self.rules,
        ]

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)

        # Additional Finetune specific defaults:
        if from_index == 0:
            if "num_labels" not in config:
                if "id2label" in config:
                    config["num_labels"] = len(config["id2label"])
                else:
                    config["num_labels"] = 2

        print(config)

        return config

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 1:
            new_config["num_labels"] = 2

        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")


class ConfigConverter_BertForQuestionAnswering_HF_CS18(
    ConfigConverter_BertForQuestionAnswering_HF_CS17,
    ConfigConverter_Bert_HF_CS18,
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")
