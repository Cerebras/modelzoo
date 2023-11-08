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

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import torch
import yaml
from tqdm import tqdm

import cerebras_pytorch as cstorch
from modelzoo.common.pytorch.model_utils.checkpoint_converters.streaming_checkpoints import (
    StreamingCSWriter,
    StreamingShardedHFReader,
    StreamingShardedHFWriter,
)


class EquivalentSubkey:
    r"""EquivalentSubkey defines the bidirectional relationship between subkeys of a model's checkpoint.
    This class is simply a 2-tuple with index bounds checking.

    For example if the normalization layer in one model is named "norm" and "ln" in the other,
    the relationship can be represented as EquivalentSubkey("norm", "ln").
    """

    def __init__(self, a: str, b: str) -> None:
        self.keys = [a, b]

    def __getitem__(self, idx: int) -> str:
        assert (
            idx == 0 or idx == 1
        ), "Invalid index into EquivalentSubkey object: {}".format(idx)
        return self.keys[idx]

    def __repr__(self) -> str:
        return "EquivalentSubkey(\"{}\", \"{}\")".format(*self.keys)


class ConversionRule:
    r"""ConversionRule defines a "rule" which:
        1. a key can be matched against
        2. procedure for converting this old key to a new one upon a successful match
        3. and an action to be taken once the new key is created (ex: updating the
           state dictionary)

    A rule consists of a sequence of regex pattern (supplied as a string),
    EquivalentSubkey object, and (possibly) a BaseDictionaryConverter as long as this
    object is last in the sequence. It also contains an "exists" argument which
    can be set to "left", "both", or "right". The "left" and "right" arguments
    are used to describe if a key exists in one checkpoint format but not the
    other and should be ignored. Without this behavior, keys that exist in one
    but not the other wouldn't be matched by any conversion rules, causing a failure
    as drop_unmatched_keys is disabled by default.

    Example:
    The following describes the conversion rule for mapping HF's layer normalization
    key to CS layer normalization in the GPT2 model.
        >>> ConversionRule(
        >>>     [
        >>>         EquivalentSubkey("h", "transformer_decoder.layers"),
        >>>         "\.\d+\.",
        >>>         EquivalentSubkey("ln_1", "norm1"),
        >>>         "\.(weight|bias)",
        >>>     ],
        >>>     action=BaseCheckpointConverter.replaceKey,
        >>> )
    This should be interpreted as:
        1. HF uses 'h' to represent the decoder name while CS uses 'transformer_decoder.layers'
        2. Both will have keys that follow with a dot, the decoder number, and then another dot
        3. HF uses 'ln_1' for the first layer norm while CS names it 'norm1'
        4. Both will have keys that follow with a dot and then either weight or bias
    This representation should make it easy to see how we can 1) build a regex which
    matches against old keys, and 2) use the matched result & EquivalentSubkey information
    to create a new key. Finally, once the new key is constructed the conversion rule
    will apply the 'action' described by the user in order to complete the conversion
    (in this case simply copying the value at old_state's old key into the new_state at the
    new key).

    As previously mentioned, a conversion rule object can also contain a checkpoint
    converter at the end of the sequence. This is used to create a new checkpoint
    converter which uses another converter to handle a portion of the conversion.
    Doing so reduces the amount of copy & pasted conversion rules. For example,
    many models have base model classes which are extended with additional layers
    for fine-tuning. For example, HF's GP2Model doesn't contain a language model head
    while GP2LMHeadModel does. Rather than copying the conversion rules, we could
    instead define a new checkpoint converter as follows:

    >>> class Converter_GPT2LMHeadModel_HF_CS17(BaseDictionaryConverter):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.rules = [
    >>>             ConversionRule(
    >>>                 ["lm_head\.(weight|bias)"],
    >>>                 action=BaseCheckpointConverter.replaceKey,
    >>>             ),
    >>>             ConversionRule(
    >>>                 [
    >>>                     EquivalentSubkey("transformer.", ""),
    >>>                     Converter_GPT2Model_HF_CS17(),
    >>>                 ],
    >>>                 action=None,
    >>>             ),
    >>>         ]

    The first rule simply notates that the lm_head key now exists (and is named
    the same in both models). The second rule notates that if the "transformer."
    prefix is encountered, we should try all of the GPT2Model HF -> CS 1.7
    conversion rules.
    """

    def __init__(
        self,
        segments: List[Union[str, EquivalentSubkey, BaseDictionaryConverter]],
        exists: str = "both",
        action: Optional[
            Callable[[str, OrderedDict, str, OrderedDict, int], None]
        ] = None,
    ) -> None:
        assert isinstance(segments, list), "Expected segments to be list"
        for elm in segments:
            assert (
                isinstance(elm, str)
                or isinstance(elm, EquivalentSubkey)
                or isinstance(elm, BaseDictionaryConverter)
            ), f"ConversionRule segment doesn't support type {type(elm)}"
        assert exists in ["left", "both", "right"]

        self.segments = segments
        self.exists = exists
        self.action = action
        self.validate_segments()

    def __repr__(self) -> str:
        single_line = len(self.segments) < 2
        out = "ConversionRule(["
        if not single_line:
            out += "\n"
        for i in range(len(self.segments)):
            mod_str = repr(self.segments[i])
            if not single_line:
                mod_str = _addindent(mod_str, 4) + ",\n"
            out += mod_str
        action_name = "self." + self.action.__name__ if self.action else "None"
        out += "], action={})".format(action_name)
        return out

    @staticmethod
    def segment_is_converter(
        elm: Union[str, EquivalentSubkey, BaseDictionaryConverter]
    ) -> bool:
        return isinstance(elm, BaseDictionaryConverter)

    def validate_segments(self):
        for seg in self.segments:
            if isinstance(seg, str):
                pattern = re.compile(seg)
                assert (
                    pattern.groups == 0
                ), "The following regex isn't supported: {}\n\
                    Compile rule's regex cannot contain capture groups.\n\
                    Use (?:a|b) instead of (a|b)".format(
                    seg
                )

    def convert_key(
        self,
        old_key: str,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        from_index: int,
        match_start: int = 0,
        prefix: str = "",
        action_fn_args: Optional[dict] = None,
        debug: bool = False,
    ) -> bool:
        regex_str = ""
        maybe_escape = (
            lambda elm, idx: re.escape(elm[idx])
            if isinstance(elm, EquivalentSubkey)
            else elm
        )

        regex_str = ""
        chained_converter = ConversionRule.segment_is_converter(
            self.segments[-1]
        )
        candidate_segments = len(self.segments)
        if chained_converter:
            candidate_segments -= 1

        for i in range(candidate_segments):
            elm = self.segments[i]
            assert not ConversionRule.segment_is_converter(
                elm
            ), "Checkpoint convert objects can only be placed at the end of rules"
            regex_str += "({})".format(maybe_escape(elm, from_index))

        pattern = re.compile(regex_str)
        match_result = (
            pattern.fullmatch(old_key, match_start)
            if not chained_converter
            else pattern.match(old_key, match_start)
        )

        if match_result is None:
            return False

        converted = prefix
        to_index = 1 - from_index
        for i in range(candidate_segments):
            if isinstance(self.segments[i], EquivalentSubkey):
                converted += self.segments[i][to_index]
            else:
                converted += match_result.group(
                    i + 1
                )  # Index 0 always contains full match

        if chained_converter:
            converter = self.segments[-1]
            return converter.convert_key(
                old_key,
                old_state_dict,
                new_state_dict,
                from_index,
                match_start=match_result.span()[1],
                prefix=converted,
                action_fn_args=action_fn_args,
                debug=debug,
            )
        else:
            if debug:
                print(
                    "Matched {} -> {} action: {}".format(
                        old_key,
                        converted,
                        self.action.__name__ if self.action else "None",
                    )
                )
            if self.action:
                self.action(
                    old_key,
                    converted,
                    old_state_dict,
                    new_state_dict,
                    from_index,
                    action_fn_args,
                )
            return True

    def exists_in_index(self, to_index: int) -> bool:
        return (
            self.exists == "both"
            or (self.exists == "left" and to_index == 0)
            or (self.exists == "right" and to_index == 1)
        )


class FormatVersions(list):
    def __init__(self, *versions) -> None:
        self.formats = [*versions]

    def __contains__(self, key):
        return key in self.formats

    def __str__(self) -> str:
        return ", ".join(self.formats)


class BaseDictionaryConverter(ABC):
    r"""A dictionary converter represents a pair of two dictionary formats that
        can be converted between each other. The converter object defines a list
        of conversion rules which should be applied when converting one dict
        format to the other (and vice-versa).

        In order to make your own dictionary converter, simply:
        1. Create a new converter class which inherits from BaseDictionaryConverter
        2. Supply a list of conversion rules (self.rules)
        3. Override the pre_model_convert or post_model_convert hooks if you
           need to execute arbitrary behavior before/after the conversion.
    """

    def __init__(self, pbar_desc=None):
        self.pbar_desc = pbar_desc

    def __repr__(self) -> str:
        out = "BaseDictionaryConverter([\n"
        for i in range(len(self.rules)):
            out += _addindent(repr(self.rules[i]), 4) + ",\n"
        out += "])"
        return out

    @staticmethod
    @abstractmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        pass

    @classmethod
    def supports_conversion(cls, src_fmt, tgt_fmt):
        return cls.get_from_index(src_fmt, tgt_fmt) is not None

    @classmethod
    def get_from_index(cls, src_fmt, tgt_fmt):
        formats = cls.formats()
        assert (
            formats is not None
        ), "Class {} hasn't provided formats() which is required.".format(
            cls.__name__
        )
        if src_fmt in formats[0] and tgt_fmt in formats[1]:
            return 0
        elif src_fmt in formats[1] and tgt_fmt in formats[0]:
            return 1
        else:
            return None

    @staticmethod
    def replaceKey(
        old_key: str,
        new_key: str,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        from_index: int,
        action_fn_args: Optional[dict] = None,
    ) -> None:
        r"""
        Copies value that exists at old_state_dict's old_key to new_state_dict's
        new_key.
        """
        new_state_dict[new_key] = old_state_dict[old_key]

    def convert_key(
        self,
        old_key: str,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        from_index: int,
        match_start: int = 0,
        prefix: str = "",
        action_fn_args: Optional[dict] = None,
        debug: bool = False,
    ) -> None:
        r"""
        Attempts to convert the old key by matching against the list of
        conversion rules. The first rule to match is used for conversion (i.e.
        even if multiple rules *would* match, the latter ones are never used).
        Returns True if a conversion occurred.
        """
        assert hasattr(
            self, "rules"
        ), "Converter must have a list of conversion rules"
        for rule in self.rules:
            did_convert = rule.convert_key(
                old_key,
                old_state_dict,
                new_state_dict,
                from_index,
                match_start,
                prefix,
                action_fn_args,
                debug=debug,
            )
            if did_convert:
                return True
        return False

    def convert_all_keys(
        self,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        from_index: int,
        action_fn_args: Optional[dict] = None,
        no_progress_bar: bool = True,
        debug: bool = False,
        suppress_unmatched_key_warning: bool = False,
    ):
        if not no_progress_bar:
            pbar = tqdm(total=len(old_state_dict), desc=self.pbar_desc)

        matched_all_keys = True
        for key in old_state_dict.keys():
            matched_current_key = self.convert_key(
                key,
                old_state_dict,
                new_state_dict,
                from_index,
                action_fn_args=action_fn_args,
                debug=debug,
            )
            if not matched_current_key and not suppress_unmatched_key_warning:
                logging.warning("Key not matched: {}".format(key))
            if not no_progress_bar:
                pbar.update(1)
            matched_all_keys = matched_all_keys and matched_current_key
        return matched_all_keys


class BaseCheckpointConverter(BaseDictionaryConverter, ABC):
    r"""Converts between checkpoint state_dict formats."""

    def __init__(self):
        super().__init__(pbar_desc="Converting Checkpoint")

    @staticmethod
    @abstractmethod
    def file_formats() -> Tuple[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def get_config_converter_class() -> BaseConfigConverter:
        pass

    @classmethod
    @abstractmethod
    def load(cls, file: str, from_index: int, **kwargs) -> OrderedDict:
        pass

    @classmethod
    @abstractmethod
    def save(
        cls,
        file_without_ext: str,
        checkpoint: OrderedDict,
        from_index: int,
        **kwargs,
    ) -> str:
        pass

    @classmethod
    @abstractmethod
    def init_output_checkpoint(
        cls, file_without_ext: str, from_index: int, **kwargs,
    ) -> str:
        r"""
        (Pre)Initializes the output checkpoint at a supplied path. This is used
        in streaming conversion when the checkpoint is written to file as
        conversion is performed rather than accumulating the full checkpoint
        in memory and saving to file at the very end.
        """

    @classmethod
    def convert(
        cls,
        input_checkpoint,
        configs,
        checkpoint_from_index,
        output_checkpoint=OrderedDict(),
        **kwargs,
    ):
        instance = cls()
        output_checkpoint = instance.convert_helper(
            input_checkpoint,
            configs,
            checkpoint_from_index,
            output_checkpoint=output_checkpoint,
            **kwargs,
        )
        return output_checkpoint

    def convert_helper(
        self,
        input_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
        output_checkpoint=OrderedDict(),
        drop_unmatched_keys: bool = False,
        no_progress_bar: bool = True,
        debug: bool = False,
    ):
        r"""
        Converts all keys in a checkpoint from `from_index` format to the other
        format. Conversion will fail if at least one of the keys did not match
        on any conversion rules and drop_unmatched_keys is not enabled. Returns
        the newly converted checkpoint.
        """
        self.pre_checkpoint_convert(
            input_checkpoint, output_checkpoint, configs, from_index
        )

        old_state_dict, new_state_dict = self.extract_model_dict(
            input_checkpoint, output_checkpoint, configs, from_index,
        )

        self.pre_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            from_index,
            drop_unmatched_keys,
        )

        matched_all_keys = self.convert_all_keys(
            old_state_dict,
            new_state_dict,
            from_index,
            action_fn_args={"configs": configs},
            no_progress_bar=no_progress_bar,
            debug=debug,
        )

        self.post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            from_index,
            drop_unmatched_keys,
        )

        if not matched_all_keys and not drop_unmatched_keys:
            assert (
                matched_all_keys
            ), "Unable to match all keys. If you want to proceed by dropping keys that couldn't matched, rerun with --drop-unmatched-keys"
        elif not matched_all_keys:
            logging.warning(
                "proceeding even though some keys weren't matched because of --drop-unmatched-keys"
            )

        self.post_checkpoint_convert(
            input_checkpoint, output_checkpoint, configs, from_index
        )
        return output_checkpoint

    def pre_model_convert(
        self,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        configs: Tuple[dict, dict],
        from_index: int,
        drop_unmatched_keys: bool,
    ):
        r"""
        Hook executes right before model conversion.
        """

    def post_model_convert(
        self,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        configs: Tuple[dict, dict],
        from_index: int,
        drop_unmatched_keys: bool,
    ):
        r"""
        Hook executes right after model conversion.
        """

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
    ):
        r"""
        Hook executes before checkpoint conversion.
        """

    def extract_model_dict(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
    ):
        r"""
        Hook to extract model state dicts out of the input/output checkpoint
        """
        return input_checkpoint, output_checkpoint

    def post_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
    ):
        r"""
        Hook executes after checkpoint conversion.
        """


class BaseCheckpointConverter_CS_CS(BaseCheckpointConverter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def file_formats() -> Tuple[str, str]:
        return ("mdl", "mdl")

    @classmethod
    def load(cls, file: str, from_index: int) -> OrderedDict:
        return cstorch.load(file, map_location="cpu")

    @classmethod
    def save(
        cls, file_without_ext: str, checkpoint: OrderedDict, from_index: int,
    ) -> OrderedDict:

        if isinstance(checkpoint, StreamingCSWriter):
            checkpoint.save()
            return checkpoint.checkpoint_file
        else:
            to_index = from_index - 1
            output_file_format = cls.file_formats()[to_index]
            file = file_without_ext + "." + output_file_format
            cstorch.save(checkpoint, file)
            return file

    @classmethod
    def init_output_checkpoint(
        cls, file_without_ext: str, from_index: int, **kwargs
    ) -> str:
        to_index = from_index - 1
        output_file_format = cls.file_formats()[to_index]
        file = file_without_ext + "." + output_file_format
        return StreamingCSWriter(file)

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
    ):
        # Copy non model keys like optimizer state:
        for category in input_checkpoint:
            if category != "model":
                output_checkpoint[category] = input_checkpoint[category]
        output_checkpoint["model"] = {}

    def extract_model_dict(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
    ):
        return input_checkpoint["model"], output_checkpoint["model"]


class BaseCheckpointConverter_HF_CS(BaseCheckpointConverter_CS_CS):
    r"""HF checkpoints contain model only while CS checkpoints package model,
    optimizer, and lr_scheduler into a single checkpoint. This class overrides
    the post_checkpoint_convert to automatically extract/package the state_dict
    correctly.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def file_formats() -> Tuple[str, str]:
        return ("bin", "mdl")

    @classmethod
    def load(cls, file: str, from_index: int,) -> OrderedDict:
        if file.endswith(".index.json"):
            assert (
                from_index == 0
            ), ".index.json files are only supported when doing HF -> CS conversion"
            return StreamingShardedHFReader(file)
        else:
            # Any other type of checkpoint
            return cstorch.load(file, map_location="cpu")

    @classmethod
    def save(
        cls, file_without_ext: str, checkpoint: OrderedDict, from_index: int,
    ) -> OrderedDict:

        if isinstance(checkpoint, StreamingCSWriter):
            checkpoint.save()
            return checkpoint.checkpoint_file
        elif isinstance(checkpoint, StreamingShardedHFWriter):
            checkpoint.save()
            return checkpoint.checkpoint_dir
        else:
            to_index = from_index - 1
            output_file_format = cls.file_formats()[to_index]
            file = file_without_ext + "." + output_file_format
            if from_index == 0:
                torch.save(checkpoint, file)
            else:
                cstorch.save(checkpoint, file)
            return file

    @classmethod
    def init_output_checkpoint(
        cls,
        file_without_ext: str,
        from_index: int,
        hf_shard_size: Union[str, int] = "10GB",
        **kwargs,
    ) -> str:
        to_index = from_index - 1
        output_file_format = cls.file_formats()[to_index]
        file = file_without_ext + "." + output_file_format

        if from_index == 0:
            # HF -> CS
            return StreamingCSWriter(file)
        else:
            # CS -> HF
            return StreamingShardedHFWriter(
                file_without_ext, shard_size=hf_shard_size
            )

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
    ):
        if from_index == 0:
            output_checkpoint["model"] = {}

    def extract_model_dict(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        from_index: int,
    ):
        if from_index == 0:
            return input_checkpoint, output_checkpoint["model"]
        else:
            return input_checkpoint["model"], output_checkpoint

    def post_model_convert(
        self,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        configs: Tuple[dict, dict],
        from_index: int,
        drop_unmatched_keys: bool,
    ):
        if from_index == 1:
            cs_config = configs[from_index]
            if cs_config.get("sparsity", {}).get("type") == "sideband":
                # Finalize the CS sparsity in CS -> HF conversion.
                logging.info(
                    "Finalizing sparsity. The output checkpoint will be dense."
                )
                for key, weight in new_state_dict.items():
                    weight[weight.isnan()] = 0
                    new_state_dict[key] = weight


class ConfigConversionError(Exception):
    "Raised when a config cannot be converted"


class BaseConfigConverter(BaseDictionaryConverter, ABC):
    def __init__(self):
        super().__init__(pbar_desc="Converting Config")
        self.pre_convert_defaults = [{}, {}]
        self.post_convert_defaults = [{}, {}]

    @staticmethod
    @abstractmethod
    def file_formats() -> Tuple[str, str]:
        pass

    @classmethod
    def load(cls, file: str, from_index: int) -> dict:
        input_file_format = cls.file_formats()[from_index]
        if input_file_format == "json":
            with open(file, "r") as f:
                return json.load(f)
        elif input_file_format == "yaml":
            with open(file, "r") as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        else:
            raise ValueError(
                "Unsupported input file format: {}".format(input_file_format())
            )

    @classmethod
    def save(cls, file_without_ext: str, config: dict, from_index: int) -> str:
        to_index = (from_index + 1) % 2
        output_file_format = cls.file_formats()[to_index]
        file = file_without_ext + "." + output_file_format
        if output_file_format == "json":
            with open(file, "w") as f:
                f.write(json.dumps(config, indent=4))
        elif output_file_format == "yaml":
            with open(file, "w") as f:
                f.write(yaml.dump(config, indent=4))
        else:
            raise ValueError(
                "Unsupported input file format: {}".format(output_file_format())
            )
        return file

    @classmethod
    def convert(
        cls,
        config,
        from_index: int,
        drop_unmatched_keys: bool = False,
        no_progress_bar: bool = True,
        debug: bool = False,
    ):
        instance = cls()
        return instance.convert_helper(
            config,
            from_index,
            drop_unmatched_keys=drop_unmatched_keys,
            no_progress_bar=no_progress_bar,
            debug=debug,
        )

    def convert_helper(
        self,
        config,
        from_index: int,
        drop_unmatched_keys: bool = False,
        no_progress_bar: bool = True,
        debug: bool = False,
    ):
        r"""
        Converts all keys in a config from `from_index` format to the other
        format. Conversion will fail if at least one of the keys did not match
        on any conversion rules and drop_unmatched_keys is not enabled. Returns
        the newly converted config.
        """

        old_config = self.pre_config_convert(config, from_index)
        new_config = {}

        matched_all_keys = self.convert_all_keys(
            old_config,
            new_config,
            from_index,
            no_progress_bar=no_progress_bar,
            debug=debug,
            suppress_unmatched_key_warning=drop_unmatched_keys,
        )

        if not matched_all_keys and not drop_unmatched_keys:
            assert matched_all_keys, "Unable to match all keys in config."

        final_config = self.post_config_convert(
            config, old_config, new_config, from_index, drop_unmatched_keys
        )
        return final_config

    def pre_config_convert(
        self, config, from_index,
    ):
        for key in self.pre_convert_defaults[from_index]:
            if key not in config:
                config[key] = self.pre_convert_defaults[from_index][key]
            elif isinstance(self.pre_convert_defaults[from_index][key], dict):
                for subkey, subvalue in self.pre_convert_defaults[from_index][
                    key
                ].items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
        return config

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        to_index = 1 - from_index
        for key in self.post_convert_defaults[to_index]:
            if key not in new_config:
                new_config[key] = self.post_convert_defaults[to_index][key]
            elif isinstance(self.post_convert_defaults[to_index][key], dict):
                for subkey, subvalue in self.post_convert_defaults[to_index][
                    key
                ].items():
                    if subkey not in new_config[key]:
                        new_config[key][subkey] = subvalue
        return new_config

    @staticmethod
    def assert_factory_fn(assert_index, assert_value):
        def assert_factory_wrapper(
            old_key,
            new_key,
            old_state_dict,
            new_state_dict,
            from_index,
            action_fn_args,
        ):

            if from_index != assert_index:
                raise ConfigConversionError(
                    f"{old_key} should not appear in the config"
                )

            if old_state_dict[old_key] != assert_value:
                raise ConfigConversionError(
                    "Can't convert config with {}={}. Only {} is supported.".format(
                        old_key, old_state_dict[old_key], assert_value
                    )
                )

        return assert_factory_wrapper


class BaseConfigConverter_HF_CS(BaseConfigConverter):
    r"""CS packages model, optimizer, and lr_scheduler into a single config.
    This class overrides the [pre|post]_config_convert fn to automatically
    extract/package the model configuration correctly.
    """

    def __init__(self):
        super().__init__()
        self.post_convert_defaults[1]["mixed_precision"] = True

    @staticmethod
    def file_formats() -> Tuple[str, str]:
        return ("json", "yaml")

    def pre_config_convert(
        self, config, from_index,
    ):
        model_config = config["model"] if from_index == 1 else config
        return super().pre_config_convert(model_config, from_index)

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        model_config = super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

        if from_index == 0:
            return {"model": model_config}
        else:
            return model_config


class BaseConfigConverter_CS_CS(BaseConfigConverter):
    r"""CS packages model, optimizer, and lr_scheduler into a single config.
    This class overrides the [pre|post]_config_convert fn to automatically
    extract/package the model configuration correctly.
    """

    def __init__(self):
        super().__init__()
        self.post_convert_defaults[0]["mixed_precision"] = True
        self.post_convert_defaults[1]["mixed_precision"] = True

    @staticmethod
    def file_formats() -> Tuple[str, str]:
        return ("yaml", "yaml")

    def pre_config_convert(
        self, config, from_index,
    ):
        return super().pre_config_convert(config["model"], from_index)

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        final_config = {
            key: original_config[key]
            for key in original_config
            if key != "model"
        }
        final_config["model"] = super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )
        return final_config


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    return s
