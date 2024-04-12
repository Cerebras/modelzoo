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

#!/bin/python3
# Converts PT checkpoints between different formats (cs releases, hf, etc)

import argparse
import logging
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Tuple, Union

from packaging.version import parse
from tabulate import tabulate

import cerebras.pytorch as cstorch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter,
    BaseConfigConverter,
    FormatIndices,
    fallback_converters,
)
from cerebras.modelzoo.tools.checkpoint_converters.bert import (
    Converter_Bert_CS17_CS18,
    Converter_Bert_CS18_CS20,
    Converter_Bert_CS20_CS21,
    Converter_BertPretrainModel_CS16_CS17,
    Converter_BertPretrainModel_CS16_CS18,
    Converter_BertPretrainModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.bert_finetune import (
    Converter_BertFinetuneModel_CS16_CS17,
    Converter_BertFinetuneModel_CS16_CS18,
    Converter_BertForQuestionAnswering_HF_CS21,
    Converter_BertForSequenceClassification_HF_CS21,
    Converter_BertForTokenClassification_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.falcon import (
    Converter_Falcon_CS20_CS21,
    Converter_Falcon_Headless_HF_CS21,
    Converter_Falcon_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.gpt2_hf_cs import (
    Converter_GPT2LMHeadModel_CS18_CS20,
    Converter_GPT2LMHeadModel_CS20_CS21,
    Converter_GPT2LMHeadModel_HF_CS21,
    Converter_GPT2Model_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.gpt_neox_hf_cs import (
    Converter_GPT_Neox_Headless_HF_CS21,
    Converter_GPT_Neox_LMHeadModel_CS18_CS20,
    Converter_GPT_Neox_LMHeadModel_CS20_CS21,
    Converter_GPT_Neox_LMHeadModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.gptj_hf_cs import (
    Converter_GPTJ_Headless_HF_CS20,
    Converter_GPTJ_LMHeadModel_CS18_CS20,
    Converter_GPTJ_LMHeadModel_CS20_CS21,
    Converter_GPTJ_LMHeadModel_HF_CS20,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    Converter_LlamaForCausalLM_CS19_CS20,
    Converter_LlamaForCausalLM_CS20_CS21,
    Converter_LlamaForCausalLM_HF_CS21,
    Converter_LlamaModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.llava import (
    Converter_LLaVA_HF_CS22,
)
from cerebras.modelzoo.tools.checkpoint_converters.roberta import (
    Converter_RobertaPretrainModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.salesforce_codegen_hf_cs import (
    Converter_Codegen_Headless_HF_CS20,
    Converter_Codegen_LMHeadModel_CS18_CS20,
    Converter_Codegen_LMHeadModel_CS20_CS21,
    Converter_Codegen_LMHeadModel_HF_CS20,
)
from cerebras.modelzoo.tools.checkpoint_converters.streaming_checkpoints import (
    StreamingShardedHFWriter,
)
from cerebras.modelzoo.tools.checkpoint_converters.t5 import (
    Converter_T5_CS16_CS17,
    Converter_T5_CS16_CS18,
    Converter_T5_CS17_CS18,
    Converter_T5_CS18_CS20,
    Converter_T5_CS20_CS21,
    Converter_T5_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.btlm_hf_cs import (  # noqa
    Converter_BTLMLMHeadModel_CS20_CS21,
    Converter_BTLMLMHeadModel_HF_CS21,
    Converter_BTLMModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.jais_hf_cs import (  # noqa
    Converter_JAISLMHeadModel_CS20_CS21,
    Converter_JAISLMHeadModel_HF_CS21,
    Converter_JAISModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.bloom_hf_cs import (  # noqa
    Converter_BloomLMHeadModel_CS19_CS20,
    Converter_BloomLMHeadModel_CS20_CS21,
    Converter_BloomLMHeadModel_HF_CS21,
    Converter_BloomModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.clip_vit import (  # noqa
    Converter_CLIPViT_Projection_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.vit import (  # noqa
    Converter_ViT_Headless_HF_CS21,
    Converter_ViT_HF_CS21,
)


from cerebras.modelzoo.tools.checkpoint_converters.mpt import (  # noqa
    Converter_MPTForCausalLM_HF_CS21,
    Converter_MPTModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.starcoder import (  # noqa
    Converter_StarcoderForCausalLM_HF_CS21,
    Converter_StarcoderModel_HF_CS21,
    Converter_StarcoderLMHeadModel_CS20_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.santacoder import (  # noqa
    Converter_SantacoderLMHeadModel_HF_CS21,
    Converter_SantacoderModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.mistral import (  # noqa
    Converter_MistralModel_HF_CS21,
    Converter_MistralForCausalLM_HF_CS21,
)

converters: Dict[str, List[BaseCheckpointConverter]] = {
    "bert": [
        Converter_BertPretrainModel_HF_CS21,
        Converter_BertPretrainModel_CS16_CS17,
        Converter_BertPretrainModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS18_CS20,
        Converter_Bert_CS20_CS21,
    ],
    "bert-sequence-classifier": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
        Converter_BertForSequenceClassification_HF_CS21,
    ],
    "bert-token-classifier": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
        Converter_BertForTokenClassification_HF_CS21,
    ],
    "bert-summarization": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
    ],
    "bert-qa": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
        Converter_BertForQuestionAnswering_HF_CS21,
    ],
    "bloom": [
        Converter_BloomLMHeadModel_CS19_CS20,
        Converter_BloomLMHeadModel_CS20_CS21,
        Converter_BloomLMHeadModel_HF_CS21,
    ],
    "bloom-headless": [
        Converter_BloomModel_HF_CS21,
    ],
    "btlm": [
        Converter_BTLMLMHeadModel_CS20_CS21,
        Converter_BTLMLMHeadModel_HF_CS21,
    ],
    "btlm-headless": [Converter_BTLMModel_HF_CS21],
    "clip-vit": [Converter_CLIPViT_Projection_HF_CS21],
    "codegen": [
        Converter_Codegen_LMHeadModel_CS18_CS20,
        Converter_Codegen_LMHeadModel_CS20_CS21,
        Converter_Codegen_LMHeadModel_HF_CS20,
    ],
    "codegen-headless": [
        Converter_Codegen_Headless_HF_CS20,
    ],
    "falcon": [
        Converter_Falcon_CS20_CS21,
        Converter_Falcon_HF_CS21,
    ],
    "falcon-headless": [
        Converter_Falcon_Headless_HF_CS21,
    ],
    "gpt2": [
        Converter_GPT2LMHeadModel_CS18_CS20,
        Converter_GPT2LMHeadModel_CS20_CS21,
        Converter_GPT2LMHeadModel_HF_CS21,
    ],
    "gpt2-headless": [
        Converter_GPT2Model_HF_CS21,
    ],
    "gptj": [
        Converter_GPTJ_LMHeadModel_CS18_CS20,
        Converter_GPTJ_LMHeadModel_CS20_CS21,
        Converter_GPTJ_LMHeadModel_HF_CS20,
    ],
    "gptj-headless": [
        Converter_GPTJ_Headless_HF_CS20,
    ],
    "gpt-neox": [
        Converter_GPT_Neox_LMHeadModel_CS18_CS20,
        Converter_GPT_Neox_LMHeadModel_CS20_CS21,
        Converter_GPT_Neox_LMHeadModel_HF_CS21,
    ],
    "gpt-neox-headless": [
        Converter_GPT_Neox_Headless_HF_CS21,
    ],
    "jais": [
        Converter_JAISLMHeadModel_CS20_CS21,
        Converter_JAISLMHeadModel_HF_CS21,
    ],
    "llama": [
        Converter_LlamaForCausalLM_CS19_CS20,
        Converter_LlamaForCausalLM_CS20_CS21,
        Converter_LlamaForCausalLM_HF_CS21,
    ],
    "llama-headless": [
        Converter_LlamaModel_HF_CS21,
    ],
    "llava": [Converter_LLaVA_HF_CS22],
    "mpt": [
        Converter_MPTForCausalLM_HF_CS21,
    ],
    "mpt-headless": [
        Converter_MPTModel_HF_CS21,
    ],
    "mistral": [Converter_MistralForCausalLM_HF_CS21],
    "mistral-headless": [Converter_MistralModel_HF_CS21],
    "roberta": [
        Converter_RobertaPretrainModel_HF_CS21,
        Converter_Bert_CS20_CS21,
    ],
    "santacoder": [Converter_SantacoderLMHeadModel_HF_CS21],
    "santacoder-headless": [Converter_SantacoderModel_HF_CS21],
    "starcoder": [
        Converter_StarcoderForCausalLM_HF_CS21,
        Converter_StarcoderLMHeadModel_CS20_CS21,
    ],
    "starcoder-headless": [
        Converter_StarcoderModel_HF_CS21,
    ],
    "t5": [
        Converter_T5_CS16_CS17,
        Converter_T5_CS16_CS18,
        Converter_T5_CS17_CS18,
        Converter_T5_CS18_CS20,
        Converter_T5_CS20_CS21,
        Converter_T5_HF_CS21,
    ],
}

# Add some model aliases
converters["ul2"] = converters["t5"]
converters["flan-ul2"] = converters["t5"]
converters["transformer"] = converters["t5"]
converters["llamaV2"] = converters["llama"]
converters["llamaV2-headless"] = converters["llama-headless"]
converters["code-llama"] = converters["llama"]
converters["code-llama-headless"] = converters["llama-headless"]
converters["octocoder"] = converters["starcoder"]
converters["octocoder-headless"] = converters["starcoder-headless"]
converters["wizardcoder"] = converters["starcoder"]
converters["wizardcoder-headless"] = converters["starcoder-headless"]
converters["sqlcoder"] = converters["starcoder"]
converters["sqlcoder-headless"] = converters["starcoder-headless"]
converters["wizardlm"] = converters["llama"]
converters["wizardlm-headless"] = converters["llama-headless"]


def _print_supported_models() -> None:
    print("The following models are supported:\n")
    print(
        tabulate(
            [[key] for key in sorted(converters)],
            headers=["model"],
            tablefmt="fancy_grid",
        )
    )


def _get_converter_notes(
    converter_class: BaseCheckpointConverter, width: Optional[int] = None
) -> str:
    if hasattr(converter_class, "converter_note"):
        note = converter_class.converter_note()
        if width is not None:
            note = textwrap.fill(note, width=width)
        return note
    else:
        return ""


def _print_supported_models_converters(
    model: Optional[str] = None, hide_notes: bool = False
) -> None:
    print("The following converters are supported:\n")
    table = []

    def _add_model_converters(table, model):
        oldest_version = _get_oldest_converter_version(model)
        existing_converters = []
        for converter in converters[model]:
            existing_converters.append(
                (converter.formats()[0], converter.formats()[1])
            )
            row = [
                model,
                "{}\n{}".format(converter.formats()[0], converter.formats()[1]),
                "{}\n{}".format(converter.formats()[1], converter.formats()[0]),
            ]
            if not hide_notes:
                row.append(_get_converter_notes(converter, width=60))
            table += [row]
        for converter in fallback_converters:
            # check if the version is older than the oldest version for this model
            version = _cs_version_to_float(converter.formats()[0][0])
            if version < oldest_version:
                continue
            # check if formats conversion already done without fallback
            for existing_converter in existing_converters:
                if (
                    converter.formats()[0][0] in existing_converter[0]
                    and converter.formats()[1][0] in existing_converter[1]
                ):
                    break
            else:
                row = [
                    model,
                    "{}\n{}".format(
                        converter.formats()[0], converter.formats()[1]
                    ),
                    "{}\n{}".format(
                        converter.formats()[1], converter.formats()[0]
                    ),
                ]
                if not hide_notes:
                    row.append(_get_converter_notes(converter, width=60))
                table += [row]

    if model is None:
        for model in sorted(converters):
            _add_model_converters(table, model)
    else:
        _add_model_converters(table, model)

    headers = ["model", "src-fmt", "tgt-fmt"]
    if not hide_notes:
        headers.append("notes")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def _cs_version_to_float(fmt: str) -> float:
    if "cs" not in fmt:
        return float("inf")
    groups = re.search(r"^cs\-(\d+\.\d+).*$", fmt).groups()
    assert len(groups) == 1
    return float(groups[0])


def _get_oldest_converter_version(model: str) -> float:
    oldest_version = float("inf")
    for converter in converters.get(model, []):
        for fmts in converter.formats():
            for fmt in fmts:
                version = _cs_version_to_float(fmt)
                if oldest_version > version:
                    oldest_version = version
    return oldest_version


def get_model_converter(
    model: str, src_fmt: str, tgt_fmt: str
) -> Optional[BaseCheckpointConverter]:
    if model in converters:
        for converter in converters[model]:
            if converter.supports_conversion(src_fmt, tgt_fmt):
                return converter
        # Get the oldest version for this model, but only if
        # the model is in the converters list
        oldest_version = _get_oldest_converter_version(model)
    elif model == "all":
        # If we only want model agnostic conversions, we don't need to check
        # the oldest version for a model
        oldest_version = -float("inf")
    else:
        return

    for converter in fallback_converters:
        # check if the version is older than the oldest version for this model
        version = _cs_version_to_float(converter.formats()[0][0])
        if version < oldest_version:
            continue
        if converter.supports_conversion(src_fmt, tgt_fmt):
            logging.warning(
                f"Checkpoint does not require changes between {src_fmt} and {tgt_fmt}. "
                f"Only updating checkpoint metadata and config."
            )
            return converter
    return None


def _select_model_and_config_converter(
    model: str, src_fmt: str, tgt_fmt: str
) -> Tuple[
    Optional[BaseCheckpointConverter],
    Optional[FormatIndices],
    Optional[BaseConfigConverter],
    Optional[FormatIndices],
]:
    converter_class = get_model_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
        print("Cannot convert", model, "from", src_fmt, "to", tgt_fmt)
        if model not in converters:
            _print_supported_models()
        else:
            _print_supported_models_converters(model)
        return None, None, None, None

    checkpoint_from_index = converter_class.get_converter_indices(
        src_fmt, tgt_fmt
    )
    assert (
        checkpoint_from_index is not None
    ), "Checkpoint converter {} supports format {} <-> {} but wanted to convert {} -> {}".format(
        converter_class.__name__, *converter_class.formats(), src_fmt, tgt_fmt
    )
    config_converter_class = converter_class.get_config_converter_class()
    config_from_index = config_converter_class.get_converter_indices(
        src_fmt, tgt_fmt
    )
    assert (
        config_from_index is not None
    ), "Config converter {} supports format {} <-> {} but wanted to convert {} -> {}".format(
        config_converter_class.__name__,
        *config_converter_class.formats(),
        src_fmt,
        tgt_fmt,
    )
    return (
        converter_class,
        checkpoint_from_index,
        config_converter_class,
        config_from_index,
    )


def _convert_checkpoint_helper(
    converter_class: BaseCheckpointConverter,
    checkpoint: dict,
    checkpoint_from_index: FormatIndices,
    config_converter_class: BaseConfigConverter,
    config: dict,
    config_from_index: FormatIndices,
    output_checkpoint: dict = {},
    drop_unmatched_keys: bool = False,
    no_progress_bar: bool = True,
    debug: bool = False,
) -> Tuple[dict, dict]:
    new_config = config_converter_class.convert(
        config,
        config_from_index,
        no_progress_bar=no_progress_bar,
        debug=debug,
        drop_unmatched_keys=True,
    )
    # Convert checkpoint:
    configs = (
        (config, new_config)
        if checkpoint_from_index.direction == 0
        else (new_config, config)
    )

    new_checkpoint = converter_class.convert(
        checkpoint,
        configs,
        checkpoint_from_index,
        output_checkpoint=output_checkpoint,
        drop_unmatched_keys=drop_unmatched_keys,
        no_progress_bar=no_progress_bar,
        debug=debug,
    )

    return new_checkpoint, new_config


def _get_cs_src_fmt_from_metadata(checkpoint: Union[str, dict]):
    if isinstance(checkpoint, str):
        checkpoint = cstorch.load(checkpoint)
    if isinstance(checkpoint, dict):
        if "__metadata__" in checkpoint:
            source_version = checkpoint["__metadata__"][-1]["version"]
            # convert to checkpoint converter format
            parsed_version = parse(source_version)
            source_version = f"cs-{parsed_version.major}.{parsed_version.minor}"
        else:
            raise ValueError(
                f"Metadata not found in checkpoint. Automatic "
                f"source format detection requires metadata "
                f"to be present in the checkpoint, which may not "
                f"be the case for checkpoints prior to 2.1 release "
                f"or for checkpoints that were saved through scripts "
                f"other than Cerebras ModelZoo. For such checkpoints "
                f"please provide an explicit source version to convert "
                f"from."
            )
    else:
        raise ValueError(
            f"Checkpoint must be passed as either a dict or str. "
            f"Got {type(checkpoint)} instead."
        )

    return source_version


def _get_cs_tgt_fmt_from_version():
    version = parse(cstorch.__version__)
    return f"cs-{version.major}.{version.minor}"


def _remove_file_extension(filename):
    """
    Returns a filename with *all* extensions removed.
    An extension is defined as a series of alphabetical chars followed by a dot.

    Concretely:
    checkpoint_1.7.mdl -> checkpoint_1.7
    pytorch_model.bin.index.json -> pytorch_model
    """
    reversed_file_name = filename[::-1]
    match = re.match(r"(?:[A-Za-z]+\.)*", reversed_file_name)
    extension_length = match.span()[1]
    return filename[:-extension_length]


def convert_checkpoint_from_file(
    model: str,
    src_fmt: str,
    tgt_fmt: str,
    checkpoint_file: str,
    config_file: str,
    outputdir: Optional[str] = None,
    hf_shard_size: str = "10GB",
    export_safetensors: bool = False,
    drop_unmatched_keys: bool = False,
    no_progress_bar: bool = True,
    debug: bool = False,
):
    if src_fmt == "cs-auto":
        src_fmt = _get_cs_src_fmt_from_metadata(checkpoint_file)
    if tgt_fmt == "cs-current":
        tgt_fmt = _get_cs_tgt_fmt_from_version()
    (
        converter_class,
        checkpoint_from_index,
        config_converter_class,
        config_from_index,
    ) = _select_model_and_config_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
        return None, None

    logging.info("Loading config & checkpoint...")
    config = config_converter_class.load(config_file, config_from_index)
    checkpoint = converter_class.load(checkpoint_file, checkpoint_from_index)

    if outputdir is not None and not os.path.exists(outputdir):
        os.makedirs(outputdir)

    checkpoint_folder, checkpoint_filename = os.path.split(checkpoint_file)
    new_checkpoint_filename_without_ext = (
        _remove_file_extension(checkpoint_filename) + "_to_" + tgt_fmt
    )

    new_checkpoint_file_without_ext = (
        os.path.join(outputdir, new_checkpoint_filename_without_ext)
        if outputdir is not None
        else os.path.join(
            checkpoint_folder, new_checkpoint_filename_without_ext
        )
    )

    output_checkpoint = converter_class.init_output_checkpoint(
        new_checkpoint_file_without_ext,
        checkpoint_from_index,
        hf_shard_size=hf_shard_size,
        export_safetensors=export_safetensors,
    )

    new_checkpoint, new_config = _convert_checkpoint_helper(
        converter_class,
        checkpoint,
        checkpoint_from_index,
        config_converter_class,
        config,
        config_from_index,
        output_checkpoint,
        drop_unmatched_keys,
        no_progress_bar,
        debug,
    )

    logging.info("Saving...")
    final_checkpoint_file = converter_class.save(
        new_checkpoint_file_without_ext,
        new_checkpoint,
        checkpoint_from_index,
    )

    config_folder, config_filename = os.path.split(config_file)
    new_config_filename_without_ext = (
        _remove_file_extension(config_filename) + "_to_" + tgt_fmt
    )

    if isinstance(output_checkpoint, StreamingShardedHFWriter):
        output_config_dir = final_checkpoint_file
        new_config_filename_without_ext = "config"
    elif outputdir is not None:
        output_config_dir = outputdir
    else:
        output_config_dir = config_folder

    new_config_file_without_ext = os.path.join(
        output_config_dir, new_config_filename_without_ext
    )

    final_config_file = config_converter_class.save(
        new_config_file_without_ext, new_config, config_from_index
    )

    return final_checkpoint_file, final_config_file


def convert_checkpoint(
    model: str,
    src_fmt: str,
    tgt_fmt: str,
    checkpoint: dict,
    config: str,
    output_checkpoint: Optional[str] = None,
    drop_unmatched_keys: bool = False,
    no_progress_bar: bool = True,
    debug: bool = False,
) -> Tuple[dict, dict]:
    if src_fmt == "cs-auto":
        src_fmt = _get_cs_src_fmt_from_metadata(checkpoint)
    if tgt_fmt == "cs-current":
        tgt_fmt = _get_cs_tgt_fmt_from_version()
    (
        converter_class,
        checkpoint_from_index,
        config_converter_class,
        config_from_index,
    ) = _select_model_and_config_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
        return None

    if output_checkpoint is None:
        output_checkpoint = {}

    return _convert_checkpoint_helper(
        converter_class,
        checkpoint,
        checkpoint_from_index,
        config_converter_class,
        config,
        config_from_index,
        output_checkpoint,
        drop_unmatched_keys,
        no_progress_bar,
        debug,
    )


def convert_config_from_file(
    model: str,
    src_fmt: str,
    tgt_fmt: str,
    config_file: str,
    outputdir: Optional[str] = None,
    drop_unmatched_keys: bool = False,
    no_progress_bar: bool = True,
    debug: bool = False,
) -> str:
    (
        converter_class,
        checkpoint_from_index,
        config_converter_class,
        config_from_index,
    ) = _select_model_and_config_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
        return None

    config = config_converter_class.load(config_file, config_from_index)
    new_config = config_converter_class.convert(
        config,
        config_from_index,
        drop_unmatched_keys=drop_unmatched_keys,
        no_progress_bar=no_progress_bar,
        debug=debug,
    )

    if outputdir is not None and not os.path.exists(outputdir):
        os.makedirs(outputdir)

    config_folder, config_filename = os.path.split(config_file)
    new_config_filename_without_ext = (
        os.path.splitext(config_filename)[0] + "_to_" + tgt_fmt
    )

    new_config_file_without_ext = (
        os.path.join(outputdir, new_config_filename_without_ext)
        if outputdir is not None
        else os.path.join(config_folder, new_config_filename_without_ext)
    )

    final_config_file = config_converter_class.save(
        new_config_file_without_ext, new_config, config_from_index
    )

    return final_config_file


def convert_config(
    model: str,
    src_fmt: str,
    tgt_fmt: str,
    config: dict,
    drop_unmatched_keys: bool = False,
    no_progress_bar: bool = True,
    debug: bool = False,
) -> dict:
    (
        converter_class,
        checkpoint_from_index,
        config_converter_class,
        config_from_index,
    ) = _select_model_and_config_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
        return None

    new_config = config_converter_class.convert(config, config_from_index)

    return new_config


TENSOR_CMP_SUPPORTED_OPS = ["equal", "allclose"]


def diff_checkpoints_from_file(
    file_left: str, file_right: str, tensor_comparison_op: str = "equal"
) -> bool:
    """
    Compare two checkpoints (left and right). Returns True if the dicts are the
    same.
    """
    file_left_exists, file_right_exists = (
        os.path.exists(file_left),
        os.path.exists(file_right),
    )
    if not file_left_exists:
        print("No such file: {}".format(file_left))
        return False
    if not file_right_exists:
        print("No such file: {}".format(file_right))
        return False
    if file_left_exists and file_right_exists:
        import cerebras.pytorch as cstorch

        print("Loading checkpoints...")
        checkpoint_left = cstorch.load(file_left)
        checkpoint_right = cstorch.load(file_right)
        print("Comparing checkpoints...")
        return diff_checkpoints(
            checkpoint_left,
            checkpoint_right,
            tensor_comparison_op=tensor_comparison_op,
        )


def diff_checkpoints(
    checkpoint_left: dict,
    checkpoint_right: dict,
    tensor_comparison_op: str = "equal",
) -> bool:
    """
    Compare state dictionaries of two checkpoints (left and right). Returns True
    if the dicts are the same. Tensors can be compared via the "equal" or
    "allclose" operators. All other types are compared for strict equality.
    """
    import torch

    def format_keys(key_path):
        return ".".join([str(e) for e in key_path])

    def diff_dict(dict_left, dict_right, prefix=[]):
        different = False
        keys_in_left_not_right = set(dict_left.keys()) - set(dict_right.keys())
        if len(keys_in_left_not_right) != 0:
            print(
                "The following keys are in the left checkpoint but not right:"
            )
            print(
                [
                    format_keys(prefix + [missing_key])
                    for missing_key in keys_in_left_not_right
                ]
            )
            different = True
        keys_in_right_not_left = set(dict_right.keys()) - set(dict_left.keys())
        if len(keys_in_right_not_left) != 0:
            print(
                "The following keys are in the right checkpoint but not left:"
            )
            print(
                [
                    format_keys(prefix + [missing_key])
                    for missing_key in keys_in_right_not_left
                ]
            )
            different = True
        keys_in_left_and_right = set(dict_left.keys()) & set(dict_right.keys())

        for key in keys_in_left_and_right:
            full_key_formatted = format_keys(prefix + [key])
            if isinstance(dict_left[key], dict) and isinstance(
                dict_right[key], dict
            ):
                subdict_is_different = diff_dict(
                    dict_left[key], dict_right[key], prefix=prefix + [key]
                )
                different = different or subdict_is_different
            elif type(dict_left[key]) != type(dict_right[key]):
                print(
                    "{} has type {} in left and type {} in right".format(
                        full_key_formatted,
                        type(dict_left[key]),
                        type(dict_right[key]),
                    )
                )
                different = True
            elif isinstance(dict_left[key], torch.Tensor):
                if dict_left[key].shape != dict_right[key].shape:
                    print(
                        "{} left tensor has shape {} while right has shape {}".format(
                            full_key_formatted,
                            dict_left[key].shape,
                            dict_right[key].shape,
                        )
                    )
                    different = True
                elif tensor_comparison_op == "equal":
                    if not torch.equal(dict_left[key], dict_right[key]):
                        print(
                            "{} left tensor is not equal to right".format(
                                full_key_formatted
                            )
                        )
                        different = True
                elif tensor_comparison_op == "close":
                    if not torch.allclose(dict_left[key], dict_right[key]):
                        print(
                            "{} left tensor is not close to right".format(
                                full_key_formatted
                            )
                        )
                        different = True
            else:
                if dict_left[key] != dict_right[key]:
                    print(
                        "{} is {} in left and {} in right".format(
                            full_key_formatted, dict_left[key], dict_right[key]
                        )
                    )
                    different = True
        return different

    assert (
        tensor_comparison_op in TENSOR_CMP_SUPPORTED_OPS
    ), "{} is not a supported tensor comparison operation. Please select one of the following: {}".format(
        tensor_comparison_op, TENSOR_CMP_SUPPORTED_OPS
    )
    assert isinstance(
        checkpoint_left, dict
    ), "Expecting left checkpoint to be a state dict"
    assert isinstance(
        checkpoint_right, dict
    ), "Expecting right checkpoint to be a state dict"
    different = diff_dict(checkpoint_left, checkpoint_right)
    print()
    print("Checkpoints {}".format("differ" if different else "are the same "))
    return not different


class CheckpointConverterCLI(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Cerebras Pytorch Checkpoint Converter Tool',
            usage='''python convert_checkpoint.py <command> [<args>]

The following commands are supported:
   convert          Convert a checkpoint & config
   convert-config   Convert a model config file only
   list             List supported checkpoint conversion formats
   diff             Compare two checkpoints
''',
        )
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        fn_name = "_{}".format(args.command.replace("-", "_"))

        if not hasattr(self, fn_name):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)

        logging.getLogger().setLevel(logging.INFO)
        # use dispatch pattern to invoke method with same name
        getattr(self, fn_name)()

    def _convert(self):
        parser = argparse.ArgumentParser(
            description='Convert a checkpoint & the corresponding config'
        )
        parser.add_argument(
            'checkpoint_file',
            metavar='checkpoint-file',
            type=str,
            help='Checkpoint file to convert (ex: .bin or .mdl file). For sharded HuggingFace checkpoints, provide the .index.json file instead.',
        )

        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Name of model. For options, run `python convert_checkpoint.py list`.',
        )

        parser.add_argument(
            '--src-fmt',
            type=str,
            required=True,
            help=(
                'Format of input. Can be "cs-X.X" (i.e. cs-2.0) for a Cerebras version type, '
                '"cs-auto" to detect Cerebras version from checkpoint, or "hf" for '
                'HuggingFace Models'
            ),
        )

        parser.add_argument(
            '--tgt-fmt',
            type=str,
            required=True,
            help=(
                'Format of output. Can be "cs-X.X" (i.e. cs-2.0) for a Cerebras version type, '
                '"cs-current" to specify the current release, or "hf" for HuggingFace Models.'
            ),
        )

        parser.add_argument(
            '--config',
            type=str,
            required=True,
            help='Config file corresponding to checkpoint',
        )

        parser.add_argument(
            '--output-dir',
            type=str,
            help='Output directory. Default: directory of input checkpoint/config',
        )

        hf_shard_size_default = "10GB"

        parser.add_argument(
            '--hf-shard-size',
            default=hf_shard_size_default,
            type=str,
            help=f'Size of HuggingFace checkpoint shards. Default: \
                   {hf_shard_size_default}. Must be of the format integer \
                   followed by unit. The following units are supported: GB \
                   (gigabyte), GiB (gibibyte), MB (megabyte), MiB (mebibyte), \
                   KB (kilobyte), and KIB (kibibyte)',
        )

        parser.add_argument(
            '--export-safetensors',
            action='store_true',
            help='When enabled, the output checkpoints will be stored as \
                  safetensors rather pickle files. This flag should only be \
                  used when converting to the Hugging Face format.',
        )

        parser.add_argument(
            '--drop-unmatched-keys',
            action='store_true',
            help="Ignore (drop) keys that aren't matched during conversion. Note that this will lead to a partially converted checkpoint.",
        )

        parser.add_argument(
            '--no-progress-bar',
            action='store_true',
            help='Disable progress bar',
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            help='Debug checkpoint key mapping',
        )

        args = parser.parse_args(sys.argv[2:])

        (
            checkpoint_output_path,
            config_output_path,
        ) = convert_checkpoint_from_file(
            args.model,
            args.src_fmt,
            args.tgt_fmt,
            args.checkpoint_file,
            args.config,
            args.output_dir,
            args.hf_shard_size,
            args.export_safetensors,
            args.drop_unmatched_keys,
            args.no_progress_bar,
            args.debug,
        )

        if checkpoint_output_path is None or config_output_path is None:
            print("\nConversion failed.")
            sys.exit(1)
        else:
            print("Checkpoint saved to {}".format(checkpoint_output_path))
            print("Config saved to {}".format(config_output_path))

    def _convert_config(self):
        parser = argparse.ArgumentParser(description='Convert config')
        parser.add_argument(
            'config_file',
            metavar='config-file',
            type=str,
            help='File to convert',
        )

        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Name of model. For options, run `python convert_checkpoint.py list`.',
        )

        parser.add_argument(
            '--src-fmt',
            type=str,
            required=True,
            help='Format of input. Can be "hf" for HuggingFace Models, or "cs-X.X" (i.e. cs-2.0) for a Cerebras version type.',
        )

        parser.add_argument(
            '--tgt-fmt',
            type=str,
            required=True,
            help='Format of output. Can be "hf" for HuggingFace Models, or "cs-X.X" (i.e. cs-2.0) for a Cerebras version type.',
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            help='Output directory. Default: directory of input config',
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            help='Debug config key mapping',
        )

        args = parser.parse_args(sys.argv[2:])

        config_output_path = convert_config_from_file(
            args.model,
            args.src_fmt,
            args.tgt_fmt,
            args.config_file,
            outputdir=args.output_dir,
            debug=args.debug,
            drop_unmatched_keys=True,
        )

        if config_output_path is None:
            print("\nConversion failed.")
            sys.exit(1)
        else:
            print("Config saved to {}".format(config_output_path))

    def _list(self):
        parser = argparse.ArgumentParser(
            description='List supported checkpoint conversion formats'
        )
        parser.add_argument(
            'model',
            type=str.lower,
            default="all",
            nargs='?',
            help="Either MODEL to list supported converters for a particular model or 'all' to list all converters",
        )
        parser.add_argument(
            '--hide-notes',
            action='store_true',
            help='Hide notes column',
        )
        args = parser.parse_args(sys.argv[2:])

        if args.model == "all":
            _print_supported_models_converters(hide_notes=args.hide_notes)
        elif args.model in converters:
            _print_supported_models_converters(
                args.model, hide_notes=args.hide_notes
            )
        else:
            print("The model {} is not supported.".format(args.model))
            _print_supported_models()
            sys.exit(1)

    def _diff(self):
        parser = argparse.ArgumentParser(description='Compare two checkpoints')
        parser.add_argument(
            'left_checkpoint',
            type=str,
            help="Path to left checkpoint",
        )
        parser.add_argument(
            'right_checkpoint',
            type=str,
            help="Path to right checkpoint",
        )
        parser.add_argument(
            '--tensor_comparison_op',
            choices=TENSOR_CMP_SUPPORTED_OPS,
            default=TENSOR_CMP_SUPPORTED_OPS[0],
        )

        args = parser.parse_args(sys.argv[2:])
        diff_checkpoints_from_file(
            args.left_checkpoint,
            args.right_checkpoint,
            tensor_comparison_op=args.tensor_comparison_op,
        )


if __name__ == '__main__':
    CheckpointConverterCLI()
