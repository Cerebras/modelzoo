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
import sys
import textwrap

from tabulate import tabulate

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from modelzoo.common.pytorch.model_utils.checkpoint_converters.bert import (  # To CS 1.7; To CS 1.8
    Converter_Bert_CS17_CS18,
    Converter_BertPretrainModel_CS16_CS17,
    Converter_BertPretrainModel_CS16_CS18,
    Converter_BertPretrainModel_HF_CS17,
    Converter_BertPretrainModel_HF_CS18,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.bert_finetune import (  # To CS 1.7; To CS 1.8
    Converter_BertFinetuneModel_CS16_CS17,
    Converter_BertFinetuneModel_CS16_CS18,
    Converter_BertForQuestionAnswering_HF_CS17,
    Converter_BertForQuestionAnswering_HF_CS18,
    Converter_BertForSequenceClassification_HF_CS17,
    Converter_BertForSequenceClassification_HF_CS18,
    Converter_BertForTokenClassification_HF_CS17,
    Converter_BertForTokenClassification_HF_CS18,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.gpt2_hf_cs import (  # To CS 1.7; To CS 1.8
    Converter_GPT2LMHeadModel_HF_CS17,
    Converter_GPT2LMHeadModel_HF_CS18,
    Converter_GPT2Model_HF_CS17,
    Converter_GPT2Model_HF_CS18,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.gpt_neox_hf_cs import (  # To CS 1.7; To CS 1.8
    Converter_GPT_Neox_Headless_HF_CS17,
    Converter_GPT_Neox_Headless_HF_CS18,
    Converter_GPT_Neox_LMHeadModel_HF_CS17,
    Converter_GPT_Neox_LMHeadModel_HF_CS18,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.gptj_hf_cs import (  # To CS 1.7; To CS 1.8
    Converter_GPTJ_Headless_HF_CS17,
    Converter_GPTJ_Headless_HF_CS18,
    Converter_GPTJ_LMHeadModel_HF_CS17,
    Converter_GPTJ_LMHeadModel_HF_CS18,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.llama import (  # To CS 1.9
    Converter_LlamaForCausalLM_HF_CS19,
    Converter_LlamaModel_HF_CS19,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.salesforce_codegen_hf_cs import (  # To CS 1.7; To CS 1.8
    Converter_Codegen_Headless_HF_CS17,
    Converter_Codegen_Headless_HF_CS18,
    Converter_Codegen_LMHeadModel_HF_CS17,
    Converter_Codegen_LMHeadModel_HF_CS18,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.t5 import (  # To CS 1.7; To CS 1.8
    Converter_T5_CS16_CS17,
    Converter_T5_CS16_CS18,
    Converter_T5_CS17_CS18,
    Converter_T5_HF_CS17,
    Converter_T5_HF_CS18,
)

from modelzoo.common.pytorch.model_utils.checkpoint_converters.falcon import (  # noqa
    Converter_Falcon_Headless_HF_CS19,
    Converter_Falcon_HF_CS19,
)

from modelzoo.common.pytorch.model_utils.checkpoint_converters.bloom_hf_cs import (  # noqa
    Converter_BloomLMHeadModel_HF_CS19,
    Converter_BloomModel_HF_CS19,
)


converters = {
    "bert": [
        Converter_BertPretrainModel_HF_CS17,
        Converter_BertPretrainModel_HF_CS18,
        Converter_BertPretrainModel_CS16_CS17,
        Converter_BertPretrainModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
    ],
    "bert-sequence-classifier": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_BertForSequenceClassification_HF_CS17,
        Converter_BertForSequenceClassification_HF_CS18,
    ],
    "bert-token-classifier": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_BertForTokenClassification_HF_CS17,
        Converter_BertForTokenClassification_HF_CS18,
    ],
    "bert-summarization": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
    ],
    "bert-q&a": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_BertForQuestionAnswering_HF_CS17,
        Converter_BertForQuestionAnswering_HF_CS18,
    ],
    "bloom": [
        Converter_BloomLMHeadModel_HF_CS19,
    ],
    "bloom-headless": [
        Converter_BloomModel_HF_CS19,
    ],
    "codegen": [
        Converter_Codegen_LMHeadModel_HF_CS17,
        Converter_Codegen_LMHeadModel_HF_CS18,
    ],
    "codegen-headless": [
        Converter_Codegen_Headless_HF_CS17,
        Converter_Codegen_Headless_HF_CS18,
    ],
    "gpt2": [
        Converter_GPT2LMHeadModel_HF_CS17,
        Converter_GPT2LMHeadModel_HF_CS18,
    ],
    "gpt2-headless": [
        Converter_GPT2Model_HF_CS17,
        Converter_GPT2Model_HF_CS18,
    ],
    "gptj": [
        Converter_GPTJ_LMHeadModel_HF_CS17,
        Converter_GPTJ_LMHeadModel_HF_CS18,
    ],
    "gptj-headless": [
        Converter_GPTJ_Headless_HF_CS17,
        Converter_GPTJ_Headless_HF_CS18,
    ],
    "gpt-neox": [
        Converter_GPT_Neox_LMHeadModel_HF_CS17,
        Converter_GPT_Neox_LMHeadModel_HF_CS18,
    ],
    "gpt-neox-headless": [
        Converter_GPT_Neox_Headless_HF_CS17,
        Converter_GPT_Neox_Headless_HF_CS18,
    ],
    "llama": [Converter_LlamaForCausalLM_HF_CS19,],
    "llama-headless": [Converter_LlamaModel_HF_CS19,],
    "t5": [
        Converter_T5_CS16_CS17,
        Converter_T5_CS16_CS18,
        Converter_T5_CS17_CS18,
        Converter_T5_HF_CS17,
        Converter_T5_HF_CS18,
    ],
    "transformer": [  # Transformer model shares same codebase as T5
        Converter_T5_CS16_CS17,
        Converter_T5_CS16_CS18,
        Converter_T5_CS17_CS18,
    ],
    "falcon": [Converter_Falcon_HF_CS19],
    "falcon-headless": [Converter_Falcon_Headless_HF_CS19],
}


def _print_supported_models():
    print("The following models are supported:\n")
    print(
        tabulate(
            [[key] for key in converters],
            headers=["model"],
            tablefmt="fancy_grid",
        )
    )


def _get_converter_notes(converter_class, width=None):
    if hasattr(converter_class, "converter_note"):
        note = converter_class.converter_note()
        if width is not None:
            note = textwrap.fill(note, width=width)
        return note
    else:
        return ""


def _print_supported_models_converters(model=None, hide_notes=False):
    print("The following converters are supported:\n")
    table = []

    def _add_model_converters(table, model):
        for converter in converters[model]:
            row = [
                model,
                "{}\n{}".format(converter.formats()[0], converter.formats()[1]),
                "{}\n{}".format(converter.formats()[1], converter.formats()[0]),
            ]
            if not hide_notes:
                row.append(_get_converter_notes(converter, width=70))
            table += [row]

    if model is None:
        for model in converters:
            _add_model_converters(table, model)
    else:
        _add_model_converters(table, model)
    # TODO: upgrade tabulate package so that we can use 'maxcolwidths' argument
    headers = ["model", "src-fmt", "tgt-fmt"]
    if not hide_notes:
        headers.append("notes")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def _get_model_converter(model, src_fmt, tgt_fmt):
    for converter in converters[model]:
        if converter.supports_conversion(src_fmt, tgt_fmt):
            return converter
    return None


def _select_model_and_config_converter(model, src_fmt, tgt_fmt):
    if model not in converters:
        _print_supported_models()
        return None, None, None, None
    converter_class = _get_model_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
        print("Cannot convert from ", src_fmt, "to", tgt_fmt)
        _print_supported_models_converters(model)
        return None, None, None, None

    checkpoint_from_index = converter_class.get_from_index(src_fmt, tgt_fmt)
    assert (
        checkpoint_from_index is not None
    ), "Checkpoint converter {} supports format {} <-> {} but wanted to convert {} -> {}".format(
        converter_class.__name__, *converter_class.formats(), src_fmt, tgt_fmt
    )
    config_converter_class = converter_class.get_config_converter_class()
    config_from_index = config_converter_class.get_from_index(src_fmt, tgt_fmt)
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
    converter_class,
    checkpoint,
    checkpoint_from_index,
    config_converter_class,
    config,
    config_from_index,
    drop_unmatched_keys=False,
    no_progress_bar=True,
    debug=False,
):
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
        if checkpoint_from_index == 0
        else (new_config, config)
    )
    new_checkpoint = converter_class.convert(
        checkpoint,
        configs,
        checkpoint_from_index,
        drop_unmatched_keys=drop_unmatched_keys,
        no_progress_bar=no_progress_bar,
        debug=debug,
    )

    return new_checkpoint, new_config


def convert_checkpoint_from_file(
    model,
    src_fmt,
    tgt_fmt,
    checkpoint_file,
    config_file,
    outputdir=None,
    export_h5_checkpoint=False,
    drop_unmatched_keys=False,
    no_progress_bar=True,
    debug=False,
):

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

    new_checkpoint, new_config = _convert_checkpoint_helper(
        converter_class,
        checkpoint,
        checkpoint_from_index,
        config_converter_class,
        config,
        config_from_index,
        drop_unmatched_keys,
        no_progress_bar,
        debug,
    )

    if outputdir is not None and not os.path.exists(outputdir):
        os.makedirs(outputdir)

    checkpoint_folder, checkpoint_filename = os.path.split(checkpoint_file)
    new_checkpoint_filename_without_ext = (
        os.path.splitext(checkpoint_filename)[0] + "_to_" + tgt_fmt
    )
    new_checkpoint_file_without_ext = (
        os.path.join(outputdir, new_checkpoint_filename_without_ext)
        if outputdir is not None
        else os.path.join(
            checkpoint_folder, new_checkpoint_filename_without_ext
        )
    )

    logging.info("Saving...")
    final_checkpoint_file = converter_class.save(
        new_checkpoint_file_without_ext,
        new_checkpoint,
        checkpoint_from_index,
        export_h5_checkpoint=export_h5_checkpoint,
    )

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

    return final_checkpoint_file, final_config_file


def convert_checkpoint(
    model,
    src_fmt,
    tgt_fmt,
    checkpoint,
    config,
    drop_unmatched_keys=False,
    no_progress_bar=True,
    debug=False,
):
    (
        converter_class,
        checkpoint_from_index,
        config_converter_class,
        config_from_index,
    ) = _select_model_and_config_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
        return None

    return _convert_checkpoint_helper(
        converter_class,
        checkpoint,
        checkpoint_from_index,
        config_converter_class,
        config,
        config_from_index,
        drop_unmatched_keys,
        no_progress_bar,
        debug,
    )


def convert_config_from_file(
    model,
    src_fmt,
    tgt_fmt,
    config_file,
    outputdir=None,
    drop_unmatched_keys=False,
    no_progress_bar=True,
    debug=False,
):

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
    model,
    src_fmt,
    tgt_fmt,
    config,
    drop_unmatched_keys=False,
    no_progress_bar=True,
    debug=False,
):
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
    file_left, file_right, tensor_comparison_op="equal"
):
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
        from modelzoo.common.pytorch import cbtorch

        print("Loading checkpoints...")
        checkpoint_left = cbtorch.load(file_left)
        checkpoint_right = cbtorch.load(file_right)
        print("Comparing checkpoints...")
        return diff_checkpoints(
            checkpoint_left,
            checkpoint_right,
            tensor_comparison_op=tensor_comparison_op,
        )


def diff_checkpoints(
    checkpoint_left, checkpoint_right, tensor_comparison_op="equal"
):
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
        parser = argparse.ArgumentParser(description='Convert checkpoint')
        parser.add_argument(
            'checkpoint_file',
            metavar='checkpoint-file',
            type=str,
            help='Checkpoint file to convert',
        )

        parser.add_argument(
            '--model', type=str, required=True, help='Type of model',
        )

        parser.add_argument(
            '--src-fmt', type=str, required=True, help='Format of input',
        )

        parser.add_argument(
            '--tgt-fmt', type=str, required=True, help='Format of output',
        )
        parser.add_argument(
            '--config', type=str, required=True, help='Config file to convert',
        )

        parser.add_argument(
            '--output-dir', type=str, help='Output directory',
        )

        parser.add_argument(
            '--export-h5-checkpoint',
            action='store_true',
            help='If enabled, store the output checkpoint in H5 format instead\
            of the standard pytorch pickle format. Using this format is recommended\
             when converting between/to CS models as it is faster.',
        )

        parser.add_argument(
            '--drop-unmatched-keys',
            action='store_true',
            help='Output directory',
        )

        parser.add_argument(
            '--no-progress-bar',
            action='store_true',
            help='Disable progress bar',
        )

        parser.add_argument(
            '--debug', action='store_true', help='Debug checkpoint key mapping',
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
            args.export_h5_checkpoint,
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
            '--model', type=str, required=True, help='Type of model',
        )

        parser.add_argument(
            '--src-fmt', type=str, required=True, help='Format of input',
        )

        parser.add_argument(
            '--tgt-fmt', type=str, required=True, help='Format of output',
        )
        parser.add_argument(
            '--output-dir', type=str, help='Output directory',
        )

        parser.add_argument(
            '--debug', action='store_true', help='Debug config key mapping',
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
            help="Either MODEL to list supported converters for a paritcular model or 'all' to list all converters",
        )
        parser.add_argument(
            '--hide-notes', action='store_true', help='Hide notes column',
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
            'left_checkpoint', type=str, help="Path to left checkpoint",
        )
        parser.add_argument(
            'right_checkpoint', type=str, help="Path to right checkpoint",
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
