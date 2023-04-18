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

converters = {
    "bert": {
        Converter_BertPretrainModel_HF_CS17.formats(): Converter_BertPretrainModel_HF_CS17,
        Converter_BertPretrainModel_HF_CS18.formats(): Converter_BertPretrainModel_HF_CS18,
        Converter_BertPretrainModel_CS16_CS17.formats(): Converter_BertPretrainModel_CS16_CS17,
        Converter_BertPretrainModel_CS16_CS18.formats(): Converter_BertPretrainModel_CS16_CS18,
        Converter_Bert_CS17_CS18.formats(): Converter_Bert_CS17_CS18,
    },
    "bert-sequence-classifier": {
        Converter_BertFinetuneModel_CS16_CS17.formats(): Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18.formats(): Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18.formats(): Converter_Bert_CS17_CS18,
        Converter_BertForSequenceClassification_HF_CS17.formats(): Converter_BertForSequenceClassification_HF_CS17,
        Converter_BertForSequenceClassification_HF_CS18.formats(): Converter_BertForSequenceClassification_HF_CS18,
    },
    "bert-token-classifier": {
        Converter_BertFinetuneModel_CS16_CS17.formats(): Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18.formats(): Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18.formats(): Converter_Bert_CS17_CS18,
        Converter_BertForTokenClassification_HF_CS17.formats(): Converter_BertForTokenClassification_HF_CS17,
        Converter_BertForTokenClassification_HF_CS18.formats(): Converter_BertForTokenClassification_HF_CS18,
    },
    "bert-summarization": {
        Converter_BertFinetuneModel_CS16_CS17.formats(): Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18.formats(): Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18.formats(): Converter_Bert_CS17_CS18,
    },
    "bert-q&a": {
        Converter_BertFinetuneModel_CS16_CS17.formats(): Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18.formats(): Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18.formats(): Converter_Bert_CS17_CS18,
        Converter_BertForQuestionAnswering_HF_CS17.formats(): Converter_BertForQuestionAnswering_HF_CS17,
        Converter_BertForQuestionAnswering_HF_CS18.formats(): Converter_BertForQuestionAnswering_HF_CS18,
    },
    "codegen": {
        Converter_Codegen_LMHeadModel_HF_CS17.formats(): Converter_Codegen_LMHeadModel_HF_CS17,
        Converter_Codegen_LMHeadModel_HF_CS18.formats(): Converter_Codegen_LMHeadModel_HF_CS18,
    },
    "codegen-headless": {
        Converter_Codegen_Headless_HF_CS17.formats(): Converter_Codegen_Headless_HF_CS17,
        Converter_Codegen_Headless_HF_CS18.formats(): Converter_Codegen_Headless_HF_CS18,
    },
    "gpt2": {
        Converter_GPT2LMHeadModel_HF_CS17.formats(): Converter_GPT2LMHeadModel_HF_CS17,
        Converter_GPT2LMHeadModel_HF_CS18.formats(): Converter_GPT2LMHeadModel_HF_CS18,
    },
    "gpt2-headless": {
        Converter_GPT2Model_HF_CS17.formats(): Converter_GPT2Model_HF_CS17,
        Converter_GPT2Model_HF_CS18.formats(): Converter_GPT2Model_HF_CS18,
    },
    "gptj": {
        Converter_GPTJ_LMHeadModel_HF_CS17.formats(): Converter_GPTJ_LMHeadModel_HF_CS17,
        Converter_GPTJ_LMHeadModel_HF_CS18.formats(): Converter_GPTJ_LMHeadModel_HF_CS18,
    },
    "gptj-headless": {
        Converter_GPTJ_Headless_HF_CS17.formats(): Converter_GPTJ_Headless_HF_CS17,
        Converter_GPTJ_Headless_HF_CS18.formats(): Converter_GPTJ_Headless_HF_CS18,
    },
    "gpt-neox": {
        Converter_GPT_Neox_LMHeadModel_HF_CS17.formats(): Converter_GPT_Neox_LMHeadModel_HF_CS17,
        Converter_GPT_Neox_LMHeadModel_HF_CS18.formats(): Converter_GPT_Neox_LMHeadModel_HF_CS18,
    },
    "gpt-neox-headless": {
        Converter_GPT_Neox_Headless_HF_CS17.formats(): Converter_GPT_Neox_Headless_HF_CS17,
        Converter_GPT_Neox_Headless_HF_CS18.formats(): Converter_GPT_Neox_Headless_HF_CS18,
    },
    "t5": {
        Converter_T5_CS16_CS17.formats(): Converter_T5_CS16_CS17,
        Converter_T5_CS16_CS18.formats(): Converter_T5_CS16_CS18,
        Converter_T5_CS17_CS18.formats(): Converter_T5_CS17_CS18,
        Converter_T5_HF_CS17.formats(): Converter_T5_HF_CS17,
        Converter_T5_HF_CS18.formats(): Converter_T5_HF_CS18,
    },
    "transformer": {  # Transformer model shares same codebase as T5
        Converter_T5_CS16_CS17.formats(): Converter_T5_CS16_CS17,
        Converter_T5_CS16_CS18.formats(): Converter_T5_CS16_CS18,
        Converter_T5_CS17_CS18.formats(): Converter_T5_CS17_CS18,
    },
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


def _get_converter_notes(converter_class):
    if hasattr(converter_class.formats, "notes"):
        return converter_class.formats.notes
    else:
        return ""


def _print_supported_models_converters(model=None, hide_notes=False):
    print("The following converters are supported:\n")
    table = []

    def _add_model_converters(table, model):
        for key, converter in converters[model].items():
            row = [
                model,
                "{}\n{}".format(key[0], key[1]),
                "{}\n{}".format(key[1], key[0]),
            ]
            if not hide_notes:
                row.append(_get_converter_notes(converter))
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
    if (src_fmt, tgt_fmt) in converters[model]:
        return converters[model][(src_fmt, tgt_fmt)]
    elif (tgt_fmt, src_fmt) in converters[model]:
        return converters[model][(tgt_fmt, src_fmt)]
    return None


def _select_model_and_config_converter(model, src_fmt, tgt_fmt):
    if model not in converters:
        _print_supported_models()
        return None, None, None, None
    converter_class = _get_model_converter(model, src_fmt, tgt_fmt)
    if converter_class is None:
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


class CheckpointConverterCLI(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Cerebras Pytorch Checkpoint Converter Tool',
            usage='''python convert_checkpoint.py <command> [<args>]

The following commands are supported:
   convert          Convert a checkpoint & config
   convert-config   Convert a model config file only
   list             List supported checkpoint conversion formats
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


if __name__ == '__main__':
    CheckpointConverterCLI()
