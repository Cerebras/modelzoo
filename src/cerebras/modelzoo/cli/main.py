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

# PYTHON_ARGCOMPLETE_OK
import argparse

import argcomplete


class ModelZooCLI:
    def __init__(self):
        from cerebras.modelzoo.cli.assistant_cli import AssistantCLI
        from cerebras.modelzoo.cli.config_mgmt_cli import ConfigMgmtCLI
        from cerebras.modelzoo.cli.data_info_cli import DataInfoCLI
        from cerebras.modelzoo.cli.data_preprocess_cli import DataPreprocessCLI
        from cerebras.modelzoo.cli.model_info_cli import ModelInfoCLI
        from cerebras.modelzoo.cli.utils import EPILOG, add_run_args
        from cerebras.modelzoo.tools.convert_checkpoint import (
            CheckpointConverterCLI,
        )

        parser = argparse.ArgumentParser(
            description=(
                "Cerebras ModelZoo CLI. This serves as a single entry-point to all ModelZoo "
                "related tasks including: training and validation, checkpoint conversion, "
                "data preprocessing and config management."
            ),
            epilog=EPILOG,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="cmd", required=True)

        #######
        # fit #
        #######
        fit_parser = subparsers.add_parser(
            "fit",
            help=(
                "Run a model by calling fit. This completes a full training run on the given "
                "train and validation dataloaders."
            ),
            epilog=(
                "For more information on how models are trained, see: "
                "https://docs.cerebras.net/en/latest/wsc/Model-zoo/trainer-overview.html"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        seen_args = add_run_args(fit_parser)
        fit_parser.set_defaults(
            func=ModelZooCLI.run_trainer,
            mode="train_and_eval",
            seen_args=seen_args,
        )

        ############
        # validate #
        ############
        validate_parser = subparsers.add_parser(
            "validate",
            help=(
                "Run a model by calling validate. This completes a full validation run on "
                "the specified validation dataloader."
            ),
            epilog=(
                "For more information on how models are validated, see: "
                "https://docs.cerebras.net/en/latest/wsc/Model-zoo/trainer-overview.html"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        seen_args = add_run_args(validate_parser)
        validate_parser.set_defaults(
            func=ModelZooCLI.run_trainer, mode="eval", seen_args=seen_args
        )

        ################
        # validate_all #
        ################
        validate_all_parser = subparsers.add_parser(
            "validate_all",
            help=(
                "Run a model by calling validate_all. This runs all upstream and downstream "
                "validation permutations."
            ),
            epilog=(
                "For more information on how models are validated, see: "
                "https://docs.cerebras.net/en/latest/wsc/Model-zoo/trainer-overview.html"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        seen_args = add_run_args(validate_all_parser)
        validate_all_parser.set_defaults(
            func=ModelZooCLI.run_trainer, mode="eval_all", seen_args=seen_args
        )

        ##############
        # checkpoint #
        ##############
        converter_parser = subparsers.add_parser(
            "checkpoint",
            help="Convert checkpoint and/or config to a different format.",
            epilog=CheckpointConverterCLI.epilog(),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        CheckpointConverterCLI.configure_parser(converter_parser)

        ###################
        # data_preprocess #
        ###################
        preprocess_parser = subparsers.add_parser(
            "data_preprocess",
            help="Run data preprocessing.",
            epilog=DataPreprocessCLI.epilog(),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        DataPreprocessCLI.configure_parser(preprocess_parser)

        #########
        # model #
        #########
        model_parser = subparsers.add_parser(
            "model",
            help="Get information on available models.",
            epilog=ModelInfoCLI.epilog(),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        ModelInfoCLI.configure_parser(model_parser)

        ########
        # data #
        ########
        data_parser = subparsers.add_parser(
            "data_processor",
            help="Get information on available data processors.",
            epilog=DataInfoCLI.epilog(),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        DataInfoCLI.configure_parser(data_parser)

        ##########
        # config #
        ##########
        config_parser = subparsers.add_parser(
            "config",
            help="Manage model config files.",
            epilog=ConfigMgmtCLI.epilog(),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        ConfigMgmtCLI.configure_parser(config_parser)

        #############
        # assistant #
        #############
        assistant_parser = subparsers.add_parser(
            "assistant",
            help="LLM assistant for cszoo",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        AssistantCLI.configure_parser(assistant_parser)

        # To enable autocomplete on bash, must run command:
        # eval "$(register-python-argcomplete <script-name>)"
        argcomplete.autocomplete(parser)
        args = parser.parse_args()
        args.func(args)

    @staticmethod
    def run_trainer(args):
        from cerebras.modelzoo.cli.utils import _args_to_params
        from cerebras.modelzoo.trainer.utils import run_trainer

        params = _args_to_params(args)

        run_trainer(args.mode, params)


def main():
    ModelZooCLI()


if __name__ == '__main__':
    main()
