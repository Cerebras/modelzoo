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


# subclass ArgumentParser to handle subparser errors better
class SmartArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        args, unrecognized = super().parse_known_args(args, namespace)
        if unrecognized:
            self.error(f"unrecognized arguments: {' '.join(unrecognized)}")
        return args, unrecognized


class ModelZooCLI:
    def __init__(self):
        from cerebras.modelzoo.cli.assistant_cli import AssistantCLI
        from cerebras.modelzoo.cli.checkpoint_cli import CheckpointCLI
        from cerebras.modelzoo.cli.config_mgmt_cli import ConfigMgmtCLI
        from cerebras.modelzoo.cli.data_info_cli import DataInfoCLI
        from cerebras.modelzoo.cli.data_preprocess_cli import DataPreprocessCLI
        from cerebras.modelzoo.cli.model_info_cli import ModelInfoCLI
        from cerebras.modelzoo.cli.utils import EPILOG, add_run_args
        from cerebras.modelzoo.common.run_bigcode_eval_harness import (
            add_bigcode_args,
        )
        from cerebras.modelzoo.common.run_eleuther_eval_harness import (
            add_eeh_args,
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
        subparsers = parser.add_subparsers(
            dest="cmd",
            required=True,
            parser_class=SmartArgumentParser,
        )

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

        ###########
        # lm_eval #
        ###########
        lm_eval_parser = subparsers.add_parser(
            "lm_eval",
            help="Invokes script for running Eleuther Eval Harness.",
            epilog=(
                "For more information on Eleuther Eval Harness, see: "
                "https://docs.cerebras.net/en/latest/wsc/Model-zoo/core_workflows/downstream_eeh.html"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        add_eeh_args(lm_eval_parser)
        seen_args = add_run_args(lm_eval_parser, devices=["CSX"])
        lm_eval_parser.set_defaults(
            func=ModelZooCLI.run_lm_eval_harness,
            seen_args=seen_args,
        )

        ################
        # bigcode_eval #
        ################
        bigcode_eval_parser = subparsers.add_parser(
            "bigcode_eval",
            help="Invokes script for running BigCode Eval Harness.",
            epilog=(
                "For more information on BigCode Eval Harness, see: "
                "https://docs.cerebras.net/en/latest/wsc/Model-zoo/core_workflows/downstream_bceh.html"
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        add_bigcode_args(bigcode_eval_parser)
        seen_args = add_run_args(bigcode_eval_parser, devices=["CSX"])
        bigcode_eval_parser.set_defaults(
            func=ModelZooCLI.run_bigcode_eval_harness,
            seen_args=seen_args,
        )

        ##############
        # checkpoint #
        ##############
        checkpoint_parser = subparsers.add_parser(
            "checkpoint",
            # TODO: Change help message
            help="Get information on or perform some action on a checkpoint(s)",
            # TODO: Change epilog
            epilog=CheckpointCLI.epilog(),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        CheckpointCLI.configure_parser(checkpoint_parser)

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
        args, argv = parser.parse_known_args()
        args.func(args)

    @staticmethod
    def run_trainer(args):
        from cerebras.modelzoo.cli.utils import _args_to_params
        from cerebras.modelzoo.trainer.restartable_trainer import (
            RestartableTrainer,
        )
        from cerebras.modelzoo.trainer.utils import run_trainer

        params = _args_to_params(args)

        if RestartableTrainer.is_restart_config(params):
            RestartableTrainer(params).run_trainer(args.mode)
        else:
            run_trainer(args.mode, params)

    @staticmethod
    def run_lm_eval_harness(args):
        from cerebras.modelzoo.cli.utils import _args_to_params
        from cerebras.modelzoo.common.run_eleuther_eval_harness import (
            eeh_parser,
            run_lm_eval,
        )
        from cerebras.modelzoo.trainer.utils import EEH_TRAINER_PARAMS_TO_LEGACY

        extra_legacy_mapping_fn = (
            lambda trainer_to_legacy_mapping: trainer_to_legacy_mapping["init"][
                "callbacks"
            ].append(EEH_TRAINER_PARAMS_TO_LEGACY)
        )

        params = _args_to_params(
            args,
            validate=False,
            extra_legacy_mapping_fn=extra_legacy_mapping_fn,
        )

        run_lm_eval(params, eeh_parser())

    @staticmethod
    def run_bigcode_eval_harness(args):
        from cerebras.modelzoo.cli.utils import _args_to_params
        from cerebras.modelzoo.common.run_bigcode_eval_harness import (
            bigcode_parser,
            run_bigcode_eval,
        )
        from cerebras.modelzoo.trainer.utils import (
            BCEH_TRAINER_PARAMS_TO_LEGACY,
        )

        extra_legacy_mapping_fn = (
            lambda trainer_to_legacy_mapping: trainer_to_legacy_mapping["init"][
                "callbacks"
            ].append(BCEH_TRAINER_PARAMS_TO_LEGACY)
        )

        params = _args_to_params(
            args,
            validate=False,
            extra_legacy_mapping_fn=extra_legacy_mapping_fn,
        )

        run_bigcode_eval(params, bigcode_parser())


def main():
    ModelZooCLI()


if __name__ == '__main__':
    main()
