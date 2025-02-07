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

"""Cerebras ModelZoo Config Management CLI Tool"""

import argparse
import os
import shutil
from pathlib import Path

from cerebras.modelzoo.cli.utils import MZ_CLI_NAME


class DataPreprocessCLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.configure_parser(parser)
        args = parser.parse_args()
        args.func(args)

    @staticmethod
    def epilog():
        return (
            f"Use `{MZ_CLI_NAME} data_preprocess -h` to learn how to configure and run data preprocessing. "
            f"See below for some basic examples.\n\n"
            f"List all data preprocessing config variants:\n"
            f"  $ {MZ_CLI_NAME} data_preprocess list\n\n"
            f"Copy a data configuration file to a specified directory:\n"
            f"  $ {MZ_CLI_NAME} data_preprocess pull summarization_preprocessing -o workdir\n\n"
            f"Run data preprocessing using given configuration:\n"
            f"  $ {MZ_CLI_NAME} data_preprocess run --config workdir/summarization_preprocessing.yaml\n\n"
            f"For more information on data preprocessing, see: "
            f"https://docs.cerebras.net/en/latest/wsc/Model-zoo/Components/Data-preprocessing/data_preprocessing.html"
        )

    @staticmethod
    def configure_parser(parser):
        subparsers = parser.add_subparsers(dest="cmd", required=True)

        list_parser = subparsers.add_parser(
            "list", help="List all data config variants."
        )
        list_parser.set_defaults(func=DataPreprocessCLI._config_list)

        pull_parser = subparsers.add_parser(
            "pull",
            help="Saves a data config file with a given variant name to the local workspace.",
        )
        pull_parser.add_argument(
            "variant",
            help="Config variant name to load.",
        )
        pull_parser.add_argument(
            "-o",
            "--outdir",
            help="Directory to save config to. If not specified, saves to cwd.",
        )
        pull_parser.set_defaults(func=DataPreprocessCLI._config_pull)

        run_parser = subparsers.add_parser(
            "run", help="Runs data preprocessing."
        )
        run_parser.set_defaults(func=DataPreprocessCLI._preprocess)
        from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
            add_preprocess_args,
        )

        add_preprocess_args(run_parser)

    @staticmethod
    def _config_list(args):
        print(DataPreprocessCLI._list_configs())

    @staticmethod
    def _config_pull(args):
        config_path = DataPreprocessCLI._get_config_path()
        variant_path = config_path / (args.variant + ".yaml")

        if not variant_path.exists():
            raise ValueError(
                f"Variant {args.variant}  not found. Please specify a valid variant from:\n"
                f"{DataPreprocessCLI._list_configs()}"
            )

        outdir = Path(args.outdir if args.outdir else os.getcwd())

        print(f"Saving config {args.variant} to {outdir}/{args.variant}.yaml")

        outdir.mkdir(parents=True, exist_ok=True)

        shutil.copy(str(variant_path), str(outdir))

    @staticmethod
    def _list_configs():
        from tabulate import tabulate

        config_path = DataPreprocessCLI._get_config_path()
        config_list = list(config_path.glob("*.yaml"))

        table = []
        for config in config_list:
            row = [config.stem]
            table.append(row)
        headers = ["Available data preprocessing configurations"]
        return tabulate(table, headers=headers, tablefmt="fancy_grid")

    @staticmethod
    def _get_config_path():
        import cerebras.modelzoo.data_preparation.data_preprocessing as data_preprocessing

        return Path(data_preprocessing.__file__).parent / "configs"

    @staticmethod
    def _preprocess(args):
        from cerebras.modelzoo.data_preparation.data_preprocessing.preprocess_data import (
            preprocess_data,
        )
        from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
            args_to_params,
        )

        params = args_to_params(args)
        preprocess_data(params)


if __name__ == '__main__':
    DataPreprocessCLI()
