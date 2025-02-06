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

"""Cerebras ModelZoo Data Processor Information CLI Tool"""

import argparse
import json
import pydoc

from cerebras.modelzoo.cli.utils import MZ_CLI_NAME


class DataInfoCLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.configure_parser(parser)
        args = parser.parse_args()
        args.func(args)

    @staticmethod
    def epilog():
        return (
            f"Use `{MZ_CLI_NAME} data_processor -h` to learn how to query and investigate available data processors. "
            f"See below for some basic examples.\n\n"
            f"List all data processors:\n"
            f"  $ {MZ_CLI_NAME} data_processor list\n\n"
            f"Get additional information on GptHDF5DataProcessor:\n"
            f"  $ {MZ_CLI_NAME} data_processor info GptHDF5DataProcessor\n\n"
            f"Get details on all configuration parameters for GptHDF5DataProcessor:\n"
            f"  $ {MZ_CLI_NAME} data_processor describe GptHDF5DataProcessor\n\n"
            f"Benchmark a dataloader based on a given configuration:\n"
            f"  $ {MZ_CLI_NAME} data_processor benchmark my_new_params.yaml\n\n"
            f"For more information on ModelZoo data processors and how they are used, see: "
            f"https://docs.cerebras.net/en/latest/wsc/Model-zoo/MZ-overview.html"
        )

    @staticmethod
    def configure_parser(parser):
        from cerebras.modelzoo.cli.utils import get_table_parser

        parent_parser = get_table_parser()

        subparsers = parser.add_subparsers(dest="cmd", required=True)

        list_parser = subparsers.add_parser(
            "list",
            parents=[parent_parser],
            add_help=False,
            help="Lists available data processors.",
        )
        list_parser.set_defaults(func=DataInfoCLI.data_processor_list)

        info_parser = subparsers.add_parser(
            "info",
            parents=[parent_parser],
            add_help=False,
            help="Gives a high level summary of a data processor and its supported components.",
        )
        info_parser.add_argument(
            "data_processor",
            default=None,
            help="Registered data processor name to display information on.",
        )
        info_parser.set_defaults(func=DataInfoCLI.data_processor_info)

        describe_parser = subparsers.add_parser(
            "describe",
            parents=[parent_parser],
            add_help=False,
            help="Provides detailed infomation about a given data processor.",
        )
        describe_parser.add_argument(
            "data_processor",
            default=None,
            help="Registered data processor name to display information on.",
        )
        describe_parser.set_defaults(func=DataInfoCLI.data_processor_describe)

        benchmark_parser = subparsers.add_parser(
            "benchmark",
            help="Benchmark a dataloader.",
        )
        benchmark_parser.add_argument(
            "params",
            help="Config file to get dataloader from.",
        )
        benchmark_parser.add_argument(
            "--num_epochs",
            default=None,
            type=int,
            help=(
                "Number of epochs to iterate over the dataloader. "
                "If unspecified, the dataloader is only iterated for one epoch."
            ),
        )
        benchmark_parser.add_argument(
            "--steps_per_epoch",
            default=None,
            type=int,
            help=(
                "Number of steps to iterate over the dataloader in each epoch. "
                "If unspecified, the dataloader is iterated in its entirety."
            ),
        )
        benchmark_parser.add_argument(
            "--sampling_frequency",
            default=None,
            type=int,
            help=(
                "Frequency at which to sample metrics. "
                "First step of each epoch is always sampled. Defaults to 100."
            ),
        )
        benchmark_parser.add_argument(
            "--profile_activities",
            nargs="+",
            default=None,
            help=(
                "List of optional activities to profile. "
                "If unspecified, no extra activities are profiled."
            ),
        )
        benchmark_parser.add_argument(
            "--logging",
            default="INFO",
            help="Specifies the default logging level. Defaults to INFO.",
        )
        benchmark_parser.set_defaults(func=DataInfoCLI.data_processor_benchmark)

    @staticmethod
    def data_processor_list(args):
        from tabulate import tabulate

        all_data_processors = list(DataInfoCLI._list_data_processors().keys())
        all_data_processors.sort()

        if args.json:
            print(json.dumps(all_data_processors))
        else:
            table = []
            for data_processor in all_data_processors:
                row = [data_processor]
                table.append(row)
            headers = ["Available data procesors in ModelZoo"]
            table_out = tabulate(table, headers=headers, tablefmt="fancy_grid")

            if args.no_pager:
                print(table_out)
            else:
                pydoc.pager(table_out)

    @staticmethod
    def data_processor_info(args):
        import pkgutil

        from tabulate import tabulate

        data_processor_name = args.data_processor

        data_processor_entry = DataInfoCLI._get_data_processor(
            data_processor_name
        )
        data_processor_path = pkgutil.get_loader(
            data_processor_entry["path"].rsplit(".", 1)[0]
        ).get_filename()
        data_processor_models = data_processor_entry["models"]

        if args.json:
            json_dict = {
                "Name": data_processor_name,
                "Path": data_processor_path,
                "Supported Models": data_processor_models,
            }
            print(json.dumps(json_dict))
        else:
            table = []

            table.append(["Name", data_processor_name])
            # TODO: add description
            table.append(["Path", data_processor_path])
            table.append(["Supported Models", '\n'.join(data_processor_models)])
            table_out = tabulate(table, tablefmt="fancy_grid")

            if args.no_pager:
                print(table_out)
            else:
                pydoc.pager(table_out)

    @staticmethod
    def data_processor_describe(args):
        from tabulate import tabulate

        from cerebras.modelzoo.config import (
            create_config_class,
            describe_fields,
        )
        from cerebras.modelzoo.registry import registry

        data_processor_name = args.data_processor

        data_processor_entry = DataInfoCLI._get_data_processor(
            data_processor_name
        )
        data_processor_cls = registry._import_class(
            data_processor_entry["path"], data_processor_name
        )

        data_processor_cfg = (
            create_config_class(data_processor_cls)
            .model_fields["config"]
            .annotation
        )
        fields = describe_fields(data_processor_cfg)

        if args.json:
            print(json.dumps(fields))
        else:
            header = list(fields[0].keys())

            table = [list(field.values()) for field in fields]

            table_out = tabulate(table, headers=header, tablefmt="fancy_grid")

            if args.no_pager:
                print(table_out)
            else:
                pydoc.pager(table_out)

    @staticmethod
    def data_processor_benchmark(args):
        import cerebras.pytorch as cstorch
        from cerebras.appliance import logger
        from cerebras.appliance.log import get_level_name
        from cerebras.modelzoo.common.utils.run.cli_parser import get_params
        from cerebras.modelzoo.trainer.utils import (
            create_dataloader_from_config,
        )
        from cerebras.modelzoo.trainer.validate import validate_trainer_params

        dl_cache = []
        name_map = []

        def run_benchmarking(data_processor_config, name):
            print(f"\n\n{name}")
            print("=" * len(name))

            if data_processor_config in dl_cache:
                duplicate_name = name_map[dl_cache.index(data_processor_config)]
                print(f"Dataloader identical to {duplicate_name}. Skipping...")
                return

            dl_cache.append(data_processor_config)
            name_map.append(name)

            data_loader_fn = lambda: create_dataloader_from_config(
                data_processor_config
            )

            metrics = cstorch.utils.benchmark.benchmark_dataloader(
                data_loader_fn,
                num_epochs=args.num_epochs,
                steps_per_epoch=args.steps_per_epoch,
                sampling_frequency=args.sampling_frequency,
                profile_activities=args.profile_activities,
                print_metrics=True,
            )

        logger.setLevel(get_level_name(args.logging))

        params = get_params(args.params)
        configs = validate_trainer_params(params)

        for trainer_idx, config in enumerate(configs):
            trainer_name = (
                "trainer" if len(configs) == 1 else f"trainer[{trainer_idx}]"
            )
            if config.fit:
                run_benchmarking(
                    config.fit.train_dataloader,
                    f"{trainer_name}.fit.train_dataloader",
                )
                if config.fit.val_dataloader:
                    for i, dl_cfg in enumerate(config.fit.val_dataloader):
                        run_benchmarking(
                            dl_cfg, f"{trainer_name}.fit.val_dataloader[{i}]"
                        )
            if config.validate:
                run_benchmarking(
                    config.validate.val_dataloader,
                    f"{trainer_name}.validate.val_dataloader",
                )
            if config.validate_all:
                for i, dl_cfg in enumerate(config.validate_all.val_dataloaders):
                    run_benchmarking(
                        dl_cfg,
                        f"{trainer_name}.validate_all.val_dataloaders[{i}]",
                    )

    @staticmethod
    def _get_data_processor(data_processor):
        all_data_procesors = DataInfoCLI._list_data_processors()

        if data_processor not in all_data_procesors:
            raise ValueError(
                f"Data processor {data_processor} was not found in list of "
                f"registered data processors. Please use `cszoo data_processor list` "
                f"for a list of all available data processors."
            )

        return all_data_procesors[data_processor]

    @staticmethod
    def _list_data_processors():
        from collections import defaultdict

        from cerebras.modelzoo.registry import registry

        # create reverse lookup of data processors to supported models
        mapping = defaultdict(list)

        for model in registry.get_model_names():
            for data_processor in registry.get_model(
                model
            ).data_processor_paths:
                mapping[data_processor].append(model)

        return {
            data_processor_path.rsplit(".", 1)[-1]: {
                "path": data_processor_path,
                "models": mapping[data_processor_path],
            }
            for data_processor_path in mapping.keys()
        }


if __name__ == '__main__':
    DataInfoCLI()
