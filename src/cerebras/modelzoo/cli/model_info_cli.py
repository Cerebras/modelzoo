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

"""Cerebras ModelZoo Model Information CLI Tool."""

import argparse
import io
import json
import os
import pydoc
import tempfile
import warnings
from contextlib import redirect_stdout

from cerebras.modelzoo.cli.utils import MZ_CLI_NAME, is_dir


class ModelInfoCLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.configure_parser(parser)
        args = parser.parse_args()
        args.func(args)

    @staticmethod
    def epilog():
        return (
            f"Use `{MZ_CLI_NAME} model -h` to learn how to query and investigate available models. "
            f"See below for some basic examples.\n\n"
            f"List all models:\n"
            f"  $ {MZ_CLI_NAME} model list\n\n"
            f"Get additional information on gpt2:\n"
            f"  $ {MZ_CLI_NAME} model info gpt2\n\n"
            f"Get details on all configuration parameters for gpt2:\n"
            f"  $ {MZ_CLI_NAME} model describe gpt2\n\n"
            f"For more information on ModelZoo models and how they are used, see: "
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
            help="Lists available models.",
        )
        list_parser.set_defaults(func=ModelInfoCLI.model_list)

        info_parser = subparsers.add_parser(
            "info",
            parents=[parent_parser],
            add_help=False,
            help="Gives a high level summary of a model and its supported components.",
        )
        info_parser.add_argument(
            "model",
            default=None,
            help="Registered model name to display information on.",
        )
        info_parser.set_defaults(func=ModelInfoCLI.model_info)

        describe_parser = subparsers.add_parser(
            "describe",
            parents=[parent_parser],
            add_help=False,
            help="Provides detailed infomation about a given model.",
        )
        describe_parser.add_argument(
            "model",
            default=None,
            help="Registered model name to display information on.",
        )
        describe_parser.set_defaults(func=ModelInfoCLI.model_describe)

        init_checkpoint_parser = subparsers.add_parser(
            "init_checkpoint",
            help="Generates an initial checkpoint for a model.",
        )
        init_checkpoint_parser.add_argument(
            "params",
            help="Path to .yaml file with model parameters.",
        )
        init_checkpoint_parser.add_argument(
            "-o",
            "--out",
            help=(
                "Path to save the checkpoint to. If it's a directory, the checkpoint is saved to that "
                "directory. Otherwise, the checkpoint is saved to the given file path. By default, the "
                "checkpoint is saved to the current working directory."
            ),
        )
        init_checkpoint_parser.set_defaults(
            func=ModelInfoCLI.model_init_checkpoint, seen_args=set(["params"])
        )

    @staticmethod
    def model_list(args):
        from tabulate import tabulate

        all_models = ModelInfoCLI._list_models()

        if args.json:
            print(json.dumps(all_models))
        else:
            table = [[model] for model in all_models]
            headers = ["Available models in ModelZoo"]
            table_out = tabulate(table, headers=headers, tablefmt="fancy_grid")

            if args.no_pager:
                print(table_out)
            else:
                pydoc.pager(table_out)

    @staticmethod
    def model_info(args):
        from tabulate import tabulate

        model_name = args.model

        model_path = ModelInfoCLI._get_model_path(model_name)
        model_configs = ModelInfoCLI._get_model_configs(model_name)
        model_dataprocessors = ModelInfoCLI._get_model_dataprocessors(
            model_name
        )

        if args.json:
            json_dict = {
                "Name": model_name,
                "Path": model_path,
                "Configs": model_configs,
                "Dataprocessors": model_dataprocessors,
            }
            print(json.dumps(json_dict))
        else:
            table = [
                ["Name", model_name],
                # TODO: add description
                ["Path", model_path],
                ["Configs", "\n".join(model_configs)],
                ["Dataprocessors", "\n".join(model_dataprocessors)],
            ]
            table_out = tabulate(table, tablefmt="fancy_grid")

            if args.no_pager:
                print(table_out)
            else:
                pydoc.pager(table_out)

    @staticmethod
    def model_describe(args):
        from tabulate import tabulate

        from cerebras.modelzoo.config import (
            create_config_class,
            describe_fields,
        )
        from cerebras.modelzoo.registry import registry

        model_name = args.model

        model_cls = registry.get_model_class(model_name)
        model_cfg = (
            create_config_class(model_cls).model_fields["config"].annotation
        )

        fields = describe_fields(model_cfg)

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
    def model_init_checkpoint(args):
        import torch

        import cerebras.pytorch as cstorch
        from cerebras.appliance.storage import StorageReader
        from cerebras.modelzoo.cli.utils import _args_to_params
        from cerebras.modelzoo.trainer.utils import (
            configure_trainer_from_config,
            convert_legacy_params_to_trainer_params,
            is_legacy_params,
        )
        from cerebras.modelzoo.trainer.validate import validate_trainer_params

        def path_exists(path: str):
            return StorageReader.path_exists(path)

        out_path = args.out or os.path.join(os.getcwd(), "")

        if not is_dir(out_path) and path_exists(out_path):
            raise ValueError(f"Checkpoint {out_path} already exists.")

        with tempfile.TemporaryDirectory() as tempdir:
            params = _args_to_params(args)

            if isinstance(params, dict) and is_legacy_params(params):
                warnings.warn(
                    f"Detected that legacy params are being used. "
                    f"Automatically converting params to new format."
                )

                params = convert_legacy_params_to_trainer_params(
                    params,
                    # Allow None values in the params
                    obj_filter=lambda obj: obj is None,
                )

                # injecting default device to config
                params["trainer"]["init"].setdefault("device", "CSX")

            if isinstance(params["trainer"], (list, tuple)):
                for trainer in params["trainer"]:
                    trainer["trainer"]["init"]["model_dir"] = tempdir
            else:
                params["trainer"]["init"]["model_dir"] = tempdir

            configs = validate_trainer_params(params)

            for i, config in enumerate(configs):
                if len(configs) > 1:
                    print(
                        f"Generating initial checkpoint for trainer instance {i}:"
                    )
                else:
                    print(f"Generating initial checkpoint")
                with redirect_stdout(io.StringIO()) as _:
                    trainer = configure_trainer_from_config(config)

                    assert str(trainer.model_dir) == str(
                        tempdir
                    ), f"{trainer.model_dir} != {tempdir}"

                    if is_dir(out_path):
                        trainer.checkpoint.checkpoint_root = out_path
                    else:
                        trainer.checkpoint.checkpoint_root = os.path.dirname(
                            out_path
                        )
                        trainer.checkpoint.checkpoint_name = os.path.basename(
                            out_path
                        )

                    if len(configs) > 1:
                        name, ext = os.path.splitext(
                            trainer.checkpoint.checkpoint_name
                        )
                        trainer.checkpoint.checkpoint_name = (
                            f"{name}_trainer{i}{ext}"
                        )

                    trainer.save_checkpoint()

                    for _ in cstorch.utils.data.DataExecutor(
                        cstorch.utils.data.DataLoader(lambda: [torch.zeros(1)]),
                        num_steps=1,
                    ):
                        break

                ckpt_file = trainer.checkpoint.get_latest_checkpoint(trainer)
                if not path_exists(ckpt_file):
                    raise ValueError(f"Checkpoint {ckpt_file} not found.")

                print(f"Saved initial checkpoint at: {ckpt_file}")

    @staticmethod
    def _get_model_path(model):
        from cerebras.modelzoo.registry import registry

        return str(registry.get_model_path(model))

    @staticmethod
    def _get_model_configs(model):
        from cerebras.modelzoo.registry import registry

        model_path = registry.get_model_path(model)

        config_list = list(model_path.rglob(f"configs/*.yaml"))

        # handle edge cases
        if not config_list:
            config_list = list(model_path.rglob(f"**/*.yaml"))
        if not config_list:
            config_list = list(model_path.parent.rglob(f"**/*.yaml"))

        # can be replaced by str.removeprefix when upgrading to Python 3.9
        def remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix) :]
            return text

        return [remove_prefix(config.stem, "params_") for config in config_list]

    @staticmethod
    def _get_model_dataprocessors(model):
        from cerebras.modelzoo.registry import registry

        return [
            dp.rsplit(".", 1)[-1]
            for dp in registry.get_model(model).data_processor_paths
        ]

    @staticmethod
    def _list_models():
        from cerebras.modelzoo.registry import registry

        return registry.get_model_names()


if __name__ == '__main__':
    ModelInfoCLI()
