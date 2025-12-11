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

import argparse
import difflib

import yaml
from datatrove.executor.local import LocalPipelineExecutor

from cerebras.modelzoo.data_preparation.data_curation.pipeline.climbmix import (
    import_cls,
)


def reader_adapter(self, record: dict, path: str, id_in_file: int | str):
    # Called as: self.adapter(record, path, id_in_file) -> self is auto-bound
    old_contents = record.get("old_contents") or ""
    new_contents = record.get("new_contents") or ""

    # Generate diff from the old and new contents
    diff = difflib.unified_diff(
        old_contents.splitlines(keepends=True),
        new_contents.splitlines(keepends=True),
        fromfile=record.get("old_file") or "",
        tofile=record.get("new_file") or "",
    )

    return {
        "id": None,  # Ignore the id as commitpack does not use it
        "text": ''.join(diff),
        "metadata": record,
    }


def main():
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cfg = config["commitpack_diff"]

    executor = LocalPipelineExecutor(
        pipeline=[
            import_cls(cfg["reader"]["cls"])(
                **cfg["reader"]["params"],
                adapter=reader_adapter,
            ),
            import_cls(cfg["writer"]["cls"])(**cfg["writer"]["params"]),
        ],
    )

    executor.run()


if __name__ == "__main__":
    main()
