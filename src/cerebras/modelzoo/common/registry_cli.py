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

from cerebras.modelzoo.common.registry import registry


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Registry for Cerebras Modelzoo'
    )

    # Add arguments
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List models supported in Cerebras Modelzoo',
    )
    parser.add_argument(
        '--list_losses',
        action='store_true',
        help='List Losses supported in Cerebras Modelzoo',
    )
    parser.add_argument(
        '--list_datasetprocessor',
        action='store_true',
        help='List dataset processor supported in Cerebras Modelzoo',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='List Model speicfic information from Cerebras modelzoo',
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.list_models:
        print(registry.list_models())

    if args.list_losses:
        print(registry.list_loss())

    if args.list_datasetprocessor:
        print(registry.list_datasetprocessor())

    if args.model is not None:
        print("Model: {}".format(args.model))
        print("\tclass: {}".format(registry.get_model_class(args.model)))
        print(
            "\tdataset processor: {}".format(
                registry.list_datasetprocessor(args.model)
            )
        )
        print("\tRun Path: {}".format(registry.get_run_path(args.model)))


if __name__ == "__main__":
    main()
