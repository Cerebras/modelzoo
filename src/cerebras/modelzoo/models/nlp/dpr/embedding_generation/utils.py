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

DEFAULT_EMBEDDINGS_PER_FILE = 1000


def extra_args_parser_fn():
    # Set `add_help`= False to prevent conflicts.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--embeddings_per_file",
        required=False,
        type=int,
        # We intentionally don't set the default in argparser as it would
        # override the yaml. Instead, it is set in the model.py file.
        help=(
            f"Number of embeddings in each output file. "
            f"Default: {DEFAULT_EMBEDDINGS_PER_FILE}"
        ),
    )
    parser.add_argument(
        "--embeddings_output_dir",
        required=False,
        type=str,
        help=(
            f"Directory to store dumped embeddings. "
            f"Default: either '<model dir>/embeddings_q_encoder' or "
            f"'<model dir>/embeddings_ctx_encoder' (depending on which encoder "
            f"is selected for embedding generation)"
        ),
    )
    return [parser]


def closest_multiple(number, base):
    return max(1, round(number / base)) * base
