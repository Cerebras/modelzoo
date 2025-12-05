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

"""Cerebras ModelZoo LLM Assistant"""

import argparse
import os
import subprocess
import sys

from cerebras.modelzoo.cli.utils import EPILOG_WITHOUT_ASSISTANT, MZ_CLI_NAME


class AssistantCLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.configure_parser(parser)
        args = parser.parse_args()
        self.run_agent(args)

    @staticmethod
    def configure_parser(parser):
        parser.add_argument(
            "user_query",
            type=str,
            help="User query for LLM assistant.",
        )
        parser.add_argument(
            '--model',
            type=str,
            default="llama-3.3-70b",
            help="The name of the model to use with Cerebras inference.",
        )
        parser.add_argument(
            "--show-reasoning",
            action='store_true',
            help=(
                "Prints LLM assistant's intermediate reasoning steps to stdout."
            ),
        )
        parser.add_argument(
            "--max-reasoning-steps",
            type=int,
            default=5,
            help=(
                "The maximum number of internal thinking/reasoning steps the "
                "model is allowed."
            ),
        )
        parser.set_defaults(func=AssistantCLI.run_agent)

    @staticmethod
    def run_agent(args):
        # Lazy import to ensure that assistant doesn't impact import time of the
        # rest of the assistant
        import time

        import click
        from termcolor import colored

        import cerebras.cloud.sdk

        if os.environ.get("CEREBRAS_API_KEY", "") == "":
            print(
                f"\nThe {MZ_CLI_NAME} LLM assistant requires a Cerebras "
                f"inference API key to be set via the CEREBRAS_API_KEY env "
                f"variable. If you don't already have an API key, you can "
                f"acquire one at https://inference.cerebras.ai/",
                file=sys.stderr,
            )
            sys.exit(1)

        client = cerebras.cloud.sdk.Cerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY"),
        )

        # Explicitly bake in the output of `cszoo -h` in the system prompt so:
        # 1. the model has some "seed" context to get started with
        # 2. we don't need to run & capture the output of `cszoo -h` every time
        #    the assistant is used.
        system_prompt = f"""You are a helpful assistant for the cerebras modelzoo command line interface (CLI) tool.

You are allowed to think before you produce a final response. Everything you output by default will be considered an "internal thought". To signal that you're ready to produce your final answer, wrap your response in the XML tag FINAL_RESPONSE.

You may also use the CLI as part of your thinking in order to respond to user queries. Whenever you want to use the tool, respond with the command that you want to execute wrapped in the XML tag EXECUTE_COMMAND.

For example, if you write:
<EXECUTE_COMMAND>
cszoo -h
</EXECUTE_COMMAND>
The tool's output will be:
```
usage: cszoo [-h] {{fit,validate,validate_all,checkpoint,data_preprocess,model,data_processor,config}} ...

Cerebras ModelZoo CLI. This serves as a single entry-point to all ModelZoo related tasks including: training and validation, checkpoint conversion, data preprocessing and config management.

positional arguments:
  {{fit,validate,validate_all,checkpoint,data_preprocess,model,data_processor,config}}
    fit                 Run a model by calling fit. This completes a full training run on the given train and validation dataloaders.
    validate            Run a model by calling validate. This completes a full validation run on the specified validation dataloader.
    validate_all        Run a model by calling validate_all. This runs all upstream and downstream validation permutations.
    checkpoint          Convert checkpoint and/or config to a different format.
    data_preprocess     Run data preprocessing.
    model               Get information on available models.
    data_processor      Get information on available data processors.
    config              Manage model config files.

optional arguments:
  -h, --help            show this help message and exit

Use `cszoo <cmd> -h` to learn how to use individual sub-commands. See below for some basic examples.

{EPILOG_WITHOUT_ASSISTANT}
```

Then if you want to respond to the user, you can write:
<FINAL_RESPONSE>
put my final response here between the XML tags
</FINAL_RESPONSE>
""".replace(
            "cszoo", MZ_CLI_NAME
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.user_query},
        ]

        errors = 0
        error_limit = 2
        turns = 0
        turn_limit = args.max_reasoning_steps

        # Let the "agent" run in a loop until either:
        # 1. it produces and answer wrapped in FINAL_RESPONSE xml tags
        # 2. or more than 1 of the commands it tried to run failed
        # commands that the model wants to run are detected by search for
        # the EXECUTE_COMMAND xml tag
        while True:

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=False,
                    stop="</EXECUTE_COMMAND>",
                    temperature=0,
                )
            except cerebras.cloud.sdk.APIConnectionError as e:
                print(
                    "The Cerebras inference API server could not be reached",
                    file=sys.stderr,
                )
                print(e.__cause__, file=sys.stderr)
                sys.exit(1)
            except cerebras.cloud.sdk.RateLimitError as e:
                print(
                    "The LLM assistant has hit the rate limit. "
                    "Waiting 1 minute before continuing.",
                    file=sys.stderr,
                )
                time.sleep(60)
                continue
            except cerebras.cloud.sdk.APIStatusError as e:
                if e.body["code"] == "wrong_api_key":
                    print(
                        f"\nThe API key that was provided is invalid. Make sure"
                        f" your CEREBRAS_API_KEY env variable is set correctly",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                else:
                    # Some other error occurred so we need to re-raise:
                    raise e

            turns += 1
            completion = response.choices[0].message.content
            if args.show_reasoning:
                print(completion)

            final_response_start_idx = completion.rfind("<FINAL_RESPONSE>")
            final_response_end_idx = completion.rfind("</FINAL_RESPONSE>")
            if final_response_start_idx != -1:
                print(
                    colored(
                        completion[
                            final_response_start_idx
                            + len("<FINAL_RESPONSE>") : final_response_end_idx
                        ],
                        "green",
                    )
                )
                break
            elif turns >= turn_limit:
                print(
                    colored(
                        completion,
                        "green",
                    )
                )
                break

            execute_command_start_idx = completion.rfind("<EXECUTE_COMMAND>")
            execute_command_end_idx = completion.rfind("</EXECUTE_COMMAND>")
            command = None
            if (
                execute_command_start_idx != -1
                and execute_command_end_idx < execute_command_start_idx
            ):
                command = completion[
                    execute_command_start_idx
                    + len("<EXECUTE_COMMAND>") : execute_command_end_idx
                ].strip()
                completion += "</EXECUTE_COMMAND>"

            messages.append({"role": "user", "content": completion})
            if command is not None:
                # Explicitly show the user which command the assistant wants to
                # run and get approval. This is to avoid scenario where the
                # assistant acts maliciously / makes an accidental mistake
                # (ex: rm -rf *)
                if not click.confirm(
                    f"Allow the assistant to run command '{command}'?",
                    default=True,
                ):
                    break
                try:
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        shell=True,
                        check=True,
                    )
                    print("Command output:\n", result.stdout)
                    messages.append(
                        {"role": "system", "content": result.stdout}
                    )
                except subprocess.CalledProcessError as e:
                    print("Command error:\n", e.stderr)
                    errors += 1
                    if errors >= error_limit:
                        break
                    else:
                        messages.append({"role": "system", "content": e.stderr})
