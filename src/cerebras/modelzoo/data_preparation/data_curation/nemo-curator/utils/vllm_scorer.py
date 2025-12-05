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

import json
import logging
from typing import Any

import dask.dataframe as dd
import pandas as pd
from dask import delayed
from dask.distributed import Client, get_worker
from more_itertools import chunked
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class VLLMScorer:
    """
    A class for scoring texts using prompts using VLLM.

    Attributes:
        client (Client): A Dask distributed client for managing distributed tasks.
        model (str): The name or path of the LLM model to use for inference.
        prompt_template (str): Template string with a {text} placeholder for prompt generation.
        json_schema (dict): JSON schema of the structured output.
        input_text_field (str): Name of the column containing input text.
        temperature (float): Sampling temperature for the LLM. Default is 0.2.
        max_tokens (int): Maximum number of tokens to generate per prompt. Default is 100.
        batch_size (int): Number of prompts to process in a single batch. Default is 8.

    Methods:
        __call__(ddf: dd.DataFrame):
            Scores a Dask DataFrame by batching and submitting its partitions for inference.

    Static Methods:
        _get_or_create_llm(model: str, json_schema: dict[str, Any], temperature: float, max_tokens: int):
            Retrieves or initializes an LLM instance and its sampling parameters on the current worker.

        _run_batch_inference(prompts: list[str], model: str, json_schema: dict[str, Any], input_text_field: str, temperature: float, max_tokens: int) -> list[str]:
            Runs inference on a batch of prompts and returns the generated outputs.

        _process_partition(df, prompt_template: str, model: str, json_schema: dict[str, Any], input_text_field: str, temperature: float, max_tokens: int, batch_size: int):
            Processes a single partition of a DataFrame by batching and running inference on its rows.

        _batch_and_submit(ddf: dd.DataFrame, client: Client, model: str, prompt_template: str, json_schema: dict[str, Any], input_text_field: str, temperature: float, max_tokens: int, batch_size: int):
            Batches and submits Dask DataFrame partitions for distributed inference, returning the combined results.
    """

    def __init__(
        self,
        client: Client,
        model: str,
        prompt_template: str,
        json_schema: dict[str, Any],
        input_text_field: str = "text",
        temperature: float = 0.2,
        max_tokens: int = 100,
        batch_size: int = 8,
    ):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.json_schema = json_schema
        self.input_text_field = input_text_field
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    @staticmethod
    def _get_or_create_llm(
        model: str,
        json_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ):
        worker = get_worker()
        if not hasattr(worker, "llm"):
            worker.llm = LLM(model=model)
            worker.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                guided_decoding=GuidedDecodingParams(json=json_schema),
            )

        return worker.llm, worker.sampling_params

    @staticmethod
    def _run_batch_inference(
        prompts: list[str],
        model: str,
        json_schema: dict[str, Any],
        input_text_field: str,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        llm, sampling_params = VLLMScorer._get_or_create_llm(
            model, json_schema, temperature, max_tokens
        )

        outputs = llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            try:
                results.append(json.loads(output.outputs[0].text))
            except json.JSONDecodeError:
                logging.error(
                    f"Failed to decode JSON: {output.outputs[0].text}"
                )
                results.append({})
        return results

    @staticmethod
    @delayed
    def _process_partition(
        df,
        prompt_template: str,
        model: str,
        json_schema: dict[str, Any],
        input_text_field: str,
        temperature: float,
        max_tokens: int,
        batch_size: int,
    ):
        texts = df[input_text_field].tolist()

        # Chunk the texts into batches
        batches = chunked(texts, batch_size)

        outputs = []
        for batch in batches:
            prompts = [prompt_template.format(text=text) for text in batch]

            # Perform batch inference
            output = VLLMScorer._run_batch_inference(
                prompts,
                model,
                json_schema,
                input_text_field,
                temperature,
                max_tokens,
            )
            outputs.extend(output)

        return outputs

    @staticmethod
    def _batch_and_submit(
        ddf: dd.DataFrame,
        client: Client,
        model: str,
        prompt_template: str,
        json_schema: dict[str, Any],
        input_text_field: str,
        temperature: float,
        max_tokens: int,
        batch_size: int,
    ):
        # Convert DataFrame into delayed partitions
        delayed_partitions = ddf[[input_text_field]].to_delayed()

        delayed_results = [
            VLLMScorer._process_partition(
                partition,
                prompt_template,
                model,
                json_schema,
                input_text_field,
                temperature,
                max_tokens,
                batch_size,
            )
            for partition in delayed_partitions
        ]

        # Submit the delayed process_partition function for each partition
        futures = client.compute(delayed_results)

        # Gather the results from the futures
        results = client.gather(futures)

        flat_results = [item for sublist in results for item in sublist]

        results_df = pd.DataFrame(flat_results).fillna("")

        # Add the new columns to the original DataFrame
        for column in results_df.columns:
            ddf[column] = dd.from_pandas(
                results_df[column], npartitions=ddf.npartitions
            )

        return ddf

    def __call__(self, ddf: dd.DataFrame):
        return VLLMScorer._batch_and_submit(
            ddf,
            self.client,
            self.model,
            self.prompt_template,
            self.json_schema,
            self.input_text_field,
            self.temperature,
            self.max_tokens,
            self.batch_size,
        )
