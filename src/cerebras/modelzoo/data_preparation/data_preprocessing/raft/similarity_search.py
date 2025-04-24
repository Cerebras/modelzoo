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

import logging
import os
import random
from typing import Any, Dict, Tuple

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

from cerebras.modelzoo.data_preparation.data_preprocessing.raft.config_verifier import (
    RaftConfig,
)


class RaftTransformation:
    def __init__(self, config: RaftConfig):
        """
        Description: Class to add distractor contexts to the dataset, memorization examples and save the processed data as a single file.

        Args:
            config (RaftConfig): RaftConfig Class.
        """
        self.config = config
        if self.config.ctx_embedding == '':
            raise ValueError("ctx_embedding key does not exist in config file")
        if self.config.ques_embedding == '':
            raise ValueError("ques_embedding key does not exist in config file")
        self.split = config.split
        self.dataset, self.dataset_id_map = self.get_dataset_instance(
            self.config.data_dir
        )
        self.ctx_dataset, self.ctx_id_map = self.get_dataset_instance(
            self.config.ctx_data_dir
        )
        self.num_tokens = None
        if self.config.k is None:
            self.num_tokens = self.config.num_tokens
            self.k = self.get_k_for_num_tokens(self.num_tokens)
        else:
            self.k = self.config.k

    def get_k_for_num_tokens(self, num_tokens: int) -> int:
        """
        Calculate the number of contexts to retreive from 'faiss' indexing when distractor context are added based on the number of tokens.

        Args:
            num_tokens (int): Number of tokens.

        Returns:
            int: Number of contexts to retreive.
        """
        num_tokens = self.ctx_dataset[self.split]['num_tokens']
        token_mean = np.mean(num_tokens)
        token_variance = np.var(num_tokens)

        num_document_mean = int(self.num_tokens / token_mean)
        num_document_variance = int(self.num_tokens / token_variance)

        # In statistics, for normally distributed data, about 95% of data points fall within 2 standard deviations
        # Although there is no guarantee that the data is normally distributed, we can use this as a heuristic
        return int(num_document_mean + 2 * np.sqrt(num_document_variance))

    def get_dataset_instance(self, input_dir) -> None:
        """
        Load dataset from files in the specified directory.
        """
        input_files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
        ]
        extensions = set(os.path.splitext(f)[1].lower() for f in input_files)
        if len(extensions) > 1:
            raise ValueError(
                "All input files must have the same file extension."
            )
        file_extension = extensions.pop()
        extension_to_format = {
            '.csv': 'csv',
            '.json': 'json',
            '.jsonl': 'json',
            '.parquet': 'parquet',
            '.arrow': 'arrow',
        }
        if file_extension not in extension_to_format:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        fmt = extension_to_format[file_extension]
        dataset = load_dataset(fmt, data_files=input_files)
        logger.info(
            f"\nDataset instance created successfully from {input_dir}\n"
        )

        id_map = {
            data_sample['id']: i
            for i, data_sample in enumerate(dataset[self.split])
        }
        return dataset, id_map

    def get_embeddings(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings from .npz files in the specified path.

        Args:
            path (str): Path to the directory containing .npz files.

        Returns:
            tuple: Tuple containing arrays of IDs and embeddings.
        """
        all_ids = []
        all_embeds = []
        for filename in os.listdir(path):
            if filename.endswith('.npz'):
                file_path = os.path.join(path, filename)
                data = np.load(file_path)
                all_ids.append(data['ids'])
                all_embeds.append(data['embds'])
            else:
                raise ValueError(
                    f"{path} must contain all files in .npz format"
                )

        all_ids = np.concatenate(all_ids)
        all_embeds = np.concatenate(all_embeds)

        return all_ids, all_embeds

    def add_hard_distractors(self) -> None:
        """
        Adds distractor context to the dataset based on embeddings.
        """
        if not os.path.isdir(self.config.ctx_embedding):
            raise ValueError(
                f"The path specified in {self.config.ctx_embedding} is not a directory"
            )

        self.ctx_ids, self.ctx_embds = self.get_embeddings(
            self.config.ctx_embedding
        )
        logger.info("ctx embeddings loaded")

        if not self.config.ctx_to_ctx:
            self.ques_ids, self.ques_embds = self.get_embeddings(
                self.config.ques_embedding
            )
            logger.info("ques embeddings loaded")
        else:
            self.ques_ids = self.ctx_ids
            self.ques_embds = self.ctx_embds

        logger.info("Index creation started")
        # get the embedding dimension
        d = self.ctx_embds.shape[1]
        # normalising the embeddings
        faiss.normalize_L2(self.ctx_embds)
        faiss.normalize_L2(self.ques_embds)
        # creating the faiss index based upon InnerProduct
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self.ctx_embds)
        logger.info("Index creation completed")

        self.most_similar_dict = dict()
        num_golden_contexts = 1

        for ques_idx, ques_embd in zip(self.ques_ids, self.ques_embds):
            ques_idx = int(ques_idx)
            if not self.config.ctx_to_ctx:
                num_golden_contexts = len(
                    self.dataset[self.split][ques_idx]['context_list']
                )

            ques_embd = np.expand_dims(ques_embd, axis=0)
            cosine_sim, indices = self.faiss_index.search(
                ques_embd,
                self.k
                + num_golden_contexts,  # doing +num_golden_contexts as top similar context may contain the golden document itself, which we will get rid of while adding distractor context
            )

            ## removing the golden context from the top k+1 similar contexts
            ## The logic we implemented here is like the JOIN operation in SQL, so try to interpret it that way
            top_k_similarities = [
                (self.ctx_ids[i], cosine_sim[0][idx])
                for idx, i in enumerate(indices[0])
                if (
                    not self.config.ctx_to_ctx
                    and ques_idx
                    not in [
                        q_id
                        for q_id, _ in self.ctx_dataset['train'][
                            self.ctx_id_map[self.ctx_ids[i]]
                        ]['global_ctx_id']
                    ]
                )
                or (self.config.ctx_to_ctx and ques_idx != self.ctx_ids[i])
            ]

            if not self.config.ctx_to_ctx:
                top_k_similarities = top_k_similarities[: self.k]
                self.most_similar_dict[ques_idx] = top_k_similarities

            else:  # this logic maps back the context id to context string

                global_ctx_ids = self.ctx_dataset[self.split][
                    self.ctx_id_map[ques_idx]
                ]['global_ctx_id']
                for global_ctx_id in global_ctx_ids:
                    ques_id, _ = global_ctx_id
                    if ques_id not in list(self.most_similar_dict.keys()):
                        self.most_similar_dict[ques_id] = top_k_similarities
                    else:
                        self.most_similar_dict[ques_id].extend(
                            top_k_similarities
                        )
                        self.most_similar_dict[ques_id] = sorted(
                            self.most_similar_dict[ques_id],
                            key=lambda x: x[1],
                            reverse=True,
                        )[: self.k]

        logger.info("Hard distractors added successfully")

    def map_fn(self, example: Dict, idx: int) -> Dict:
        """
        Map function to add distractor contexts to each example.

        Args:
            example (dict): A single example from the dataset.
            idx (int): Index of the example.

        Returns:
            dict: Updated example with distractor contexts.
        """
        question_id = example['id']

        if self.config.distractor_type == "hard":
            top_k_similarities = self.most_similar_dict[question_id]
            if self.num_tokens is None:
                top_k_ctx_list = [
                    self.ctx_dataset[self.split][self.ctx_id_map[ctx_id]][
                        'context'
                    ]
                    for ctx_id, _ in top_k_similarities
                ]
            else:
                # Adding distractor context until it meets the num_tokens criteria.
                # It may happen that the top k contexts are not enough to meet the num_tokens criteria.
                # In that case, we only add all the context thet we retrieved.
                top_k_ctx_list = []
                total_tokens = 0
                for ctx_id, _ in top_k_similarities:
                    tokens = self.ctx_dataset[self.split][
                        self.ctx_id_map[ctx_id]
                    ]['num_tokens']
                    ctx = self.ctx_dataset[self.split][self.ctx_id_map[ctx_id]][
                        'context'
                    ]
                    total_tokens += tokens
                    if total_tokens <= self.num_tokens:
                        top_k_ctx_list.append(ctx)
                    else:
                        example['distractor_tokens'] = total_tokens - tokens
                        break

            example["distractor_contexts"] = top_k_ctx_list

        else:
            N = len(self.dataset)
            indices = np.arange(N)
            distractor_indices = np.random.choice(
                indices, self.k, replace=False
            ).tolist()
            distractor_contexts = [
                self.dataset[i]['context'] for i in distractor_indices
            ]
            example["distractor_contexts"] = distractor_contexts

        return example

    def add_distractor_contexts(self) -> None:
        """
        Add distractor contexts to the dataset.
        """
        if self.config.distractor_type == "hard":
            self.add_hard_distractors()
        self.output_data = self.dataset.map(self.map_fn, with_indices=True)
        logger.info("Distractor contexts added successfully")

    def add_memorization_examples(self) -> None:
        """
        Add memorization examples to the dataset.
        """

        def replace_with_string(value: Any, replacement_string: str) -> Any:
            '''
            Recursively replace all values in a dictionary or list with a string.
            Replaces all string instance if answer is in a format of list[Union[str, List[str],dict[key, value[str]]]] or dict[key, value[str]] or str

            Args:
                value (Any): Value to be replaced.
                replacement_string (str): Replacement string.

            Returns:
                Any: Value with all instances replaced with the replacement string.

            '''
            if isinstance(value, dict):
                for k, v in value.items():
                    value[k] = replace_with_string(v, replacement_string)
            elif isinstance(value, list):
                for i in range(len(value)):
                    value[i] = replace_with_string(value[i], replacement_string)
            else:
                if self.replace_with_answer:
                    return replacement_string + " " + value

                return replacement_string
            return value

        def map_memorization_fn(example: Dict, idx: int) -> Dict:
            self.replace_with_answer = False
            if random.random() <= self.config.context_removal_probability:
                example['context_list'] = []
                if random.random() < self.config.answer_refusal_percent:
                    example[self.config.answer_key] = replace_with_string(
                        example[self.config.answer_key],
                        "There is not enough information in the context provided to answer the question.",
                    )
                else:
                    self.replace_with_answer = True
                    example[self.config.answer_key] = replace_with_string(
                        example[self.config.answer_key],
                        "The answer is not included in the documents, but based on my internal knowledge the answer is:",
                    )

            return example

        self.output_data = self.output_data.map(
            map_memorization_fn, with_indices=True
        )
        logger.info("Memorization examples added successfully")

    def raft_process(self) -> None:
        """
        Execute the RAFT processing pipeline.
        """
        self.add_distractor_contexts()
        self.add_memorization_examples()

        self.save_processed_data()

        logger.info(
            f"Raft process completed successfully and data is saved to {self.config.output_dir}"
        )

    def save_processed_data(self) -> None:
        """
        Save the processed data to a single parquet file.
        """
        dataframes = []
        for split, dataset in self.output_data.items():
            df = pd.DataFrame(dataset)
            df.drop(
                columns=[self.config.context_key], inplace=True
            )  # Drop the column
            dataframes.append(df)

        concatenated_df = pd.concat(dataframes, ignore_index=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save to JSONL file
        jsonl_file_path = os.path.join(
            self.config.output_dir, "raft_processed_data.jsonl"
        )
        concatenated_df.to_json(jsonl_file_path, orient='records', lines=True)
        logger.info(f"Saved combined data to {jsonl_file_path}")
