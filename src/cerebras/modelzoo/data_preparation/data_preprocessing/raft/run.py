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
import logging
from typing import Dict, Tuple

import yaml

from cerebras.modelzoo.common.utils.utils import UniqueKeyLoader
from cerebras.modelzoo.data_preparation.data_preprocessing.raft.config_verifier import (
    ConfigVerifier,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.raft.generate_embeddings import (
    generate_embeddings,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.raft.similarity_search import (
    RaftTransformation,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.raft.split_dataset import (
    SplitDataset,
)

logging.basicConfig()
logger = logging.getLogger(__file__)


logger.setLevel(logging.INFO)


def load_yaml_file(file_path: str) -> Dict:
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict: Parsed configuration parameters.
    """
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=UniqueKeyLoader)


def extract_tokenizer_params(emb_config_file: str) -> Tuple[str, Dict]:
    """
    Extract tokenizer parameters from the embedding config.

    Args:
        emb_config_file (str): Configuration file path for embedding which contains the model used for embedding generation.

    Returns:
        Tuple[str, Dict]: Tokenizer string and parameters.
    """
    embedding_config = load_yaml_file(emb_config_file)
    hf_tokenizer_str = embedding_config['data']['processing'].get(
        "huggingface_tokenizer", None
    )
    tokenizer_params = embedding_config['data']['processing'].get(
        'tokenizer_params', None
    )

    if hf_tokenizer_str:
        return hf_tokenizer_str, tokenizer_params
    else:
        raise ValueError(
            "No HF tokenizer provided in the config file, Currently only HF tokenizers are supported"
        )


def raft_main(params: ConfigVerifier) -> None:
    """
    Main function to handle the Raft transformation process.

    Args:
        params (Dict): ConfigVerifier Class object
    """

    dataset_params = params.dataset

    # Raft Configs
    raft_config = params.raft_config

    if raft_config is None:
        raise KeyError(
            "The config file needs to have raft config for transformation."
        )

    raft_config.split = dataset_params.setup.data.split

    ctx_to_ctx = params.ctx_to_ctx

    ctx_embeddings = params.context_embeddings
    ques_embeddings = params.question_embeddings

    dataset_split = False
    ques_split_dir = None

    # Splitting the dataset into context and question
    splitObj = SplitDataset(dataset_params)

    # Getting the directory path for context embeddings
    add_num_tokens, tokenizer_str, tokenizer_params = False, None, None

    if raft_config.k is None:
        add_num_tokens = True
        tokenizer_str, tokenizer_params = extract_tokenizer_params(
            ctx_embeddings
        )

    ques_split_dir, ctx_split_dir = splitObj.load_and_split(
        add_num_tokens, tokenizer_str, tokenizer_params
    )

    if ctx_embeddings.suffix == '.yaml':

        dataset_split = True
        ctx_params = load_yaml_file(ctx_embeddings)

        logger.info("\n\nEmbedding generation started for context\n\n")
        ctx_embed_dir = generate_embeddings(ctx_params)
    else:
        ctx_embed_dir = ctx_embeddings

    # Getting the directory path for question embeddings
    if not ctx_to_ctx:
        if ques_embeddings.suffix == '.yaml':
            ques_params = load_yaml_file(ques_embeddings)

            logger.info("\n\nEmbedding generation started for question\n\n")
            ques_embed_dir = generate_embeddings(ques_params)
        else:
            ques_embed_dir = ques_embeddings
    else:
        ques_embed_dir = ctx_embed_dir
    if ctx_embed_dir is None or ques_embed_dir is None:
        raise KeyError(
            "The config file needs to have directory paths for context and question if it doesn't have config files for embedding generation."
        )

    if not ctx_to_ctx:
        logger.info(
            f"\nEmbeddings fetched successfully for context at {ctx_embed_dir} and question at {ques_embed_dir}\n"
        )
    else:
        logger.info(
            f"\nEmbeddings fetched successfully for context at {ctx_embed_dir}\n"
        )

    # Preparing the final dataset based on similarity search using FAISS Index
    raft_config.ctx_embedding = ctx_embed_dir
    raft_config.ques_embedding = ques_embed_dir
    raft_config.ctx_to_ctx = ctx_to_ctx

    raft = RaftTransformation(raft_config)
    logger.info("\nRaft Transformation object created successfully\n")
    raft.raft_process()
    logger.info("Raft process completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Raft Config for CSX')
    parser.add_argument(
        '--config', type=str, help='Config file path', required=True
    )

    args = parser.parse_args()
    params = load_yaml_file(args.config)
    verified_params = ConfigVerifier(**params)
    # verified_dict = verified_params.model_dump()
    raft_main(verified_params)
    print(verified_params)
