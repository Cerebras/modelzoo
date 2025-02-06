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
import gzip
import io
import json
import logging
import numbers
import os
import re
from dataclasses import asdict, dataclass
from itertools import repeat
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import h5py
import jsonlines
import numpy as np
import pyarrow.parquet as pq
import yaml
import zstandard
from lm_dataformat import tarfile_reader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from cerebras.modelzoo.data_preparation.nlp.tokenizers.BPETokenizer import (
    BPETokenizer,
)
from cerebras.modelzoo.data_preparation.nlp.tokenizers.HFTokenizer import (
    HFTokenizer,
)
from cerebras.modelzoo.data_preparation.utils import split_list

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

## Added .parquet extension to the list of valid extensions
VALID_EXTENSIONS = [
    '.jsonl',
    '.jsonl.zst',
    '.jsonl.zst.tar',
    '.txt',
    '.json.gz',
    '.parquet',
]


SYSTEM_PROMPT_REGISTRY = {
    "zephyr": "<|system|>\n</s>",
    "vicuna_v0": (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    "vicuna_v1": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    "llava_plain": "",
    "llava_v0": (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    "llava_v1": (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    "mistral_instruct": "",
}


def has_valid_extension(file):
    return any([file.endswith(ext) for ext in VALID_EXTENSIONS])


def _listdir_or_file(x):
    if isinstance(x, list):
        return reduce(lambda x, y: x + y, map(listdir_or_file, sorted(x)))
    if os.path.isfile(x):
        return [x]
    elif os.path.isdir(x):
        return [str(Path(x) / fn) for fn in sorted(os.listdir(x))]
    else:
        raise FileNotFoundError(f"{x} not found")


def listdir_or_file(x):
    return list(filter(has_valid_extension, _listdir_or_file(x)))


def add_common_args(parser):
    """
    For the argparse to parse arguments for subcommands, we add common
    command line arguments to each subcommand parser here.
    """
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to the YAML config file for setting dataset preprocessing hyper-parameters.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory where raw data is stored.",
    )
    parser.add_argument(
        "--metadata_files",
        type=str,
        default=None,
        help="Path to text file containing a list of file names "
        "corresponding to the raw input documents to be "
        "processed and stored; can handle multiple metadata files "
        "separated by comma.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where HDF5 files will be stored.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["GPT2Tokenizer", "NeoXTokenizer", "HuggingFaceTokenizer"],
        help=(
            "Type of tokenizer to use for HDF5 dataset generation. "
            "Can be one of `GPT2Tokenizer`, `NeoXTokenizer` or `HuggingFaceTokenizer`."
        ),
    )
    parser.add_argument(
        "--huggingface_tokenizer",
        type=str,
        default=None,
        help=(
            "Name/Path to the HuggingFace tokenizer."
            "Only used when tokenizer_type=HuggingFaceTokenizer"
        ),
    )
    parser.add_argument(
        "--vocab_file", type=str, help="Path to the vocabulary file."
    )
    parser.add_argument(
        "--encoder_file", type=str, help="Path to the encoder file."
    )
    parser.add_argument(
        "--eos_id",
        type=int,
        help="Token id of the end of sentence token",
    )
    parser.add_argument(
        "--pad_id", type=int, help="Token id of the padding token."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.0,
        help=(
            "Probability of creating sequences which are shorter than the"
            + " maximum sequence length."
        ),
    )
    parser.add_argument(
        "--use_ftfy",
        type=str,
        choices=["True", "False"],
        help="Whether to fix text with ftfy. Defaults to `True`.",
    )
    parser.add_argument(
        "--ftfy_normalizer",
        type=str,
        choices=["NFC", None],
        help=(
            "Choose what kind of unicode normalization is applied. Usually, we "
            "apply `NFC` normalization, so that letters followed by combining "
            "characters become single combined characters. Using `None` "
            "applies no normalization while fixing text."
        ),
    )
    parser.add_argument(
        "--wikitext_detokenize",
        type=str,
        choices=["True", "False"],
        help="Whether to use wikitext detokenizer to fix text. Defaults to `False`.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="examples",
        help=(
            "Name of the dataset; i.e. prefix to use for HDF5 file names."
            + "Defaults to `examples`."
        ),
    )
    parser.add_argument(
        "--files_per_record",
        type=int,
        help="Text files to write per HDF5 file.",
    )
    parser.add_argument(
        "--write_in_batch",
        type=str,
        choices=["True", "False"],
        help="Whether to write the samples in batch for the HDF5 format, "
        "setting to false will save memory but a bit slower. Defaults to "
        "`True`.",
    )
    parser.add_argument(
        "--write_remainder",
        type=str,
        choices=["True", "False"],
        help="Write the remainder files when data is left over from "
        "processing. Defaults to `True`.",
    )
    parser.add_argument(
        "--min_sequence_len",
        type=int,
        default=6,
        help=(
            "sequences shorter than min_sequence_len tokens in length will be skipped"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        choices=["True", "False"],
        help="Resume record writing from a given checkpoint. Defaults to `False`.",
    )
    parser.add_argument(
        "--display_pbar",
        type=str,
        choices=["True", "False"],
        help="Display progress while runs. Defaults to `False`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        help="""Max chunk size in KB. This is supported
                only for chunk preprocessing pipeline""",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        help="""Whether to perform online shuffling. This is supported
                only for chunk preprocessing pipeline""",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        help="""Randomization seed to be used with online shuffling. This is supported
                only for chunk preprocessing pipeline""",
    )
    parser.add_argument(
        "--fraction_of_RAM_alloted",
        type=int,
        help="Fraction of RAM designated for data sharing among processes in the chunk preprocessing pipeline.",
    )


def add_lm_args(parser):
    """
    The language-modeling format is common enough (FIM is very similar)
    that we can re-use the arguments for it
    """
    parser.add_argument(
        "--jsonl_key",
        type=str,
        default=None,
        help="The key name in input jsonl files from which the raw text will be "
        "extracted in order to further process it.",
    )
    parser.add_argument(
        "--split_text_to_tokenize",
        type=str,
        choices=["True", "False"],
        help="Whether to split the text into smaller chunks before tokenizing.",
    )
    parser.add_argument(
        "--chunk_len_to_split",
        type=int,
        help="Length of the chunk size to split the text document into.",
    )
    parser.add_argument(
        "--remove_bos_in_chunks",
        type=str,
        choices=["True", "False"],
        help="Whether to ignore bos token id in chunks when splitting the text.",
    )
    parser.add_argument(
        "--pack_sequences",
        type=str,
        choices=["True", "False"],
        help="Concatenate a document smaller than maximum sequence length with "
        "other documents, instead of filling it with Padding token. Defaults "
        "to `True`.",
    )
    # Add the '--auth_token' argument
    parser.add_argument(
        "--auth_token",
        type=str,
        default=None,  # Make the token optional by setting the default to None
        help="The token to use as HTTP bearer authorization for remote files. If not provided, no token is used.",
    )


def add_summarization_args(parser):
    parser.add_argument(
        "--sep_token",
        type=str,
        default=None,
        help="Token added between prompt and completion in preprocessed sequences.",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        help="Json key for the prompt.",
    )
    parser.add_argument(
        "--completion_key",
        type=str,
        help="Json key for the completion.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        help="Chat Template passed as jinja string which will be used to override the tokenizer's default chat template.",
    )
    parser.add_argument(
        "--drop_input",
        type=str,
        help="Passed as a list of strings which can be used to drop specific regions of the input. For example passing - ['prompt'] will drop prompt region of the data input.",
    )
    parser.add_argument(
        "--loss_mask_weight",
        type=str,
        help="Used to specify weights for masking the loss corresponding to specific regions of the input.",
    )


def add_summarization_vsl_args(parser):
    add_summarization_args(parser)
    parser.add_argument(
        "--position_ids_dtype",
        type=str,
        help="Dtype for VSL position ids. Defaults to `int32`.",
    )
    parser.add_argument(
        "--multi_turn_key",
        type=str,
        help="""
        Json key for column where multi-turn dialogue is contained.
        Note that this should not be specified at the same time as prompt_key
        and completion_key -- either specify this flag or both of the other flags.
        """,
    )
    parser.add_argument(
        "--multi_turn_content_key",
        type=str,
        help="""
        If column specified by --multi_turn_key is a dictionary rather
        than a list, this specifies the json key for obtaining the message
        within one element of the above column. For example you could have
        one entry that is formatted as follows:
        [
            {"content": "First message", "role": "user"},
            {"content": "Response to first message", "role": "assistant"}
        ]
        and you would specify --multi_turn_content_key as "content". We make
        no assumption about other keys within the dictionary
        """,
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        help="""
        If specified, this will be added before the prompt in every sequence.
        Example usage is to add "<|user|>" before the user message in a
        multi-turn dialogue.
        """,
    )
    parser.add_argument(
        "--completion_prefix",
        type=str,
        help="""
        Similar to `prompt_prefix`, but for the completion.
        Example usage is to add "<|assistant|>" before the model's response in
        a multi-turn dialogue.
        """,
    )
    parser.add_argument(
        "--eos_after_prompt",
        type=bool,
        help="""
        Some current chat templates will include an EOS token after the end of
        the user input in a multi-turn dialogue. If this flag is specified,
        there will be EOS tokens after all prompts.
        """,
    )


def add_llava_common_args(parser):
    add_summarization_args(parser)
    parser.add_argument(
        "--eos_after_prompt",
        type=bool,
        help="""
        Some current chat templates will include an EOS token after the end of
        the user input in a multi-turn dialogue. If this flag is specified,
        there will be EOS tokens after all prompts.
        """,
    )
    parser.add_argument(
        "--multi_turn_key",
        type=str,
        help="""
        Json key for column where multi-turn dialogue is contained.
        Note that this should not be specified at the same time as prompt_key
        and completion_key -- either specify this flag or both of the other flags.
        """,
    )
    parser.add_argument(
        "--multi_turn_content_key",
        type=str,
        help="""
        If column specified by --multi_turn_key is a dictionary rather
        than a list, this specifies the json key for obtaining the message
        within one element of the above column. For example you could have
        one entry that is formatted as follows:
        [
            {"content": "First message", "role": "user"},
            {"content": "Response to first message", "role": "assistant"}
        ]
        and you would specify --multi_turn_content_key as "content". We make
        no assumption about other keys within the dictionary
        """,
    )
    parser.add_argument(
        "--image_key",
        type=str,
        help="Image key of the LLaVA dataset.",
    )
    parser.add_argument(
        "--multi_modal_non_image_ex_key",
        type=str,
        help="The non image key in the LLaVA dataset for text only examples",
    )
    parser.add_argument(
        "--image_token",
        type=str,
        help="""
        String that represents where in the text the image patches will be inserted.
        For example, the original LLaVA dataset contained the string "<image>" in the prompt.
        """,
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        help="Number of patches to represent an image.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Absolute path of image directory. Used along with the relative path under the `image_key` field to check that images exist",
    )


def add_llava_phase_1_args(parser):
    parser.add_argument(
        "--multi_modal_remove_text_prompt",
        type=bool,
        help="Whether to remove the prompt from the LLaVA dataset.",
    )


def add_llava_phase_2_args(parser):
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        help="""
        If specified, this will be added before the prompt in every sequence.
        Example usage is to add "<|user|>" before the user message in a
        multi-turn dialogue.
        """,
    )
    parser.add_argument(
        "--completion_prefix",
        type=str,
        help="""
        Similar to `prompt_prefix`, but for the completion.
        Example usage is to add "<|assistant|>" before the model's response in
        a multi-turn dialogue.
        """,
    )
    parser.add_argument(
        "--system_prompt_style",
        type=int,
        help="""
        Key to obtain the system prompt used for the LLM backbone within LLaVA.
        For example, if you training a LLaVA model based on the Vicuna model, you would specify "vicuna_v1".
        """,
    )


def add_multimodal_args(parser):
    add_summarization_args(parser)
    parser.add_argument(
        "--multi_turn_key",
        type=str,
        help="""
        Json key for column where multi-turn dialogue is contained.
        Note that this should not be specified at the same time as prompt_key
        and completion_key -- either specify this flag or both of the other flags.
        """,
    )
    parser.add_argument(
        "--image_key",
        type=str,
        help="Image key of the multimodal dataset.",
    )
    parser.add_argument(
        "--image_token",
        type=str,
        help="""
        String that represents where in the text the image patches will be inserted.
        For example, the original LLaVA dataset contained the string "<image>" in the prompt.
        """,
    )
    parser.add_argument(
        "--multimodal_mode",
        type=str,
        help="""
        String that represents whether the mode is 'interleaved' or 'non_interleaved'.
        For example, if you want the image patches to occur interleaved with text specify 'interleaved'.
        """,
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        help="Number of patches to represent an image.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Absolute path of image directory. Used along with the relative path under the `image_key` field to check that images exist",
    )
    parser.add_argument(
        "--system_prompt_style",
        type=int,
        help="""
        Key to obtain the system prompt used for the LLM backbone within the multimodal model.
        For example, if you training a LLaVA model based on the Vicuna model, you would specify "vicuna_v1".
        """,
    )


def add_dpo_args(parser):
    parser.add_argument(
        "--prompt_key",
        type=str,
        help="Json key for the prompt.",
    )
    parser.add_argument(
        "--chosen_key",
        type=str,
        help="Json key for the chosen response.",
    )
    parser.add_argument(
        "--rejected_key",
        type=str,
        help="Json key for the rejected response.",
    )
    parser.add_argument(
        "--user_role",
        type=str,
        default=None,
        help="Specify user role tag for a dailogue.",
    )
    parser.add_argument(
        "--assistant_role",
        type=str,
        default=None,
        help="Specify assistant role tag for a dailogue.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Specify chat template for the tokenizer.",
    )
    parser.add_argument(
        "--response_delimiter",
        type=str,
        default=None,
        help="Specify the token which separates prompt and responses. This is used to extract prompt from a dialogue.",
    )


def get_parser(desc):
    """Argparser definition for command line arguments from user.

    Returns:
        Argparse namespace object with command line arguments.
    """
    parser = argparse.ArgumentParser(description=desc)
    subparser = parser.add_subparsers(
        description="Sub command for HDF5 conversion.",
        dest="mode",
        required=True,
        help="Sub command to choose saving the raw text into HDF5 files or "
        "pre-processed text converted into token ids at desired maximum "
        "sequence length.",
    )

    ### LMData ###
    lm_parser = subparser.add_parser(
        "LMData", help="Language modeling dataset in `.jsonl` or `.txt` format."
    )
    add_common_args(lm_parser)
    add_lm_args(lm_parser)

    ### Summarization ###
    summarization_parser = subparser.add_parser(
        "Summarization", help="Fine-tuning dataset in plane text format."
    )
    add_common_args(summarization_parser)
    add_summarization_args(summarization_parser)
    ### FIM ###
    fim_parser = subparser.add_parser(
        "FIM", help="Pre-processing to allow Fill-in-the-Middle objective"
    )
    add_common_args(fim_parser)
    add_lm_args(fim_parser)
    fim_parser.add_argument(
        "--fim_rate",
        type=float,
        default=0.9,
        help="Percent of samples to undergo FIM transformation",
    )
    fim_parser.add_argument(
        "--spm_rate",
        type=float,
        default=0.5,
        help="""Percent of FIM samples to go into SPM format (as opposed
        to PSM)""",
    )
    fim_parser.add_argument(
        "--fim_prefix_tok",
        type=str,
        help="Can specify the special token denoting FIM prefix section",
    )
    fim_parser.add_argument(
        "--fim_middle_tok",
        type=str,
        help="Can specify the special token denoting FIM middle section",
    )
    fim_parser.add_argument(
        "--fim_suffix_tok",
        type=str,
        help="Can specify the special token denoting FIM suffix section",
    )

    ### LMData (VSL) ###
    lm_vsl_parser = subparser.add_parser(
        "LMData_VSL",
        help="Language modeling dataset for variable sequence length training.",
    )
    add_common_args(lm_vsl_parser)
    add_lm_args(lm_vsl_parser)
    lm_vsl_parser.add_argument(
        "--fold_long_doc",
        type=str,
        choices=["True", "False"],
        help="Whether to fold long documents into multiple sequences. Defaults to `True`.",
    )
    lm_vsl_parser.add_argument(
        "--position_ids_dtype",
        type=str,
        help="Dtype for VSL position ids. Defaults to `int32`.",
    )

    ### Summarization (VSL) ###
    summarization_vsl_parser = subparser.add_parser(
        "Summarization_VSL",
        help="Fine-tuning dataset in plane text format for variable sequence length training.",
    )
    add_common_args(summarization_vsl_parser)
    add_summarization_vsl_args(summarization_vsl_parser)

    ### DPO parser ###
    dpo_parser = subparser.add_parser(
        "DPO",
        help="DPO data preprocessing flag.",
    )
    add_common_args(dpo_parser)
    add_dpo_args(dpo_parser)

    ### Llava Phase 1 parser ###
    llava_phase_1_parser = subparser.add_parser(
        "LlavaPhaseOne",
        help="Llava Phase 1 preprocessing.",
    )
    add_common_args(llava_phase_1_parser)
    add_llava_common_args(llava_phase_1_parser)
    add_llava_phase_1_args(llava_phase_1_parser)

    ### Llava Phase 2 parser ###
    llava_phase_2_parser = subparser.add_parser(
        "LlavaPhaseTwo",
        help="Llava Phase 2 preprocessing.",
    )
    add_common_args(llava_phase_2_parser)
    add_llava_common_args(llava_phase_2_parser)
    add_llava_phase_2_args(llava_phase_2_parser)

    ##Multimodal Parser
    generic_multimodal_parser = subparser.add_parser(
        "multimodal",
        help="Generic multimodal preprocessing.",
    )
    add_common_args(generic_multimodal_parser)
    add_multimodal_args(generic_multimodal_parser)

    ### Customize ###
    custom_parser = subparser.add_parser(
        "Customize", help="Provide customized dataset processor."
    )
    add_common_args(custom_parser)
    custom_parser.add_argument(
        "--module",
        type=str,
        help="Python file name contains the custom dataset processor.",
    )
    custom_parser.add_argument(
        "--dataset_processor",
        type=str,
        help="Name of the custom dataset processor.",
    )

    return parser.parse_args()


def update_params(params, args):
    """
    Update config parameters with CLI arguments
    """
    setup_params = [
        "input_dir",
        "metadata_files",
        "output_dir",
        "processes",
        "module",
        "dataset_processor",
    ]
    processing_params = [
        "tokenizer_type",
        "huggingface_tokenizer",
        "vocab_file",
        "encoder_file",
        "eos_id",
        "pad_id",
        "split_text_to_tokenize",
        "chunk_len_to_split",
        "remove_bos_in_chunks",
        "max_seq_length",
        "short_seq_prob",
        "output_name",
        "files_per_record",
        "write_in_batch",
        "write_remainder",
        "resume_from_checkpoint",
        "display_pbar",
        "seed",
        "fim_rate",
        "spm_rate",
        "fim_prefix_tok",
        "fim_middle_tok",
        "fim_suffix_tok",
        "auth_token",
        "max_chunk_size",
        "shuffle",
        "shuffle_seed",
        "fraction_of_RAM_alloted",
        "drop_input",
        "loss_mask_weight",
        "chat_template",
        "multimodal_mode",
    ]
    dataset_params = [
        "use_ftfy",
        "ftfy_normalizer",
        "wikitext_detokenize",
        "jsonl_key",
        "pack_sequences",
        "min_sequence_len",
        "input_ids_dtype",
        "input_mask_dtype",
        "inverted_mask",
        "prompt_key",
        "completion_key",
        "sep_token",
        "fold_long_doc",
        "seq_lengths_dtype",
        "chosen_key",
        "rejected_key",
        "user_role",
        "assistant_role",
        "chat_template",
        "respose_delimiter",
        "prompt_prefix",
        "completion_prefix",
        "eos_after_prompt",
        "multi_turn_key",
        "multi_turn_content_key",
        "image_key",
        "image_token",
        "multi_modal_non_image_ex_key",
        "image_dir",
        "num_patches",
        "system_prompt_style",
    ]
    processor_map = {
        "lmdata": "LMDataPreprocessor",
        "summarization": "SummarizationPreprocessor",
        "fim": "FIMDataPreprocessor",
        "lmdata_vsl": "VSLLMDataPreprocessor",
        "summarization_vsl": "VSLSummarizationPreprocessor",
        "dpo": "DPOPreprocessor",
        "llavaphaseone": "LlavaPhaseOnePreprocessor",
        "llavaphasetwo": "LlavaPhaseTwoPreprocessor",
        "multimodal": "MultiModalTokenGenerator",
    }

    mode = args.pop("mode").lower()
    if mode != "customize":
        params["setup"]["dataset_processor"] = processor_map[mode]

    for key, value in args.items():
        if value in ["True", "False"]:
            value = value == "True"
        if value is not None:
            if key in setup_params:
                params["setup"][key] = value
            elif key in processing_params:
                params["processing"][key] = value
            elif key in dataset_params:
                params["dataset"][key] = value
            else:
                raise ValueError(f"Unexpected arguments: {key}")
    set_defaults(params)


def set_defaults(params):
    params["processing"]["eos_id"] = params["processing"].get("eos_id")
    params["processing"]["pad_id"] = params["processing"].get("pad_id")
    params["processing"]["split_text_to_tokenize"] = params["processing"].get(
        "split_text_to_tokenize", False
    )
    params["processing"]["chunk_len_to_split"] = params["processing"].get(
        "chunk_len_to_split", 2000
    )
    params["processing"]["remove_bos_in_chunks"] = params["processing"].get(
        "remove_bos_in_chunks", False
    )
    params["processing"]["write_in_batch"] = params["processing"].get(
        "write_in_batch", True
    )
    params["processing"]["write_remainder"] = params["processing"].get(
        "write_remainder", True
    )
    params["processing"]["resume_from_checkpoint"] = params["processing"].get(
        "resume_from_checkpoint", False
    )
    params["processing"]["display_pbar"] = params["processing"].get(
        "display_pbar", False
    )
    params["processing"]["auth_token"] = params["processing"].get(
        "auth_token", None
    )
    params["dataset"]["use_ftfy"] = params["dataset"].get("use_ftfy", True)
    params["dataset"]["ftfy_normalizer"] = params["dataset"].get(
        "ftfy_normalizer", "NFC"
    )
    params["dataset"]["wikitext_detokenize"] = params["dataset"].get(
        "wikitext_detokenize", False
    )


def get_params(desc):
    """Retrieve configuration parameters
    Returns:
        params (Dict): Dictionary contains the parameters used to configure
            the data processing.
    """
    args = get_parser(desc)
    args = vars(args)

    params_file = args.pop("params", None)
    if params_file:
        with open(params_file, 'r') as stream:
            params = yaml.safe_load(stream)
    else:
        params = {}

    for section in ["setup", "processing", "dataset"]:
        if not params.get(section, None):
            params[section] = {}

    update_params(params, args)
    return params


def dump_args(args, json_params_file):
    """
    Write the input params to file.
    """
    # write initial params to file
    with open(json_params_file, "w") as _fout:
        json.dump(args, _fout, indent=4, sort_keys=True)


def dump_result(
    results,
    dataset_stats,
    json_params_file,
    eos_id=None,
    pad_id=None,
    vocab_size=None,
):
    """
    Write outputs of execution
    """
    with open(json_params_file, "r") as _fin:
        data = json.load(_fin)

    post_process = {}
    post_process["discarded_files"] = results["discarded"]
    post_process["processed_files"] = results["processed"]
    post_process["successful_files"] = results["successful"]
    post_process["n_examples"] = results["examples"]
    post_process["raw_chars_count"] = results["raw_chars_count"]
    post_process["raw_bytes_count"] = results["raw_bytes_count"]

    if eos_id is not None:
        post_process["eos_id"] = eos_id
    if pad_id is not None:
        post_process["pad_id"] = pad_id
    if vocab_size is not None:
        post_process["vocab_size"] = vocab_size

    data["post-process"] = post_process
    data["h5_dataset_stats"] = asdict(dataset_stats)

    with open(json_params_file, "w") as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)


def multimodal_add_image_patch_start_idx(json_params_file, dataset_processor):
    # check whether the data-processor is enabled for multi-modal
    # and it had multi-modal examples to find the image-patch
    # start index
    if (
        hasattr(dataset_processor, "multimodal_preprocessor")
        and dataset_processor.multimodal_preprocessor
        and hasattr(dataset_processor, "image_patch_start_idx")
        and dataset_processor.image_patch_start_idx is not None
    ):
        with open(json_params_file, "r") as _fin:
            data = json.load(_fin)

        image_patch_info = dict()
        image_patch_info["image_patch_start_idx"] = int(
            dataset_processor.image_patch_start_idx
        )  # have to cast numpy to python int
        data["image_patch_info"] = image_patch_info

        with open(json_params_file, "w") as _fout:
            json.dump(data, _fout, indent=4, sort_keys=True)


@dataclass
class VerificationArgs:
    processes: int
    files_per_record: int
    max_seq_length: int
    tokenizer_obj: object
    eos_id: int
    pad_id: int
    num_features: int


def get_verification_args(processes, data_processor):
    """Get arguments for verifying HDF5 dataset.
    Args:
        params (dict): Dictionary containing parameters for verifying HDF5 dataset.
        data_processor: Class containing methods that specify how the dataset
            will be processed and written into HDF5 files.
    """
    verification_args = VerificationArgs(
        processes,
        data_processor.files_per_record,
        data_processor.max_seq_length,
        data_processor.tokenizer,
        data_processor.eos_id,
        data_processor.pad_id,
        data_processor.num_features,
    )

    return verification_args


def process_dataset(files, dataset_processor, processes):
    """Process a dataset and write it into HDF5 format.

    Args:
        files (list): List of files to process.
        dataset_processor: Class containing methods that specify how the dataset
            will be processed and written into HDF5 files.
        processes (int): Number of processes to use.

    Returns:
        Dictionary containing results of execution, specifically as number of
            processed, discarded, and successful files as well as number of examples
            from all processes.
    """
    if processes < 2:
        # Run only single process run, with process number set as 0.
        return dataset_processor.create_dataset((files, 0))

    try:
        n_proc = processes
        n_chunks = ceil(len(files) / n_proc)
        remain = len(files) % n_proc
        if n_chunks == 1 and remain:
            n_proc = remain
            logger.warning(
                f"There aren't enough files to distribute to {processes} "
                f"processes, resetting it to {n_proc}. If you're working with a "
                "small number of compressed archives and could extract it into "
                "txt files, you might be able to get more benefits from the "
                f"available {processes} processes."
            )
        files = split_list(files, n_chunks)
    except ValueError as e:
        # We hit errors in two potential scenarios,
        # 1) Files is an empty list, in which case there is nothing to split
        # 2) There are more processes than files, in which case we cannot split
        #    the files to processes correctly, as there will be many idle
        #    processes which are not doing anything.
        logger.error(e)
        raise

    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(
                dataset_processor.create_dataset,
                zip(
                    files,
                    range(len(files)),
                ),
            ),
            total=len(files),
        )
        meta = {
            "discarded": 0,
            "processed": 0,
            "successful": 0,
            "examples": 0,
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
        }
        for results in pbar:
            pbar.update()
            for k, v in results.items():
                meta[k] += v

        return meta


@dataclass
class DatasetStats:
    num_sequences: int
    num_tokens: int
    detokenized_bytes: int
    detokenized_chars: int
    non_pad_tokens: int
    loss_valid_tokens: int


def collect_stats(data_arr, args):
    """Collect statistics of the dataset.

    Args:
        data_arr (numpy.ndarray): Numpy array containing the dataset.
        args (ValidationArgs): Arguments for verifying HDF5 dataset.
    """
    num_sequences = data_arr.shape[0]
    num_tokens = data_arr.shape[0] * data_arr.shape[2]
    non_pad_tokens = np.logical_and(
        data_arr[:, 0, :] != args.eos_id, data_arr[:, 0, :] != args.pad_id
    ).sum()
    loss_valid_tokens = data_arr[:, 1, :].sum()
    detokenized_bytes = 0
    detokenized_chars = 0

    for i in range(data_arr.shape[0]):
        line_str = args.tokenizer_obj.decode(data_arr[i, 0, :])
        detokenized_bytes += len(line_str.encode("utf-8"))
        detokenized_chars += len(line_str)

    return DatasetStats(
        num_sequences,
        num_tokens,
        detokenized_bytes,
        detokenized_chars,
        int(non_pad_tokens),  # cast to int to support saving to json
        int(loss_valid_tokens),  # cast to int to support saving to json
    )


def verify_saved_hdf5_files(params):
    """
    This function is used to do sanity checks at the end of the creation
    of hdf5 files.
    This function loads every .h5 files generated and checks:
        1. The data type
        2. Shape of the dataset
        3. Fact that labels and inputs are as expected
    """
    h5_files_path, args, vocab_size = params
    h5_stats = DatasetStats(
        0, 0, 0, 0, 0, 0
    )  # stats over list of files in a process
    for h5_file_path in h5_files_path:
        with h5py.File(h5_file_path, mode="r") as h5_file:
            n_examples = h5_file.attrs["n_examples"]
            dataset = h5_file["data"]
            data_arr = dataset[()]
            expected_dtype = "i4"
            expected_shape = (args.num_features, args.max_seq_length)
            assert dataset.dtype == expected_dtype, (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"datatype is unexpected. Expected: {expected_dtype}, "
                f"received {dataset.dtype}."
            )
            data_shape = data_arr.shape
            assert (
                data_shape[1:] == expected_shape or args.max_seq_length == -1
            ), (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"shape of example is unexpected. Expected:"
                f" {expected_shape}, received {data_shape[1:]}."
            )
            assert (data_arr < vocab_size).all(), (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"input ids are greater than vocab size."
                f"Please ensure that a correct tokenizer is used "
                f"and the eos_id and pad_id are correct within the "
                f"tokenizer vocabulary size."
            )
            file_stats = collect_stats(data_arr, args)
            assert n_examples == file_stats.num_sequences, (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"number of examples in file is unexpected. Expected:"
                f" {n_examples}, collected {file_stats.num_sequences}."
            )
            assert file_stats.num_tokens == n_examples * args.max_seq_length, (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"number of tokens in file is unexpected. Expected:"
                f" {n_examples * args.max_seq_length}, collected "
                f"{file_stats.num_tokens}."
            )
        h5_stats.num_sequences += file_stats.num_sequences
        h5_stats.num_tokens += file_stats.num_tokens
        h5_stats.detokenized_bytes += file_stats.detokenized_bytes
        h5_stats.detokenized_chars += file_stats.detokenized_chars
        h5_stats.non_pad_tokens += file_stats.non_pad_tokens
        h5_stats.loss_valid_tokens += file_stats.loss_valid_tokens

    return h5_stats


def verify_saved_hdf5_files_mp(files, args, vocab_size):
    """Verify the generated HDF5 dataset.
    Args:
        files (list): List of files to process.
        args (VerificationArgs): Arguments for verifying HDF5 dataset.
        vocab_size (int): Size of the vocabulary from data_processor.
    """
    if args.processes == 1:
        return verify_saved_hdf5_files((files, args, vocab_size))

    try:
        n_proc = args.processes
        n_chunks = ceil(len(files) / n_proc)
        remain = len(files) % n_proc
        if n_chunks == 1 and remain:
            n_proc = remain
            logger.warning(
                f"There aren't enough files to distribute to {args.processes} "
                f"processes, resetting it to {n_proc}."
            )
        files = split_list(files, n_chunks)
    except ValueError as e:
        ## In this case files is an empty list. This happens if no output hdf5 file is created.
        ## This may happen when the output preprocessed dataset is too small to fit in 1 hdf5 file and write_remainder = False
        logger.info(
            "No output hdf5 files are created. This \
        may happen when the output preprocessed dataset is too small to fit in 1 hdf5 file and write_remainder = False\
        Change write_remainder flag to True to get output hdf5 files."
        )
        return DatasetStats(0, 0, 0, 0, 0, 0)

    dataset_stats = DatasetStats(0, 0, 0, 0, 0, 0)

    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            desc="Verifying HDF5 files",
            total=len(files),
        )
        for stats in pool.imap(
            verify_saved_hdf5_files,
            zip(
                files,
                repeat(args),
                repeat(vocab_size),
            ),
        ):
            dataset_stats.num_sequences += stats.num_sequences
            dataset_stats.num_tokens += stats.num_tokens
            dataset_stats.detokenized_bytes += stats.detokenized_bytes
            dataset_stats.detokenized_chars += stats.detokenized_chars
            dataset_stats.non_pad_tokens += stats.non_pad_tokens
            dataset_stats.loss_valid_tokens += stats.loss_valid_tokens
            pbar.update()

    return dataset_stats


def handle_jsonl(
    jsonl_reader, get_meta, autojoin_paragraphs, para_joiner, key=None
):
    for ob in jsonl_reader:
        # naive jsonl where each object is just the string itself, with no meta. For legacy compatibility.
        if isinstance(ob, str):
            assert not get_meta
            yield ob
            continue

        if key == None:
            yield ob
        else:
            text = ob[key]

            if autojoin_paragraphs and isinstance(text, list):
                text = para_joiner.join(text)
            if isinstance(text, numbers.Number):
                text = str(text)
            if get_meta:
                yield text, (ob['meta'] if 'meta' in ob else {})
            else:
                yield text


# Slightly modified version of the Reader class from lm_dataformat.
# from https://github.com/leogao2/lm_dataformat/blob/master/lm_dataformat/__init__.py
class Reader:
    def __init__(self, in_path, tokenizable_columns, multi_turn=False):
        self.in_path = in_path
        ## required for reading parquet data
        self.tokenizable_columns = tokenizable_columns
        self.autojoin_paragraphs = not multi_turn

    def stream_data(self, get_meta=False):
        self.f_name = ""
        files = listdir_or_file(self.in_path)
        jsonl_key = self.tokenizable_columns.get('jsonl_key', None)
        prompt_key = self.tokenizable_columns.get('prompt_key', None)
        completion_key = self.tokenizable_columns.get('completion_key', None)

        if not files:
            raise FileNotFoundError(f"No valid file(s) found in {self.in_path}")

        for f in files:
            self.f_name = f
            if f.endswith('.jsonl'):
                yield from self.read_jsonl(f, get_meta, key=jsonl_key)
            elif f.endswith('.jsonl.zst'):
                yield from self.read_jsonl_zst(f, get_meta, key=jsonl_key)
            elif f.endswith('.jsonl.zst.tar'):
                yield from self.read_jsonl_tar(f, get_meta, key=jsonl_key)
            elif f.endswith('.json.zst'):
                assert not get_meta

                yield from self.read_json(f)
            elif f.endswith('.txt'):
                assert not get_meta

                yield from self.read_txt(f)
            elif f.endswith('.json.gz'):
                assert not get_meta

                yield from self.read_jsongz(f)
            elif f.endswith('parquet'):
                yield from self.read_parquet(
                    f,
                    jsonl_key=jsonl_key,
                    prompt_key=prompt_key,
                    completion_key=completion_key,
                )
            else:
                # shouldn't be reached
                print(
                    f'Skipping {f} as streaming for that filetype is not implemented'
                )

    def read_txt(self, file):
        with open(file, 'r') as fh:
            yield fh.read()

    def read_gz(self, file):
        with gzip.open(file, 'rb') as f:
            for line in f:
                yield line.decode('utf-8')

    def read_jsongz(self, file):
        for line in self.read_gz(file):
            yield json.loads(line)

    def read_json(self, file):
        with open(file, 'rb') as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = cctx.stream_reader(fh)
            ob = json.load(reader)
            yield from ob

    def read_jsonl(
        self,
        file,
        get_meta=False,
        para_joiner='\n\n',
        key=None,
    ):
        with jsonlines.open(file) as rdr:
            yield from handle_jsonl(
                rdr, get_meta, self.autojoin_paragraphs, para_joiner, key
            )

    def read_parquet(
        self, file, jsonl_key=None, prompt_key=None, completion_key=None
    ):
        source = pq.ParquetFile(file)
        num_row_groups = source.num_row_groups
        for idx in range(num_row_groups):
            table = source.read_row_group(
                idx
            )  # Read the table outside of the blocks
            if jsonl_key:
                for cell in table.column(jsonl_key):
                    yield str(cell.as_py())
            elif prompt_key and completion_key:
                doc = {}
                zipped_columns = zip(
                    table.column(prompt_key), table.column(completion_key)
                )
                for prompt, completion in zipped_columns:
                    yield {
                        prompt_key: str(prompt.as_py()),
                        completion_key: str(completion.as_py()),
                    }
            else:
                ## the file is corrupted. So return empty doc
                yield {}

    def read_jsonl_zst(
        self,
        file,
        get_meta=False,
        para_joiner='\n\n',
        key=None,
    ):
        with open(file, 'rb') as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            yield from handle_jsonl(
                rdr, get_meta, self.autojoin_paragraphs, para_joiner, key
            )

    def read_jsonl_tar(
        self,
        file,
        get_meta=False,
        para_joiner='\n\n',
        key=None,
    ):
        with open(file, 'rb') as fh:
            for f in tarfile_reader(fh, streaming=True):
                cctx = zstandard.ZstdDecompressor()
                reader = io.BufferedReader(cctx.stream_reader(f))
                rdr = jsonlines.Reader(reader)
                yield from handle_jsonl(
                    rdr, get_meta, self.autojoin_paragraphs, para_joiner, key
                )
                f.close()


def validate_tokens(tokens, min_len=2):
    is_valid = len(tokens) >= min_len
    if not is_valid:
        logger.warning(
            f"token_ids must have at least {min_len} elements, skipping this example..."
        )
    return is_valid


def create_features_auto_lm(
    token_ids,
    max_sequence_length,
    short_seq_prob=0,
    inverted_mask=False,
    pad_id=0,
    min_len=10,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    rng=None,
):
    """Given a list of token_ids, generate input sequence and labels.

    Args:
        token_ids (sequence): List containing token ids for creating features,
            labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        short_seq_prob (float): Probability of generating short sequences from
            data. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        rng (random.Random obj): Instance of random object, with states set.
            Defaults to `None`.

    Returns:
        Tuple containing features and labels
    """
    if not validate_tokens(token_ids, min_len=min_len):
        return []

    if rng.random() < short_seq_prob:
        token_ids = token_ids[0 : rng.randint(2, max_sequence_length - 1)]

    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = [1] * len(input_ids)

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad

    input_ids.extend(padding)
    labels.extend(padding)
    input_mask.extend([0] * num_pad)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # create feature dict
    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


def create_features_summarization(
    prompt_ids,
    completion_ids,
    max_sequence_length,
    eos_id=0,
    sep_id=None,
    pad_id=0,
    min_len=10,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Given a list of prompt_ids and completion_ids, generate input sequence
    and labels.

    Args:
        prompt_ids (sequence): List containing token ids for the prompt to
            create features,labels and input mask from.
        completion_ids (sequence): List containing token ids for the completion
            create features,labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        pad_id (int): Id for pad token. Defaults to `0`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
    """

    # extra <EOS>
    total_len = len(prompt_ids) + len(completion_ids) + 1
    if sep_id is not None:
        total_len += 1
    if total_len > max_sequence_length:
        logger.warning(
            "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
        )
        return []
    if total_len < min_len:
        logger.warning(
            "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
        )
        return []

    token_ids = prompt_ids
    if sep_id is not None:
        token_ids = token_ids + [sep_id]
    token_ids = token_ids + completion_ids + [eos_id]

    token_mask = [0] * (len(prompt_ids))
    if sep_id is not None:
        token_mask += [1]
    else:
        # if no sep_id, prediction starts at the last token of prompt_ids
        token_mask[-1] = 1
    token_mask += [1] * len(completion_ids)
    token_mask += [0]  # EOS

    # add padding
    token_ids_pad = max_sequence_length + 1 - len(token_ids)
    input_mask_pad = max_sequence_length - len(token_mask)

    token_ids.extend([pad_id] * token_ids_pad)
    token_mask.extend([0] * input_mask_pad)

    input_ids = token_ids[:-1]
    labels = token_ids[1:]

    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(token_mask) == max_sequence_length
    ), "Wrong sequence length"

    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(token_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


def create_features_auto_lm_vsl(
    bin,
    max_sequence_length,
    num_pad,
    pad_id=0,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    attention_span_dtype="int32",
    position_ids_dtype="int32",
):
    """Given a list of VSL sequences, generate input features and labels.

    Args:
        bin (list(sequence)): list of VSL sequences.
        max_sequence_length (int): Maximum sequence length for data writes.
        num_pad (int): number of padding tokens in the sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        attention_span_dtype (str): Dtype as string for keys attention span in VSL.
            Defaults to `int32`.
        position_ids_dtype (str): Dtype as string for position ids and
            attention span in VSL. Defaults to `int32`.

    Returns:
        Tuple containing features and labels
    """
    input_ids, labels, attention_span, position_ids = [], [], [], []
    for sample in bin:
        input_ids.extend(sample[:-1])


def validate_tokens(tokens, min_len=2):
    is_valid = len(tokens) >= min_len
    if not is_valid:
        logger.warning(
            f"token_ids must have at least {min_len} elements, skipping this example..."
        )
    return is_valid


def create_features_auto_lm(
    token_ids,
    max_sequence_length,
    short_seq_prob=0,
    inverted_mask=False,
    pad_id=0,
    min_len=10,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    rng=None,
):
    """Given a list of token_ids, generate input sequence and labels.

    Args:
        token_ids (sequence): List containing token ids for creating features,
            labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        short_seq_prob (float): Probability of generating short sequences from
            data. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        rng (random.Random obj): Instance of random object, with states set.
            Defaults to `None`.

    Returns:
        Tuple containing features and labels
    """
    if not validate_tokens(token_ids, min_len=min_len):
        return []

    if rng.random() < short_seq_prob:
        token_ids = token_ids[0 : rng.randint(2, max_sequence_length - 1)]

    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = [1] * len(input_ids)

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad

    input_ids.extend(padding)
    labels.extend(padding)
    input_mask.extend([0] * num_pad)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # create feature dict
    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


def create_features_summarization(
    prompt_ids,
    completion_ids,
    max_sequence_length,
    eos_id=0,
    sep_id=None,
    pad_id=0,
    min_len=10,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Given a list of prompt_ids and completion_ids, generate input sequence
    and labels.

    Args:
        prompt_ids (sequence): List containing token ids for the prompt to
            create features,labels and input mask from.
        completion_ids (sequence): List containing token ids for the completion
            create features,labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        pad_id (int): Id for pad token. Defaults to `0`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
    """

    # extra <EOS>
    total_len = len(prompt_ids) + len(completion_ids) + 1
    if sep_id is not None:
        total_len += 1
    if total_len > max_sequence_length:
        logger.warning(
            "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
        )
        return []
    if total_len < min_len:
        logger.warning(
            "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
        )
        return []

    token_ids = prompt_ids
    if sep_id is not None:
        token_ids = token_ids + [sep_id]
    token_ids = token_ids + completion_ids + [eos_id]

    token_mask = [0] * (len(prompt_ids))
    if sep_id is not None:
        token_mask += [1]
    else:
        # if no sep_id, prediction starts at the last token of prompt_ids
        token_mask[-1] = 1
    token_mask += [1] * len(completion_ids)
    token_mask += [0]  # EOS

    # add padding
    token_ids_pad = max_sequence_length + 1 - len(token_ids)
    input_mask_pad = max_sequence_length - len(token_mask)

    token_ids.extend([pad_id] * token_ids_pad)
    token_mask.extend([0] * input_mask_pad)

    input_ids = token_ids[:-1]
    labels = token_ids[1:]

    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(token_mask) == max_sequence_length
    ), "Wrong sequence length"

    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(token_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


def create_features_auto_lm_vsl(
    bin,
    max_sequence_length,
    num_pad,
    pad_id=0,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    attention_span_dtype="int32",
    position_ids_dtype="int32",
):
    """Given a list of VSL sequences, generate input features and labels.

    Args:
        bin (list(sequence)): list of VSL sequences.
        max_sequence_length (int): Maximum sequence length for data writes.
        num_pad (int): number of padding tokens in the sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        attention_span_dtype (str): Dtype as string for keys attention span in VSL.
            Defaults to `int32`.
        position_ids_dtype (str): Dtype as string for position ids and
            attention span in VSL. Defaults to `int32`.

    Returns:
        Tuple containing features and labels
    """
    input_ids, labels, attention_span, position_ids = [], [], [], []
    for sample in bin:
        input_ids.extend(sample[:-1])


def validate_tokens(tokens, min_len=2):
    is_valid = len(tokens) >= min_len
    if not is_valid:
        logger.warning(
            f"token_ids must have at least {min_len} elements, skipping this example..."
        )
    return is_valid


def create_features_auto_lm(
    token_ids,
    max_sequence_length,
    short_seq_prob=0,
    inverted_mask=False,
    pad_id=0,
    min_len=10,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    rng=None,
):
    """Given a list of token_ids, generate input sequence and labels.

    Args:
        token_ids (sequence): List containing token ids for creating features,
            labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        short_seq_prob (float): Probability of generating short sequences from
            data. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        rng (random.Random obj): Instance of random object, with states set.
            Defaults to `None`.

    Returns:
        Tuple containing features and labels
    """
    if not validate_tokens(token_ids, min_len=min_len):
        return []

    if rng.random() < short_seq_prob:
        token_ids = token_ids[0 : rng.randint(2, max_sequence_length - 1)]

    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = [1] * len(input_ids)

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad

    input_ids.extend(padding)
    labels.extend(padding)
    input_mask.extend([0] * num_pad)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # create feature dict
    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


def create_features_summarization(
    prompt_ids,
    completion_ids,
    max_sequence_length,
    eos_id=0,
    sep_id=None,
    pad_id=0,
    min_len=10,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Given a list of prompt_ids and completion_ids, generate input sequence
    and labels.

    Args:
        prompt_ids (sequence): List containing token ids for the prompt to
            create features,labels and input mask from.
        completion_ids (sequence): List containing token ids for the completion
            create features,labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        pad_id (int): Id for pad token. Defaults to `0`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
    """

    # extra <EOS>
    total_len = len(prompt_ids) + len(completion_ids) + 1
    if sep_id is not None:
        total_len += 1
    if total_len > max_sequence_length:
        logger.warning(
            "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
        )
        return []
    if total_len < min_len:
        logger.warning(
            "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
        )
        return []

    token_ids = prompt_ids
    if sep_id is not None:
        token_ids = token_ids + [sep_id]
    token_ids = token_ids + completion_ids + [eos_id]

    token_mask = [0] * (len(prompt_ids))
    if sep_id is not None:
        token_mask += [1]
    else:
        # if no sep_id, prediction starts at the last token of prompt_ids
        token_mask[-1] = 1
    token_mask += [1] * len(completion_ids)
    token_mask += [0]  # EOS

    # add padding
    token_ids_pad = max_sequence_length + 1 - len(token_ids)
    input_mask_pad = max_sequence_length - len(token_mask)

    token_ids.extend([pad_id] * token_ids_pad)
    token_mask.extend([0] * input_mask_pad)

    input_ids = token_ids[:-1]
    labels = token_ids[1:]

    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(token_mask) == max_sequence_length
    ), "Wrong sequence length"

    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(token_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


def create_features_auto_lm_vsl(
    bin,
    max_sequence_length,
    num_pad,
    pad_id=0,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    attention_span_dtype="int32",
    position_ids_dtype="int32",
):
    """Given a list of VSL sequences, generate input features and labels.

    Args:
        bin (list(sequence)): list of VSL sequences.
        max_sequence_length (int): Maximum sequence length for data writes.
        num_pad (int): number of padding tokens in the sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        attention_span_dtype (str): Dtype as string for keys attention span in VSL.
            Defaults to `int32`.
        position_ids_dtype (str): Dtype as string for position ids and
            attention span in VSL. Defaults to `int32`.

    Returns:
        Tuple containing features and labels
    """
    input_ids, labels, attention_span, position_ids = [], [], [], []
    for sample in bin:
        input_ids.extend(sample[:-1])
        labels.extend(sample[1:])
        sample_len = len(sample) - 1
        attention_span.extend(list(range(sample_len - 1, -1, -1)))
        position_ids.extend(list(range(sample_len)))

    input_mask = [1] * len(input_ids)

    # padding
    padding = [pad_id] * num_pad
    input_ids.extend(padding)
    labels.extend(padding)

    padding = [0] * num_pad
    input_mask.extend(padding)
    attention_span.extend(padding)
    position_ids.extend(padding)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
        and len(attention_span) == max_sequence_length
        and len(position_ids) == max_sequence_length
    ), "Wrong sequence length"

    input_ids = getattr(np, input_ids_dtype)(input_ids)
    input_mask = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask.dtype)

    labels = getattr(np, labels_dtype)(labels)
    attention_span = getattr(np, attention_span_dtype)(attention_span)
    position_ids = getattr(np, position_ids_dtype)(position_ids)

    return np.stack(
        [input_ids, input_mask, labels, attention_span, position_ids]
    )


class DocObject:
    def __init__(self, prompt_comp_pairs, multi_modal=False, img_path=None):
        self.prompt_comp_pairs = prompt_comp_pairs
        self.multi_modal = multi_modal
        self.img_path = img_path
        self.tokens = []


def create_features_summarization_vsl(
    bin,
    max_sequence_length,
    completion_prefix_mask_len=0,
    pad_id=0,
    eos_id=0,
    eos_after_prompt=False,
    sep_id=None,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    attention_span_dtype="int32",
    position_ids_dtype="int32",
):
    """Given a list of VSL sequences, generate input features and labels.

    Args:
        bin (list(sequence)): list of VSL sequences.
        max_sequence_length (int): Maximum sequence length for data writes.
        num_pad (int): number of padding tokens in the sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        attention_span_dtype (str): Dtype as string for keys attention span in VSL.
            Defaults to `int32`.
        position_ids_dtype (str): Dtype as string for position ids in VSL.
            Defaults to `int32`.

    Returns:
        Tuple containing features and labels
    """
    input_ids, input_mask, labels, attention_span, position_ids = (
        [],
        [],
        [],
        [],
        [],
    )
    for doc_obj in bin:
        token_ids, token_mask = [], []
        for prompt_ids, completion_ids in doc_obj.tokens:
            if eos_after_prompt:
                prompt_ids = prompt_ids + [eos_id]
            if sep_id is not None:
                prompt_ids = prompt_ids + [sep_id]
            completion_ids += [eos_id]
            token_ids += prompt_ids + completion_ids

            token_mask += [0] * (len(prompt_ids) - 1)
            token_mask += [0] * completion_prefix_mask_len
            # start prediction on the last prompt token (including if it's sep or eos) or the last completion prefix token
            token_mask += [1]
            token_mask += [1] * (
                len(completion_ids) - completion_prefix_mask_len - 1
            )
            # don't want to learn from predicting next prompt after end of completion
            token_mask += [0]

        input_ids.extend(token_ids[:-1])
        labels.extend(token_ids[1:])
        input_mask.extend(token_mask[:-1])
        sample_len = len(token_ids) - 1
        attention_span.extend(list(range(sample_len - 1, -1, -1)))
        position_ids.extend(list(range(sample_len)))

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad
    input_ids.extend(padding)
    labels.extend(padding)

    padding = [0] * num_pad
    input_mask.extend(padding)
    attention_span.extend(padding)
    position_ids.extend(padding)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
        and len(attention_span) == max_sequence_length
        and len(position_ids) == max_sequence_length
    ), "Wrong sequence length"

    input_ids = getattr(np, input_ids_dtype)(input_ids)
    input_mask = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask.dtype)

    labels = getattr(np, labels_dtype)(labels)
    attention_span = getattr(np, attention_span_dtype)(attention_span)
    position_ids = getattr(np, position_ids_dtype)(position_ids)

    return np.stack(
        [input_ids, input_mask, labels, attention_span, position_ids]
    )


def create_features_llava_phase1(
    doc_obj,
    max_sequence_length,
    num_patches=None,
    pad_id=0,
    eos_id=0,
    bos_id=None,
    eos_after_prompt=False,
    sep_id=None,
    inverted_mask=False,
    handle_default_bos_token=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """Given a list of VSL sequences, generate input features and labels.

    Args:
        bin (list(sequence)): list of VSL sequences.
        max_sequence_length (int): Maximum sequence length for data writes.
        num_pad (int): number of padding tokens in the sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.

    Returns:
        Tuple containing features and labels
    """
    (
        input_ids,
        input_mask,
        labels,
        attention_mask,
    ) = (
        [],
        [],
        [],
        [],
    )

    token_ids, text_token_mask = [], []
    if doc_obj.multi_modal:
        img_attn_mask_value = int(doc_obj.img_path is not None)
        if handle_default_bos_token:
            token_ids += [bos_id]
            # if there is an image we don't want to have loss on it,
            # but we do for text
            input_mask += [1 - img_attn_mask_value]
            attention_mask += [1]

        token_ids.extend([pad_id] * num_patches)
        # we never have loss over the image patches
        img_loss_mask = [0] * (num_patches - 1)
        img_loss_mask += [img_attn_mask_value]
        # we do attend to the image patches if there is an image

        img_attn_mask = [img_attn_mask_value] * num_patches
        attention_mask.extend(img_attn_mask)

    for prompt_ids, completion_ids in doc_obj.tokens:
        null_prompt = len(prompt_ids) == 0  # LLaVA Phase 1

        if eos_after_prompt and not null_prompt:
            prompt_ids = prompt_ids + [eos_id]
        if sep_id is not None and not null_prompt:
            prompt_ids = prompt_ids + [sep_id]

        completion_ids += [eos_id]

        token_ids += prompt_ids + completion_ids

        text_token_mask += [0] * (len(prompt_ids) - 1)
        # start prediction on the last prompt token (including if it's sep or eos)
        if not null_prompt:
            text_token_mask += [1]
        text_token_mask += [1] * (len(completion_ids) - 1)
        # don't want to learn from predicting next prompt after end of completion
        text_token_mask += [0]

    input_ids.extend(token_ids[:-1])
    labels.extend(token_ids[1:])

    if attention_mask:
        attention_mask = attention_mask + text_token_mask
        attention_mask = attention_mask[:-1]

    token_mask = img_loss_mask + text_token_mask
    input_mask.extend(token_mask[:-1])

    sample_len = len(token_ids) - 1

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad
    input_ids.extend(padding)
    labels.extend(padding)

    padding = [0] * num_pad
    input_mask.extend(padding)

    if attention_mask:
        attention_mask.extend(padding)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # used for attending to image or masking out image placeholder tokens for
    # multimodal examples
    if attention_mask is not None:
        assert len(attention_mask) == max_sequence_length
        attention_mask = getattr(np, input_ids_dtype)(attention_mask)

    input_ids = getattr(np, input_ids_dtype)(input_ids)
    input_mask = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask.dtype)

    # NOTE this is because our internal stack requires the inverted mask and
    # doesn't do the inversion internally
    attention_mask = np.equal(attention_mask, 0).astype(input_mask.dtype)

    labels = getattr(np, labels_dtype)(labels)

    out = [
        input_ids,
        input_mask,
        labels,
        attention_mask,
    ]
    out = [x for x in out if x is not None]
    return np.stack(out)


def create_features_llava_phase2(
    doc_obj,
    system_prompt_toks,
    max_sequence_length,
    num_patches=None,
    user_prompt_tok_len=None,
    asst_prompt_tok_len=None,
    eos_id=0,
    pad_id=0,
    bos_id=None,
    space_id=None,
    eos_after_prompt=False,
    sep_id=None,
    inverted_mask=False,
    handle_default_bos_token=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """Given a list of VSL sequences, generate input features and labels.

    Args:
        bin (list(sequence)): list of VSL sequences.
        max_sequence_length (int): Maximum sequence length for data writes.
        num_pad (int): number of padding tokens in the sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.

    Returns:
        Tuple containing features and labels
    """
    (
        input_ids,
        input_mask,
        labels,
        attention_mask,
    ) = (
        [],
        [],
        [],
        [],
    )

    token_ids, text_token_mask = [], []
    if len(system_prompt_toks) > 0:
        token_ids.extend(system_prompt_toks)
        text_token_mask.extend([0] * len(system_prompt_toks))
        attention_mask.extend([1] * len(system_prompt_toks))

    for i, (prompt_ids, completion_ids) in enumerate(doc_obj.tokens):
        null_prompt = len(prompt_ids) == 0  # LLaVA Phase 1
        img_loss_mask, img_attn_mask = [], []
        if eos_after_prompt and not null_prompt:
            prompt_ids = prompt_ids + [eos_id]
        if sep_id is not None and not null_prompt:
            prompt_ids = prompt_ids + [sep_id]

        if doc_obj.multi_modal and i == 0:
            import copy

            orig_prompt_ids = copy.deepcopy(prompt_ids)
            user_prompt_ids = prompt_ids[:user_prompt_tok_len]
            text_token_mask += [0] * len(user_prompt_ids)
            attention_mask += [1] * len(user_prompt_ids)

            prompt_ids = prompt_ids[user_prompt_tok_len:]
            img_attn_mask_value = int(doc_obj.img_path is not None)
            if handle_default_bos_token and (len(system_prompt_toks) == 0):
                token_ids += [bos_id]
                # if there is an image we don't want to have loss on it,
                # but we do for text
                input_mask += [0]
                attention_mask += [1]

            token_ids.extend(user_prompt_ids)
            if space_id:
                token_ids += [space_id]
                img_loss_mask = [0]
                img_attn_mask = [img_attn_mask_value]

            token_ids.extend([pad_id] * num_patches)
            # we never have loss over the image patches
            img_loss_mask += [0] * (num_patches)
            text_token_mask.extend(img_loss_mask)
            # we do attend to the image patches if there is an image
            img_attn_mask += [img_attn_mask_value] * (num_patches)
            attention_mask.extend(img_attn_mask)

        completion_ids += [eos_id]

        token_ids += prompt_ids + completion_ids

        text_token_mask += [0] * (len(prompt_ids) - 1)
        # start prediction on the last prompt token (including if it's sep or eos)
        if not null_prompt:
            text_token_mask += [0]

        if asst_prompt_tok_len != 0:
            text_token_mask += [0] * (asst_prompt_tok_len - 1)
            text_token_mask += [1] * (len(completion_ids) - asst_prompt_tok_len)
        else:
            text_token_mask += [1] * (len(completion_ids) - 1)

        # don't want to learn from predicting next prompt after end of completion
        text_token_mask += [0]

        attention_mask.extend([1] * (len(prompt_ids) + len(completion_ids)))

    input_ids.extend(token_ids[:-1])
    labels.extend(token_ids[1:])
    input_mask.extend(text_token_mask[:-1])
    attention_mask = attention_mask[:-1]

    # padding
    num_pad = max_sequence_length - len(input_ids)

    if num_pad < 0:
        return []

    padding = [pad_id] * num_pad
    input_ids.extend(padding)
    labels.extend(padding)

    padding = [0] * num_pad
    input_mask.extend(padding)

    if attention_mask:
        attention_mask.extend(padding)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # used for attending to image or masking out image placeholder tokens for
    # multimodal examples
    if attention_mask is not None:
        assert len(attention_mask) == max_sequence_length
        attention_mask = getattr(np, input_ids_dtype)(attention_mask)

    input_ids = getattr(np, input_ids_dtype)(input_ids)
    input_mask = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask.dtype)

    # NOTE this is because our internal stack requires the inverted mask and
    # doesn't do the inversion internally
    attention_mask = np.equal(attention_mask, 0).astype(input_mask.dtype)

    labels = getattr(np, labels_dtype)(labels)

    out = [
        input_ids,
        input_mask,
        labels,
        attention_mask,
    ]
    out = [x for x in out if x is not None]
    return np.stack(out)


def get_files(input_dir=None, filetypes=None, metadata_files=None):
    """Get all files of given filetypes from input directory.

    Args:
        input_dir (str): Input directory to read files from.
        filetypes (list): File types to fetch from the given input
            directory. Defaults to `None`.
        metadata_files (str): Comma separated string of metadata files.

    Returns:
        List of lists containing all file paths as strings
    """
    if not filetypes:
        filetypes = [
            '.jsonl',
            '.json.gz',
            '.jsonl.zst',
            '.jsonl.zst.tar',
            '.txt',
            '.parquet',
        ]
    if isinstance(filetypes, str):
        filetypes = [filetypes]
    filetypes = tuple(filetypes)
    assert input_dir or metadata_files, (
        "User need to provide `input_dir` or `metadata_files`, "
        "but neither was provided."
    )
    if metadata_files:
        if isinstance(metadata_files, str):
            metadata_files = [metadata_files]

        if input_dir:
            logger.warning(
                "Both `input_dir` and `metadata_files` were provided, "
                "ignoring `input_dir` and using `metadata_files`."
            )

        input_files = []
        for _file in metadata_files:
            with open(_file, "r") as _fin:
                input_files.extend(_fin.readlines())

        input_files_list = [x.strip() for x in input_files if x]
        flattened_list = [x for x in input_files_list if x.endswith(filetypes)]
    else:
        files = [list(Path(input_dir).rglob(f"*{ft}")) for ft in filetypes]
        # flatten list of list -> list and stringify Paths
        flattened_list = [str(item) for sublist in files for item in sublist]
    if not flattened_list:
        raise Exception(
            f"Did not find any files at this path {input_dir}, please "
            f"ensure your files are in format {filetypes}."
        )
    return flattened_list


def wikitext_detokenizer(string):
    """Detokenizer for wikitext. Used for special handling of data for substrings.

    Args:
        string (str): String to detoknize before tokenization.

    Returns:
        Detokenized string
    """
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def read_checkpoint(checkpoint_path, resume_from_checkpoint=True):
    """Checkpoint reader for execution.

    Args:
        checkpoint_path (str): Path to read checkpoint data from
        resume_from_checkpoint (bool): Resume from checkpoint for execution.
            Defaults to `True`.

    Returns:
        Tuple containing number of files processed and the count of tfrecords/HDF5 files
            written to output directory.
    """
    if resume_from_checkpoint and os.path.isfile(checkpoint_path):
        try:
            resume_files_processed, count = [
                int(i) for i in open(checkpoint_path, "r").read().split(", ")
            ]
            logger.info(
                f"Resuming from file number: {count}, "
                f"with raw file number processed: {resume_files_processed}"
            )
            return resume_files_processed, count
        except Exception as e:
            # if checkpoint path is at initialization,
            # file may exist, but no data might be written in the file
            # in that event, do not do anything, go to the final return
            logger.error(e)
    return 0, 0


# routine to split the text into smaller sequences
def split_text_and_tokenize(
    text, tokenizer, max_tok_len=2000, remove_bos_in_chunks=True
):
    """Function to split the text into smaller sequences of length max_tok_len
    and then tokenize each of the smaller sequences. This is done to avoid
    performance issues with tokenizers like LlamaTokenizer which are slow for
    long sequences.

    Args:
        text (str): text to be tokenized
        tokenizer (Tokenizer): tokenizer to be used
        max_tok_len (int, optional): max length of each sequence. Defaults to 2000.
        remove_bos_in_chunks (bool, optional): whether to ignore bos token id in
            chunks. Defaults to True.
    Returns:
        tok_ids (list): list of token ids for the text
    """
    if len(text) == 0:
        return []

    curr_start = 0
    tok_ids = []

    while curr_start < len(text):
        curr_end = min(text.find(' ', curr_start + max_tok_len), len(text))
        if curr_end < 0:
            curr_substr = text[curr_start:]
            curr_end = len(text)
        else:
            curr_substr = text[curr_start:curr_end]
        if curr_start == 0:
            # keep special tokens for the first chunk
            bos_token_id = [tokenizer.encode(curr_substr)[0]]
        curr_tok_ids = (
            tokenizer.encode(curr_substr)[1:]
            if remove_bos_in_chunks
            else tokenizer.encode(curr_substr)
        )
        tok_ids.extend(curr_tok_ids)
        curr_start = curr_end
    # concatenated tok_ids chunks together by using `extend` to return full sequence of tokens

    # NOTE: add bos token id if it is needed here, eos id is added in the next line
    # which calls this function
    return bos_token_id + tok_ids if remove_bos_in_chunks else tok_ids


def chunk(
    sample,
    tokenizer,
    fim_rate,
    spm_rate,
):
    """
    Since we do character-level FIM we need to detokenize, determine boundaries
    to split, and re-tokenize after splitting. We chunk but do not shuffle and add
    special tokens because we might have to truncate or pad the tokens since they
    have been split at the character-level and re-tokenized, leading to potentially
    different lengths than the original sequence.
    If the sub-context is designated to be an AR (auto-regressive) sequence and not FIM, we store
    as [[], [], [sequence]] for convenience in the truncate_helper function.

    Args:
        sample (np.array):
        tokenizer (Tokenizer):
        fim_rate (float):
        spm_rate (float):

    Returns:
        List[List[int]], str: List of token lists corresponding to the
          prefix/middle/suffix tokens, or 2 empty lists plus the whole
          sequence in case of auto-regressive (AR) sequence. Also returns
          string representing the format of the sequence (i.e. SPM or
          PSM or AR)
    """
    if np.random.binomial(1, fim_rate):  # sample bernoulli dist
        contents = tokenizer.decode(sample, skip_special_tokens=False)
        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(
                np.random.randint(low=0, high=len(contents) + 1, size=2)
            )
            boundaries.sort()
        except ValueError as e:
            logging.info(len(contents))
            logging.info(contents)
            logging.info(e)
            raise e

        prefix = contents[: boundaries[0]]
        middle = contents[boundaries[0] : boundaries[1]]
        suffix = contents[boundaries[1] :]

        prefix = tokenizer.encode(prefix)
        middle = tokenizer.encode(middle)
        suffix = tokenizer.encode(suffix)

        is_spm = np.random.binomial(1, spm_rate)
        fim_format = "SPM" if is_spm else "PSM"
        return [prefix, middle, suffix], fim_format
    else:
        # don't do FIM preproc
        fim_format = "AR"
        return [[], [], sample.tolist()], fim_format


def format_fim(
    segment_fim_format_pairs,
    max_seq_len,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    eos_tok_id,
    opt_bos_tok_id,
):
    """
    Takes in list of prefix/middle/suffix token lists, along with respective FIM (or AR) formats.
    Performs the correct transformation according to the format, adding the special tokens
    and shuffling the sections, before concatenating everything together.

    Args:
        segments_fim_format_pairs (List[Tuple[List[List[int]], str]]): This list of tuples is used
        to store the prefix/middle/suffix token-id lists and the corresponding FIM formats (PSM/SPM) to
        be used downstream in the FIM formatting.
        max_seq_len (int): Max sequence length that each sequence is expected
          to match
        suffix_tok_id (int): Id for suffix token
        prefix_tok_id (int): Id for suffix token
        middle_tok_id (int): Id for suffix token
        eos_tok_id (int): Id for suffix token
        opt_bos_tok_id (list): Optionally a list containing the bos token id,
          otherwise will be empty list. Empty list will be a no-op in the
          concatenation. Bos-token will only exist if model's tokenizer adds
          bos-token by default. Both have to be lists so that np concat works

    Returns:
        sample (np.array): Array of token ids in the FIMed order
          along with special tokens
        mask (np.array): Array of 1's and 0's corresponding to true
          tokens and padding respectively
        label (np.array): Token i of label corresponds to token i+1 in
          sample array. Same elements except that label ends in eos
          (end-of-sequence) token
    """

    prefix_idx, middle_idx, suffix_idx = 0, 1, 2
    sample = []
    total_padding_len = 0
    for sample_i, fim_format in segment_fim_format_pairs:
        optional_padding = sample_i[-1] if len(sample_i) > 3 else []
        total_padding_len += len(optional_padding)
        if fim_format == "PSM":
            sample_i = np.concatenate(
                [
                    opt_bos_tok_id,
                    [prefix_tok_id],
                    sample_i[prefix_idx],
                    [suffix_tok_id],
                    sample_i[suffix_idx],
                    [middle_tok_id],
                    sample_i[middle_idx],
                    [eos_tok_id],
                ]
            )
        elif fim_format == "SPM":
            sample_i = np.concatenate(
                [
                    opt_bos_tok_id,
                    [prefix_tok_id, suffix_tok_id],
                    sample_i[suffix_idx],
                    [middle_tok_id],
                    sample_i[prefix_idx],
                    sample_i[middle_idx],
                    [eos_tok_id],
                ]
            )
        else:
            sample_i = np.concatenate(
                [
                    opt_bos_tok_id,
                    sample_i[prefix_idx],
                    sample_i[middle_idx],
                    sample_i[suffix_idx],
                    [eos_tok_id],
                ]
            )
        sample_i = np.concatenate([sample_i, optional_padding])
        sample.append(sample_i)
    sample = np.concatenate(sample).astype(np.int64)
    label = sample[1:]
    sample = sample[:-1]
    sample_mask = np.ones(max_seq_len - total_padding_len)
    padding_mask = np.zeros(total_padding_len)
    mask = np.concatenate([sample_mask, padding_mask])
    return sample, mask, label


def truncate_helper(samples_lst, diff, sample_idx):
    """
    The goal of our truncation scheme is to avoid removing tokens from the
    middle section. We first remove from the end of suffix, and then from the
    beginning of the prefix. We store the chunks in lists in the original order
    so that we can easily perform this truncation. Since each sub-context can have
    different amounts of tokens in suffix/prefix, we store unique indices for the
    section to remove from. If we run out of tokens to remove from, we switch to the next.
    This way we can switch to the prefix of one context while still removing from suffix
    of another. If the sub-context is AR (auto-regressive) and not FIM, the AR sequence
    is stored as [[], [], [sequence]] so that the remove_idx being 2 will simultaneously
    work for the AR and FIM sequences.

    Args:
        samples_lst (List[List[int]]): List of lists that contain token ids
        diff (int): Number of tokens to pad
        sample_idx (int): Index for the sample from the dataset, for use in
          logging if we remove from the middle.

    Returns:
        (List[List[int]]): List of lists of token ids that have been truncated
    """
    num_groups = len(samples_lst)
    remove_idxs = [2] * num_groups  # remove from suffixes first
    i = 0

    while diff:
        remove_idx_i = remove_idxs[i]
        sample_i = samples_lst[i]
        if sample_i[remove_idx_i]:
            pop_idx = (
                -1 if remove_idx_i == 2 else 0
            )  # remove from end of suffix but beginning of prefix
            sample_i[remove_idx_i].pop(pop_idx)
            diff -= 1
        else:
            remove_idxs[i] = (
                remove_idxs[i] + 1
            ) % 3  # order of removal is end of suffix, beginning of prefix, then beginning of middle
            if remove_idxs[i] == 1:
                logging.info(
                    f"""Context {i} in the {sample_idx}-th data sample has
                        begun truncating from the middle section, meaning
                        the prefix and suffix sections have been exhausted.
                      """
                )
        i = (i + 1) % num_groups

    return samples_lst


def pad_helper(samples_lst, diff, fim_pad_tok_id):
    """
    Helper for padding. We put all padding tokens into the last sequence.

    Args:
        samples_lst (List[List[int]]): List of lists that contain token ids
        diff (int): Number of tokens to pad
        fim_pad_tok_id (int): Id for padding token

    Returns:
        (List[List[int]]): List of lists of token ids with padding
    """
    padding = np.full(np.abs(diff), fim_pad_tok_id)
    samples_lst[-1].append(padding)
    return samples_lst


def truncate_or_pad_helper(
    segments_fim_format_pairs, diff, fim_pad_tok_id, sample_idx
):
    """
    Since we perform FIM at character-level, we potentially split characters
    in the middle of a word. This can lead to non-standard token sequences,
    and after re-tokenizing we might need to truncate or pad to get back to
    the original context length. This function ensures that our outputs are
    back at their original length.

    Args:
        segments_fim_format_pairs (List[Tuple[List[List[int]], str]]): This list of tuples is used
        to store the prefix/middle/suffix token-id lists and the corresponding FIM formats (PSM/SPM) to
        be used downstream in the FIM formatting.
        diff (int): The number of tokens to add or remove. Positive means truncate, negative means pad
        fim_pad_tok_id (int): Id of padding token

    Returs:
        (List[Tuple[List[List[int]], str]]): The element of the tuples will
        now be lists that are truncated or padded such that the concatenation of all these tokens, along
        with the special tokens, will be equal to the original sequence length.
    """
    segments = [pair[0] for pair in segments_fim_format_pairs]
    fim_formats = [pair[1] for pair in segments_fim_format_pairs]
    if diff >= 0:
        segments = truncate_helper(segments, diff, sample_idx)
    else:
        segments = pad_helper(segments, diff, fim_pad_tok_id)
    return [(segments[i], fim_formats[i]) for i in range(len(segments))]


def fim(
    sample_array,
    sample_idx,
    tokenizer,
    fim_rate,
    spm_rate,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    fim_pad_tok_id,
    eos_tok_id,
    opt_bos_tok_id,
):
    """
    Takes in an array of input_ids, mask, and labels, and performs the
    FIM operation to re-arrange into PSM and SPM format with some probability

    Args:
        sample_array (np.array): Stack of input_ids, mask, and labels after tokenization. Labels are off-by-one of input_ids
        as in standard auto-regressive training
        i (int): Index of sample from dataset, used for logging.
        tokenizer (Tokenizer): Tokenizer object
        fim_rate (float): Determines what percentage of contexts are FIM'ed
        spm_rate (float): Determines what percentage of FIM'ed contexts are in SPM format. 1 - spm_rate determines PSM
        suffix_tok_id (int): Id for special token denoting suffix section in a FIM'ed context
        prefix_tok_id (int): Id for special token denoting prefix section in a FIM'ed context
        middle_tok_id (int): Id for special token denoting middle section in a FIM'ed context
        fim_pad_tok_id (int): Id for padding
        eos_tok_id (int): Id for the end-of-seqence
        opt_bos_tok_id (list): Optionally a list containing the bos token id,
          otherwise will be empty list. Empty list will be a no-op in the
          concatenation. Bos-token will only exist if model's tokenizer adds
          bos-token by default.

    Returns:
        fim_outputs (np.array): Stack of input_ids, mask, and labels after FIM transformation. Mask and labels have been
        adjusted to still filter padding tokens and represent the following token, respectively.
    """
    assert (
        fim_rate <= 1 and fim_rate >= 0
    ), "FIM rate must be a probability 0 <= rate <= 1"
    sample = sample_array[0, :]
    mask = sample_array[1, :]
    max_seq_len = sample.shape[0]

    segment_breaks = np.argwhere(
        sample == eos_tok_id
    )  # split sample by document
    segments_fim_format_pairs = []
    if segment_breaks.shape != (0, 1):  # FIM each sub-context
        curr_start_position = 0
        for loc in np.nditer(segment_breaks):
            # Only permute non-empty segments.
            if loc - curr_start_position > 0:
                segments, fim_format = chunk(
                    sample=sample[curr_start_position:loc],
                    tokenizer=tokenizer,
                    fim_rate=fim_rate,
                    spm_rate=spm_rate,
                )
                segments_fim_format_pairs.append((segments, fim_format))
            curr_start_position = loc + 1  # jump over the EOD token
        # Permute the segment after the last EOD
        segments, fim_format = chunk(
            sample=sample[curr_start_position:],
            tokenizer=tokenizer,
            fim_rate=fim_rate,
            spm_rate=spm_rate,
        )
        segments_fim_format_pairs.append((segments, fim_format))
    else:  # FIM over full context
        segments, fim_format = chunk(
            sample=sample,
            tokenizer=tokenizer,
            fim_rate=fim_rate,
            spm_rate=spm_rate,
        )
        segments_fim_format_pairs.append((segments, fim_format))

    def flatten_2d(arr):
        return np.concatenate([np.concatenate(subarr) for subarr in arr])

    total_len = flatten_2d(
        [pair[0] for pair in segments_fim_format_pairs]
    ).shape[0]
    # we factor in the final EOS, which we add before splitting into
    # inputs and labels, i.e. sequence[:-1] and sequence[1:], and the
    # optional bos token
    add_constant = -1
    for _, fmt in segments_fim_format_pairs:
        if fmt == "AR":
            add_constant += 1
        else:
            add_constant += 4
        if opt_bos_tok_id:
            add_constant += 1
    diff = (total_len + add_constant) - max_seq_len
    segments_fim_format_pairs = truncate_or_pad_helper(
        segments_fim_format_pairs,
        diff,
        fim_pad_tok_id,
        sample_idx,
    )
    inputs, mask, labels = format_fim(
        segments_fim_format_pairs,
        max_seq_len,
        suffix_tok_id,
        prefix_tok_id,
        middle_tok_id,
        eos_tok_id,
        opt_bos_tok_id,
    )

    try:
        assert inputs.shape[0] == max_seq_len
        assert mask.shape[0] == max_seq_len
        assert labels.shape[0] == max_seq_len
    except:
        logging.error(
            "The inputs/masks/labels were not the correct\
                      sized after FIM process. Shapes of each are printed\
                      below, along with the correct max seqeunce length\
                      that each sequence should be."
        )
        logging.error(inputs.shape, max_seq_len)
        logging.error(mask.shape, max_seq_len)
        logging.error(labels.shape, max_seq_len)
        raise AssertionError
    try:
        assert labels[-1] == eos_tok_id
    except:
        logging.error("The sequence did not end with an EOS token")
        raise AssertionError
    # end FIM-specific code
    fim_outputs = np.stack([inputs, mask, labels], axis=0)
    return fim_outputs


def get_tokenizer_vocab(tokenizer):
    if isinstance(tokenizer, BPETokenizer):
        tokenizer_vocab = tokenizer.encoder
    elif isinstance(tokenizer, HFTokenizer):
        tokenizer_vocab = tokenizer.tokenizer.get_vocab()
    elif isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
        tokenizer, PreTrainedTokenizerFast
    ):
        tokenizer_vocab = tokenizer.vocab
    else:
        raise NotImplementedError(
            "We do not support specified tokenizer\
                                  type."
        )
    return tokenizer_vocab


def check_fim_special_tokens(params, tokenizer):
    # Check that input config lists the FIM special tokens
    assert (
        "fim_suffix_tok" in params['processing']
        and "fim_prefix_tok" in params['processing']
        and "fim_middle_tok" in params['processing']
    ), """Configs for FIM pre-processing must include the special tokens that
    denote prefix, middle, and suffix tokens."""
    # Check that the provided tokens are in the tokenizer
    pre_tok = params['processing'].get("fim_prefix_tok")
    mid_tok = params['processing'].get("fim_middle_tok")
    suf_tok = params['processing'].get("fim_suffix_tok")
    tokenizer_vocab = get_tokenizer_vocab(tokenizer)
    assert (
        pre_tok in tokenizer_vocab
        and mid_tok in tokenizer_vocab
        and suf_tok in tokenizer_vocab
    ), """Please ensure that the provided FIM special tokens are in the
    specified tokenizer."""


def handle_bos_token_default(tokenizer):
    """
    When performing FIM, we tokenize each chunk again after splitting.
    Therefore, if the tokenizer adds bos-token by default, we will get
    extra bos-tokens in the middle of the sequence. In this function,
    we set the tokenizer bos default to False, and return a flag that
    indicates whether we will need to add bos-token in the final
    fim formatting function.
    """
    if hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token:
        tokenizer.add_bos_token = False
        bos_tok_id = tokenizer.encode(tokenizer.bos_token)[-1]
        return True, [bos_tok_id]
    return False, []
