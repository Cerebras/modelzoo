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
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from cerebras.modelzoo.data_preparation.nlp.tokenizers.BPETokenizer import (
    BPETokenizer,
)
from cerebras.modelzoo.data_preparation.nlp.tokenizers.HFTokenizer import (
    HFTokenizer,
)

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
    '.fasta',
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


def dump_result(
    results,
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
    post_process["discarded_files"] = results.pop("discarded", 0)
    post_process["processed_files"] = results.pop("processed", 0)
    post_process["successful_files"] = results.pop("successful", 0)
    post_process["n_examples"] = results.pop("examples", 0)
    post_process["raw_chars_count"] = results.pop("raw_chars_count", 0)
    post_process["raw_bytes_count"] = results.pop("raw_bytes_count", 0)

    ## dump features for dpo to be used in DPODataProcessor
    if "features" in results:
        features = results.pop("features")
        data["features"] = features

    ## put remaining key,value pairs in post process
    for key, value in results.items():
        post_process[key] = value

    if eos_id is not None:
        post_process["eos_id"] = eos_id
    if pad_id is not None:
        post_process["pad_id"] = pad_id
    if vocab_size is not None:
        post_process["vocab_size"] = vocab_size

    data["post-process"] = post_process
    with open(json_params_file, "w") as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)


def dump_args(args, json_params_file):
    """
    Write the input params to file.
    """
    logger.info(f"User arguments can be found at {json_params_file}.")

    redundant_params = [
        "eos_id",
        "pad_id",
        "display_pbar",
        "files_per_record",
        "output_name",
        "write_remainder",
    ]

    relevant_args = copy.deepcopy(args)
    # Iterate through the dictionary and remove the redundant params
    for key in redundant_params:
        for sub_dict in relevant_args.values():
            if key in sub_dict:
                del sub_dict[key]

    # write initial params to file
    with open(json_params_file, "w") as _fout:
        json.dump(args, _fout, indent=4, sort_keys=True)


def get_parser(desc):
    """Argparser definition for command line arguments from user.

    Returns:
        Argparse namespace object with command line arguments.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file for setting dataset preprocessing hyper-parameters.",
    )
    return parser.parse_args()


def update_params(params, args):
    """
    Update config parameters with CLI arguments
    """
    setup_params = [
        "data",
        "metadata_files",
        "output_dir",
        "processes",
        "module",
        "dataset_processor",
        "mode",
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
        "resume_from_checkpoint",
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
    params["processing"]["resume_from_checkpoint"] = params["processing"].get(
        "resume_from_checkpoint", False
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

    params_file = args.pop("config", None)
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


def validate_tokens(tokens, min_len=2):
    is_valid = len(tokens) >= min_len
    if not is_valid:
        logger.warning(
            f"token_ids must have at least {min_len} elements, skipping this example..."
        )
    return is_valid


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
    input_mask = []
    for i, sample in enumerate(bin):
        input_ids.extend(sample)
        labels.extend(sample)
        sample_len = len(sample)
        if i == len(bin) - 1:
            attention_span.extend(list(range(sample_len - 2, -1, -1)))
            position_ids.extend(list(range(sample_len - 1)))
            input_mask.extend([1] * (sample_len - 1))
        else:
            attention_span.extend(list(range(sample_len - 1, -1, -1)))
            position_ids.extend(list(range(sample_len)))
            input_mask.extend(
                [1] * (sample_len - 1) + [0]
            )  ## The separator should have 0 as eos token

    input_ids = input_ids[:-1]
    labels = labels[1:]
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
            '.fasta',
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


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(
        obj, (str, bytes, bytearray)
    ):
        size += sum([get_size(i, seen) for i in obj])
    return size


def append_eos_to_multiple_semantic_regions(
    formatted_data,
    data_ranges,
    eos_token,
    image_token,
    is_chat_data,
):

    if data_ranges == [] or not eos_token:
        return data_ranges
    eos_indices = []
    start_search_index = data_ranges[0].get("indices")[0]
    while start_search_index < len(formatted_data):
        eos_start_idx = formatted_data.find(eos_token, start_search_index)
        if eos_start_idx == -1:
            ## No eos found. Break
            break
        eos_end_idx = eos_start_idx + len(eos_token)
        start_search_index = eos_end_idx
        eos_indices.append((eos_start_idx, eos_end_idx))

    current_eos_pos = 0
    current_data_range_pos = 0
    while current_eos_pos < len(eos_indices) and current_data_range_pos < len(
        data_ranges
    ):
        eos_start_idx, eos_end_idx = eos_indices[current_eos_pos]
        region_start_idx, region_end_idx = data_ranges[
            current_data_range_pos
        ].get("indices")
        ## EOS occurs in the current region
        if region_start_idx <= eos_start_idx < region_end_idx:
            current_eos_pos += 1
            continue

        if current_data_range_pos + 1 < len(data_ranges):
            next_region_start_idx, next_region_end_idx = data_ranges[
                current_data_range_pos + 1
            ].get("indices")
            ## Check if eos occurs between current and next region
            if region_end_idx <= eos_start_idx < next_region_start_idx:
                image_start_idx = (
                    -1
                    if image_token is None
                    else formatted_data[region_end_idx:eos_start_idx].find(
                        image_token
                    )
                )
                if image_start_idx == -1:
                    indices_incl_eos = (region_start_idx, eos_end_idx)
                    data_ranges[current_data_range_pos][
                        "indices"
                    ] = indices_incl_eos
                    current_eos_pos += 1
        else:
            ## insert EOS in the last region
            image_start_idx = (
                -1
                if image_token is None
                else formatted_data[region_end_idx:eos_start_idx].find(
                    image_token
                )
            )
            if image_start_idx == -1:
                indices_incl_eos = (region_start_idx, eos_end_idx)
                data_ranges[current_data_range_pos][
                    "indices"
                ] = indices_incl_eos
                current_eos_pos += 1
        current_data_range_pos += 1

    if (
        not is_chat_data or len(eos_indices) > 1
    ):  ## 1 because the last eot could be eos
        return data_ranges

    for i in range(1, len(data_ranges)):
        start_idx, end_idx = data_ranges[i].get("indices")
        previous_start_idx, previous_end_idx = data_ranges[i - 1].get("indices")
        if previous_end_idx != start_idx:
            handle_turn_token = True
            data_ranges[i - 1]["handle_turn_token"] = True
        if i == len(data_ranges) - 1:
            if end_idx < len(formatted_data):
                data_ranges[i]["handle_turn_token"] = True

    return data_ranges


def find_region_in_formatted_string(text_semantic_region_list, formatted_data):

    string_search_idx = 0
    for semantic_region in text_semantic_region_list:
        region_name = semantic_region.get("region_modality")
        region_identifier = semantic_region.pop("region_identifier", "")
        region_len = semantic_region.get("region_len")
        loss_weight = semantic_region.get("loss_weight")
        attention_mask = semantic_region.get("attention_mask", None)
        region_identifier_start_idx = formatted_data.find(
            region_identifier, string_search_idx
        )
        assert (
            region_identifier_start_idx != -1
        ), f"Unable to find region_identifier - {region_identifier} in the string - {formatted_data}"
        formatted_data = formatted_data.replace(region_identifier, "")
        start_idx = region_identifier_start_idx
        end_idx = start_idx + region_len
        string_search_idx = end_idx
        semantic_region.update({"indices": (start_idx, end_idx)})

    return formatted_data, text_semantic_region_list


def find_token_range(region, offsets, starting_offset_position):

    string_start, string_end = region.pop('indices')
    token_start = next(
        (
            i
            for i in range(starting_offset_position, len(offsets))
            if (offsets[i][0] <= string_start and offsets[i][1] > string_start)
            or (
                offsets[i][0] > string_start
            )  ## this condition is useful for neox tokenizer which treats space as an additional token
        ),
        None,
    )
    if token_start is None:
        raise ValueError(
            f"The implementation of offset mapping of this tokenizer may be incorrect. Check the huggingface implementation for more details."
        )
    token_end = next(
        (
            i
            for i in range(starting_offset_position, len(offsets))
            if offsets[i][1] >= string_end and offsets[i][0] < string_end
        ),
        None,
    )
    if token_end is None:
        raise ValueError(
            f"The huggingface implementation of offset mapping of this tokenizer may be incorrect. Check the huggingface implementation for more details."
        )
    data = {
        "indices": (token_start, token_end + 1),
        "loss_weight": region.get("loss_weight"),
        "attention_mask": region.get("attention_mask"),
    }

    return data
