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

"""Defines the `CSEvalHarnessAdapter` class for executing eval harnesses on CSX."""

import atexit
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.instance import Instance
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import cerebras.pytorch as cstorch
from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.modelzoo.common.utils.input.utils import SamplesSaver
from cerebras.modelzoo.data.nlp.gpt.InferenceDataProcessor import RequestType


@named_class_logger("CSEvalHarnessAdapter")
class CSEvalHarnessAdapter(ClassLogger):
    """Defines cstorch components (i.e. data preprocessing) required for executing eval harness
    on appliance.
    """

    def __init__(self, trainer, dataloader_args: Dict[str, Any]):
        """
        Args:
            trainer: The Trainer object to use to run validation.
            dataloader_args: A dictionary consisting of arguments to pass to
                the dataloader.
        """
        super().__init__()

        self.trainer = trainer
        self.dataloader_args = dataloader_args

        data_dir = dataloader_args.pop("data_dir", None)

        if data_dir is None:
            raise RuntimeError(
                "No data directory specified in params. "
                "Please specify `data_dir`, a valid path to a mounted "
                "directory where data samples will be saved, "
                f"to {type(self).__name__}'s constructor."
            )

        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            if cstorch.distributed.is_streamer():
                raise RuntimeError(
                    f"Invalid path to mounted directory specified: {data_dir} "
                    f"Please ensure that the path dir is valid dir visible "
                    f"to the appliance nodes."
                )
            else:
                data_dir.mkdir(parents=True, exist_ok=False)

        self.eh_tasks_dir = data_dir / (
            "eval_harness_tasks_data_" + uuid.uuid4().hex
        )

        self.keep_data_dir = dataloader_args.pop("keep_data_dir")

        self.batch_size = dataloader_args.get("batch_size")
        if (
            self.batch_size is None
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise RuntimeError(
                "Please specify a positive integer for the batch size "
                f"for {type(self).__name__} to preprocess input data samples "
                "from the specified eval harness tasks."
            )

        self.msl = dataloader_args.pop("max_sequence_length", None)
        if self.msl is None:
            info_msg = (
                f"No maximum sequence length provided to {type(self).__name__}. "
                "This setting is required for preprocessing input data samples from the specified "
                "eval harness tasks.\n{0} Note that input sequences will be truncated to fit "
                "within this length."
            )

            max_position_embeddings = getattr(
                trainer.model, "max_position_embeddings", None
            )
            if max_position_embeddings is None:
                model = getattr(trainer.model, "model", None)
                if model is not None:
                    max_position_embeddings = getattr(
                        model, "max_position_embeddings", None
                    )
            if max_position_embeddings is None:
                raise RuntimeError(
                    info_msg.format(
                        "Please specify the maximum sequence (or context) length."
                    )
                )

            self.msl = max_position_embeddings
            self.logger.info(
                info_msg.format(
                    f"Defaulting to max sequence length of {max_position_embeddings}, as "
                    "specified by the model's max_position_embeddings attribute."
                )
            )

        from cerebras.modelzoo.data.nlp.gpt.InferenceDataProcessor import (
            InferenceDataProcessor,
        )

        def input_fn(
            dataloader_args,
            samples_file_list,
            dataset_size,
            request_type,
            **dataset_kwargs,
        ):
            return InferenceDataProcessor.from_request_type(
                request_type,
                dataloader_args,
                samples_file_list,
                dataset_size,
                **dataset_kwargs,
            ).create_dataloader()

        self.input_fn = input_fn
        self.data_fn = InferenceDataProcessor.gen_data_samples

        # Initialize tokenizer
        self._create_tokenizer()

    def _subclass_name(self) -> str:
        """Helper to extract the name of the subclass."""
        return self.__class__.__name__

    def _create_tokenizer(self) -> None:
        """Helper method to create the tokenizer upon initialization using the dataloader args."""
        tokenizer_file_path = self.dataloader_args.get(
            "tokenizer_file_path", None
        )
        eos_token_id = self.dataloader_args.pop("eos_id", None)

        if tokenizer_file_path is not None:
            error_log = (
                "No {config_desc} specified under {config} for "
                f"custom pretrained tokenizer: {tokenizer_file_path}\n "
                "Please specify the {config_desc} under "
                f"{type(self).__name__}."
            )
            if eos_token_id is None:
                raise RuntimeError(
                    error_log.format(
                        config_desc="end of sentence token id",
                        config="`eos_id`",
                    )
                )
            tok = Tokenizer.from_file(tokenizer_file_path)

            bos_token = self.dataloader_args.pop("bos_token", None)
            if (
                self._subclass_name() == "BigCodeEvaluator"
                and bos_token is None
            ):
                raise RuntimeError(
                    error_log.format(
                        config_desc="beginning of sentence token",
                        config="`bos_token`",
                    )
                )
            # Wrap the tokenizers.Tokenizer object into a HF transformers tokenizer to be able
            # to access shared properties and methods
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_file_path,
                eos_token=tok.decode(
                    [eos_token_id],
                    skip_special_tokens=False,
                ),  # TODO: Change to accept eos token instead of its id?
                bos_token=bos_token,
            )
        elif (
            pretrained_model_name_or_path := self.dataloader_args.get(
                "pretrained_model_name_or_path"
            )
        ) is not None:

            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path
            )
            eos_token_id = self.tokenizer.eos_token_id
            self.logger.info(
                f"No custom tokenizer file path specified under `tokenizer_file_path`. "
                f"Using {pretrained_model_name_or_path} tokenizer with end of sentence token id "
                f"{eos_token_id}, as specified under config `pretrained_model_name_or_path`."
            )
        else:
            raise RuntimeError(
                f"Tokenizer config is missing. Please either specify a custom "
                f"tokenizer using config `tokenizer_file_path` or a huggingface pretrained "
                f"tokenizer using `pretrained_model_name_or_path` under {type(self).__name__}."
            )

        vocab_size = self.trainer.model.vocab_size
        if eos_token_id >= vocab_size:
            raise ValueError(
                "The tokenizer being used for this run does not match with "
                "the chosen model architecture: the EOS token id for the "
                f"tokenizer is {eos_token_id}, whereas the model's max vocab "
                f"size is only {vocab_size}. Please ensure that the tokenizer "
                "corresponds to the specified model architecture."
            )

        # (Re-)add (updated) "eos_id" to dataloader args
        self.dataloader_args["eos_id"] = eos_token_id

    def preprocess_dataset(
        self,
        requests: List[Union[Instance, str]],
        request_type: RequestType,
        max_tokens: Optional[int] = None,
    ) -> Tuple[
        List[str],
        int,
        List[Tuple[int, int]],
    ]:
        """Tokenize raw text requests and returns metadata associated with the
        request type.

        Args:
            requests: List of EEH's Instance dataclass objects
                holding raw text data
            request_type: The type of request
            max_tokens: (Generative EH preprocessing) The max number of tokens to generate

        Returns:
            tuple of
            - list of file paths where the samples are dumped;
            - int representing the size of the dataset (total no. of samples);
            - List of metadata tuples needed for postprocessing.
        """

        # Create data samples and dump these to file
        MAX_FILE_SIZE = 1024 * 1024 * 500  # 500 MB

        if (
            request_type == RequestType.eeh_generate_until
            or request_type == RequestType.bigcode_eh
        ):
            stop_words_cache = {}
            # NOTE: Default to 4 total number of stop tokens with
            # each stop sequence being of max length 8 for EEH. Note that
            # these values may be updated to correctly represent the
            # task(s) data when we pass over all samples in the
            # preprocess_dataset method.
            if request_type == RequestType.eeh_generate_until:
                stop_sequence_shape = (4, 8)
            else:
                # For BCEH, we increase the shape to cater to the fact
                # that each generative task runs sequentially and so
                # we don't fetch all tasks' stop sequences up front.
                stop_sequence_shape = (10, 10)
        else:
            stop_words_cache = None
            stop_sequence_shape = None

        samples_saver = SamplesSaver(
            data_dir=str(self.eh_tasks_dir),
            max_file_size=MAX_FILE_SIZE,
            filename_prefix=f"requests_{self.msl}_msl",
            dtype=object,
        )

        samples_file_list, dataset_size, metadata = self.data_fn(
            requests,
            self.batch_size,
            self.msl,
            self.tokenizer,
            samples_saver,
            request_type=request_type,
            max_input_len=0,
            inf_start_token=getattr(self.trainer.model, "start_token", None),
            max_gen_tokens=(
                max_tokens
                if max_tokens is not None
                else getattr(self.trainer.model, "max_tokens", None)
            ),
            stop_words_cache=stop_words_cache,
            stop_sequence_shape=stop_sequence_shape,
        )

        total_steps = dataset_size // self.batch_size
        self.logger.info(f"Total eval steps: {total_steps}")

        # Register clean-up method to remove data dumps
        if not self.keep_data_dir:
            atexit.register(samples_saver.delete_data_dumps)

        return samples_file_list, dataset_size, metadata
