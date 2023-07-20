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


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """

    if params["train_input"]["data_processor"] == "Gpt2SyntheticDataProcessor":
        if "train_input" in params:
            params["train_input"]["vocab_size"] = params["train_input"].get(
                "vocab_size", params["model"]["vocab_size"]
            )
            assert (
                params["train_input"]["vocab_size"]
                == params["model"]["vocab_size"]
            ), f"Found different vocab_size in train_input ({params['train_input']['vocab_size']}) vs. model ({params['model']['vocab_size']})"
            params["train_input"]["max_sequence_length"] = params[
                "train_input"
            ].get(
                "max_sequence_length",
                params["model"]["max_position_embeddings"],
            )

        if "eval_input" in params:
            params["eval_input"]["vocab_size"] = params["eval_input"].get(
                "vocab_size", params["model"]["vocab_size"]
            )
            assert (
                params["eval_input"]["vocab_size"]
                == params["model"]["vocab_size"]
            ), f"Found different vocab_size in eval_input ({params['eval_input']['vocab_size']}) vs. model ({params['model']['vocab_size']})"
            params["eval_input"]["max_sequence_length"] = params[
                "eval_input"
            ].get(
                "max_sequence_length",
                params["model"]["max_position_embeddings"],
            )

    params["model"]["use_bfloat16"] = params["model"].get("use_bfloat16", True)
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )

    params["runconfig"]["precision_opt_level"] = params["runconfig"].get(
        "precision_opt_level", 1
    )

    from modelzoo.transformers.pytorch.gpt2.utils import set_attention_kernel

    set_attention_kernel(params)

    loss_scaling = params["model"].get("loss_scaling", "num_tokens")
    use_cs_grad_accum = params["runconfig"].get("use_cs_grad_accum", False)
    num_csx = params["runconfig"].get("num_csx")  # default from command line
    target_device = params["runconfig"].get("target_device")
    if (
        loss_scaling == "num_tokens"
        and use_cs_grad_accum
        and num_csx > 1
        and target_device == "CSX"
    ):
        logging.warning(
            "Using loss scaling by num_tokens with use_cs_grad_accum is not "
            "recommended and may lead to instabilities in the training in "
            "distributed runs.\n"
            "Please use loss_scaling='batch_size' instead and provide value "
            "for 'loss_weight'.\n"
            "Loss weight is 1/(average number of non-masked, non-padded tokens per sequence.)"
            " e.g. '1/max_sequence_length' when all sequences are fully packed."
        )
