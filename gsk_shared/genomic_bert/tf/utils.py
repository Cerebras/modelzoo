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

import yaml

from modelzoo.common.model_utils.count_lines import count_lines


def get_params(params_file):
    # Load yaml into params
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
    adjust_params(params)
    return params


def adjust_params(params):

    params["model_type"] = "GenomicBertModel"

    # Set up defaults/preprocessing
    params["model"]["position_embedding_type"] = (
        params["model"].get("position_embedding_type", "learned").lower()
    )

    # Train vocabulary size
    vocab_size_dna = params["train_input"].get("vocab_size_dna")
    if not vocab_size_dna:
        vocab_size_dna = count_lines(params["train_input"]["vocab_file_dna"])
    params["train_input"]["vocab_size_dna"] = vocab_size_dna

    vocab_size_ideas = params["train_input"].get("vocab_size_ideas")
    if not vocab_size_ideas:
        vocab_size_ideas = count_lines(
            params["train_input"]["vocab_file_ideas"]
        )
    params["train_input"]["vocab_size_ideas"] = vocab_size_ideas

    params["eval_input"] = params.get("eval_input", {})
    # Eval vocabulary size
    if not params["eval_input"]:
        params["eval_input"]["vocab_size_dna"] = params["train_input"][
            "vocab_size_dna"
        ]
        params["eval_input"]["vocab_size_ideas"] = params["train_input"][
            "vocab_size_ideas"
        ]
    else:
        vocab_size_dna = params["eval_input"].get("vocab_size_dna")
        if not vocab_size_dna:
            vocab_size_dna = count_lines(params["eval_input"]["vocab_file_dna"])
        params["eval_input"]["vocab_size_dna"] = vocab_size_dna

        vocab_size_ideas = params["eval_input"].get("vocab_size_ideas")
        if not vocab_size_ideas:
            vocab_size_ideas = count_lines(
                params["eval_input"]["vocab_file_ideas"]
            )
        params["eval_input"]["vocab_size_ideas"] = vocab_size_ideas

    loss_scaling_type = params["model"].get("mlm_loss_scaling", "batch_size")
    assert loss_scaling_type in ["batch_size", "precomputed_num_masked"], (
        f"loss scaling type must be either 'batch_size' or "
        f" 'precomputed_num_masked'. Got {loss_scaling_type}"
    )
    if loss_scaling_type == "precomputed_num_masked":
        params["train_input"]["scale_mlm_weights"] = True
        params["eval_input"]["scale_mlm_weights"] = True
    else:
        params["train_input"]["scale_mlm_weights"] = False
        params["eval_input"]["scale_mlm_weights"] = False

    # eval_input required parameters
    params["eval_input"]["shuffle"] = params["eval_input"].get("shuffle", False)
    params["eval_input"]["repeat"] = params["eval_input"].get("repeat", False)
    params["eval_input"]["add_special_tokens"] = params["eval_input"].get(
        "add_special_tokens", False
    )

    params["model"]["mlm_nonlinearity"] = params["model"].get(
        "mlm_nonlinearity", None
    )
    mixed_precision = params["model"].get("mixed_precision", False)

    params["model"]["layer_norm_epsilon"] = float(
        params["model"].get(
            "layer_norm_epsilon", 1e-6 if mixed_precision else 1e-8
        )
    )
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )
    params["optimizer"]["max_gradient_norm"] = params["optimizer"].get(
        "max_gradient_norm", None
    )
    params["optimizer"]["grad_accum_steps"] = params["optimizer"].get(
        "grad_accum_steps", 1
    )
    params["model"]["disable_layer_norm"] = params["model"].get(
        "disable_layer_norm", False
    )
    params["model"]["enable_gpu_optimizations"] = params["model"].get(
        "enable_gpu_optimizations", False
    )

    # vsl defaults
    use_vsl = params["model"].get("use_vsl", False)
    if use_vsl:
        for key, default in zip(["input_pad_id", "mlm_pad_id"], [0, -1]):
            # Padding values are required to match between the specification
            # for `train_input` and for `eval_input`.
            train_input_val = params["train_input"].get(key, None)
            eval_input_val = params["eval_input"].get(key, None)
            if train_input_val is None and eval_input_val is None:
                params["train_input"][key] = default
                params["eval_input"][key] = default
            elif train_input_val is None and eval_input_val is not None:
                params["train_input"][key] = eval_input_val
            elif train_input_val is not None and eval_input_val is None:
                params["eval_input"][key] = train_input_val
            elif train_input_val != eval_input_val:
                raise ValueError(
                    f"If specified, {key} must match in train and eval input "
                    f"parameters. Got {train_input_val} and {eval_input_val}."
                )
        input_pad_id = params["train_input"]["input_pad_id"]
        if (
            0 < input_pad_id < max(vocab_size_ideas, vocab_size_dna)
            or input_pad_id == -1
        ):
            raise ValueError(
                "input_pad_id must be the pad token, out of vocab, or a "
                f"negative index less than -1. Got {input_pad_id}."
            )
        if not params["train_input"]["mlm_pad_id"] == -1:
            raise ValueError(
                f"mlm_pad_id must be -1. Got {params['train_input']['mlm_pad_id']}"
            )
        params["model"]["input_pad_id"] = params["train_input"]["input_pad_id"]
    else:
        # hard-code padding values to turn off VSL
        params["model"]["input_pad_id"] = -1

    # handling eval
    params["runconfig"]["throttle_secs"] = params["runconfig"].get(
        "throttle_secs", 0
    )

    params["train_input"]["mixed_precision"] = mixed_precision
    params["eval_input"]["mixed_precision"] = mixed_precision

    # Runconfig parameters
    params["runconfig"]["save_summary_steps"] = params["runconfig"].get(
        "save_summary_steps", 100
    )
    params["runconfig"]["save_checkpoints_steps"] = params["runconfig"].get(
        "save_checkpoints_steps", 1000
    )
    params["runconfig"]["keep_checkpoint_max"] = params["runconfig"].get(
        "keep_checkpoint_max", 5
    )
    params["runconfig"]["tf_random_seed"] = params["runconfig"].get(
        "tf_random_seed", None
    )
    params["runconfig"]["enable_distributed"] = params["runconfig"].get(
        "enable_distributed", False
    )
    params["runconfig"]["multiple_workers"] = params["runconfig"].get(
        "multiple_workers", False
    )

    params["csconfig"] = params.get("csconfig", {})
    params["csconfig"]["use_cbfloat16"] = params["csconfig"].get(
        "use_cbfloat16", False
    )
    if (
        params["csconfig"]["use_cbfloat16"]
        and not params["model"]["mixed_precision"]
    ):
        raise ValueError(
            "use_cbfloat16=True can only be used in mixed precision"
            " mode. Set mixed_precision to True."
        )
