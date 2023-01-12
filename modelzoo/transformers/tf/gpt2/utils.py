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

import math

import yaml

from modelzoo.common.model_utils.count_lines import count_lines
from modelzoo.common.tf.run_utils import is_cs
from modelzoo.transformers.tf.transformer_utils import get_bits_per_x_dataset

try:
    from cerebras.pb.stack.autogen_pb2 import AP_ENABLED
    from cerebras.pb.stack.full_pb2 import FullConfig
except ImportError:
    pass  # non-cbcore run


def get_params(params_file):
    # Load yaml into params
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)

    set_defaults(params)
    return params


def set_defaults(params):
    params = get_bits_per_x_dataset(params)

    # Set up defaults/preprocessing
    params["model"]["position_embedding_type"] = (
        params["model"].get("position_embedding_type", "learned").lower()
    )
    params["model"]["embedding_size"] = params["model"].get(
        "embedding_size", params["model"]["hidden_size"]
    )
    params["model"]["initializer"] = params["model"].get(
        "initializer", {"name": "truncated_normal", "mean": 0.0, "stddev": 0.02}
    )
    params["model"]["embedding_initializer"] = params["model"].get(
        "embedding_initializer", None
    )
    num_hidden_layers = params["model"]["num_hidden_layers"]
    params["model"]["output_layer_initializer"] = params["model"].get(
        "output_layer_initializer",
        {
            "name": "truncated_normal",
            "mean": 0.0,
            # By default, use the GPT-2 scaled initialization:
            # Divide standard deviation by the square-root of the number
            # of residual connections in the model (2 * # layers)
            "stddev": 0.02 / math.sqrt(2 * num_hidden_layers),
        },
    )
    params["model"]["weight_initialization_seed"] = params["model"].get(
        "weight_initialization_seed", None
    )
    params["model"]["use_ffn_bias_in_attention"] = params["model"].get(
        "use_ffn_bias_in_attention", True
    )
    params["model"]["use_ffn_bias"] = params["model"].get("use_ffn_bias", True)

    # Train vocabulary size
    vocab_size = params["train_input"].get("vocab_size")
    if not vocab_size:
        vocab_size = count_lines(params["train_input"]["vocab_file"])
    params["train_input"]["vocab_size"] = vocab_size

    # train_input required parameters
    params["train_input"]["shuffle"] = params["train_input"].get(
        "shuffle", True
    )
    params["train_input"]["repeat"] = params["train_input"].get("repeat", True)

    params["eval_input"] = params.get("eval_input", {})
    # Eval vocabulary size
    if not params["eval_input"]:
        params["eval_input"]["vocab_size"] = params["train_input"]["vocab_size"]
    else:
        vocab_size = params["eval_input"].get("vocab_size")
        if not vocab_size:
            vocab_size = count_lines(params["eval_input"]["vocab_file"])
        params["eval_input"]["vocab_size"] = vocab_size

    # eval_input required parameters
    params["eval_input"]["shuffle"] = params["eval_input"].get("shuffle", False)
    params["eval_input"]["repeat"] = params["eval_input"].get("repeat", False)
    params["eval_input"]["add_special_tokens"] = params["eval_input"].get(
        "add_special_tokens", False
    )

    params["model"]["max_position_embeddings"] = params["model"].get(
        "max_position_embeddings", None
    )

    params["model"]["use_pre_normalization"] = params["model"].get(
        "use_pre_normalization", True
    )

    params["model"]["fixed_sparse_attention"] = params["model"].get(
        "fixed_sparse_attention", None
    )
    params["model"]["dropout_seed"] = params["model"].get("dropout_seed", None)
    params["model"]["dropout_rate"] = params["model"].get("dropout_rate", 0.0)
    params["model"]["attention_dropout_rate"] = params["model"].get(
        "attention_dropout_rate", 0.0
    )
    params["model"]["share_embedding_weights"] = params["model"].get(
        "share_embedding_weights", True
    )
    params["model"]["layer_norm_epsilon"] = float(
        params["model"].get("layer_norm_epsilon", 1e-8)
    )
    params["model"]["attention_softmax_fp32"] = params["model"].get(
        "attention_softmax_fp32", True
    )

    params["model"]["loss_scaling"] = (
        params["model"].get("loss_scaling", "num_tokens").lower()
    )
    params["model"]["loss_weight"] = params["model"].get("loss_weight", 1.0)

    # optimizer parameters
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )
    params["optimizer"]["max_gradient_norm"] = params["optimizer"].get(
        "max_gradient_norm", None
    )
    params["optimizer"]["grad_accum_steps"] = params["optimizer"].get(
        "grad_accum_steps", 1
    )
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )

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
    params["runconfig"]["max_steps"] = params["runconfig"].get(
        "max_steps", None
    )
    params["runconfig"]["throttle_secs"] = params["runconfig"].get(
        "throttle_secs", 0
    )
    params["runconfig"]["eval_steps"] = params["runconfig"].get("eval_steps", 0)

    params["eval_input"]["batch_size"] = params["eval_input"].get(
        "batch_size", params["train_input"]["batch_size"]
    )

    # Pass along the TFBaseModel's mixed_precision flag to the data processors
    params["train_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )
    params["eval_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )

    params["runconfig"]["log_step_count_steps"] = params["runconfig"].get(
        "log_step_count_steps", params["runconfig"]["save_summary_steps"]
    )

    # Diasable standard estimator logs and use a custom
    # logging hook when trained with gradient accumulation.
    params["runconfig"]["disable_standard_logs"] = params["runconfig"].get(
        "disable_standard_logs", params["optimizer"]["grad_accum_steps"] > 1
    )

    # set default weight_servers
    params["runconfig"]["num_wgt_servers"] = params["runconfig"].get(
        "num_wgt_servers", 12
    )


def get_custom_stack_params(params):
    stack_params = dict()
    runconfig_params = params["runconfig"]
    use_cs = is_cs(runconfig_params)
    stack_params["ir_mode"] = "mlir-cirh"
    if (
        use_cs
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        stack_params["config"] = set_custom_config(FullConfig(), params)
        return stack_params


def set_custom_config(config, params):
    config.matching.kernel.inc_pwt_estimate = True
    config.matching.kernel.enable_pipelined_mlm_loss = True
    use_vsl = params["model"].get("use_vsl", False)
    config.matching.kernel.use_legacy_vsl = use_vsl

    runconfig_params = params["runconfig"]
    if runconfig_params.get("mode", "train") == "train":
        config.matching.autogen_policy = AP_ENABLED
    else:
        config.matching.add_pack_and_unpack.max_egress_per_pack = 1
        config.placement.prep_recolor_kernels.wrap_pack_kernel = True
        config.matching.match_lair.disabled_converters.append(
            "UnsortedGatherConverter"
        )
    return config
