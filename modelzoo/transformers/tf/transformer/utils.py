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

from copy import deepcopy

import yaml

from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.tf.model_utils.vocab_utils import get_vocab_size
from modelzoo.common.tf.run_utils import is_cs

if CSOFT_PACKAGE == CSoftPackage.SRC:
    import cerebras.pb.common.cerebras_type_pb2 as cb_types
    from cerebras.pb.common.tri_state_pb2 import TS_DISABLED, TS_ENABLED
    from cerebras.pb.stack.full_pb2 import FullConfig
elif CSOFT_PACKAGE == CSoftPackage.WHEEL:
    import cerebras_appliance.pb.common.cerebras_type_pb2 as cb_types
    from cerebras_appliance.pb.common.tri_state_pb2 import (
        TS_DISABLED,
        TS_ENABLED,
    )
    from cerebras_appliance.pb.stack.full_pb2 import FullConfig
elif CSOFT_PACKAGE == CSoftPackage.NONE:
    pass
else:
    assert False, f"Invalid value for `CSOFT_PACKAGE`: {CSOFT_PACKAGE}"


def get_params(params_file):
    # Load yaml into params
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)

    set_defaults(params)
    return params


def set_defaults(params, mode=None):

    # Embeddings.

    params["model"]["position_embedding_type"] = (
        params["model"].get("position_embedding_type", "learned").lower()
    )
    validate_input(
        "position_embedding_type",
        params["model"]["position_embedding_type"],
        ["fixed", "learned"],
    )

    params["model"]["max_position_embeddings"] = params["model"].get(
        "max_position_embeddings", None
    )

    # Initialization.
    params["model"]["embedding_initializer"] = params.get(
        "embedding_initializer",
        {
            "name": "truncated_normal",
            "mean": 0.0,
            "stddev": params["model"]["hidden_size"] ** -0.5,
        },
    )
    params["model"]["attention_initializer"] = params["model"].get(
        "attention_initializer",
        {
            "name": "variance_scaling",
            "scale": 1.0 / 9.0,
            "mode": "fan_avg",
            "distribution": "uniform",
        },
    )
    params["model"]["feed_forward_initializer"] = params["model"].get(
        "feed_forward_initializer", {"name": "glorot_uniform"}
    )
    params["model"]["weight_initialization_seed"] = params["model"].get(
        "weight_initialization_seed", None
    )

    # Encoder.
    params["model"]["use_pre_normalization"] = params["model"].get(
        "use_pre_normalization", False
    )
    params["model"]["dropout_seed"] = params["model"].get("dropout_seed", None)
    params["model"]["attention_dropout_rate"] = params["model"].get(
        "attention_dropout_rate", 0.1
    )
    params["model"]["layer_norm_epsilon"] = float(
        params["model"].get("layer_norm_epsilon", 1e-8)
    )
    params["model"]["encoder_nonlinearity"] = (
        params["model"].get("encoder_nonlinearity", "relu").lower()
    )
    params["model"]["decoder_nonlinearity"] = (
        params["model"].get("decoder_nonlinearity", "relu").lower()
    )

    params["model"]["use_ffn_bias"] = params["model"].get("use_ffn_bias", True)

    # Optimizer.
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

    # Train input function params.
    # Pass along the model's mixed_precision flag to the data processors.
    params["train_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )

    params["train_input"]["src_vocab_size"] = get_vocab_size(
        params["train_input"].get("src_vocab_file"),
        params["train_input"].get("src_vocab_size"),
    )

    params["train_input"]["tgt_vocab_size"] = get_vocab_size(
        params["train_input"].get("tgt_vocab_file"),
        params["train_input"].get("tgt_vocab_size"),
    )

    # Set the default for share_encoder_decoder_embedding based on
    # whether the source and target vocab sizes are the same or not.
    params["model"]["share_encoder_decoder_embedding"] = params["model"].get(
        "share_encoder_decoder_embedding",
        params["train_input"]["src_vocab_size"]
        == params["train_input"]["tgt_vocab_size"],
    )

    # `eval_input` required parameters.
    if "eval_input" in params:
        params["eval_input"]["src_vocab_size"] = get_vocab_size(
            params["eval_input"].get("src_vocab_file"),
            params["eval_input"].get("src_vocab_size"),
        )

        params["eval_input"]["tgt_vocab_size"] = get_vocab_size(
            params["eval_input"].get("tgt_vocab_file"),
            params["eval_input"].get("tgt_vocab_size"),
        )
    else:
        # In case no `eval_input` is specified, set the params from `train_input`.
        params["eval_input"] = deepcopy(params["train_input"])

    params["eval_input"]["shuffle"] = params["eval_input"].get("shuffle", False)
    params["eval_input"]["repeat"] = params["eval_input"].get("repeat", False)
    params["eval_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )

    # Set defaults for VSL padding parameters.
    if params["model"].get("use_vsl") and mode != "train":
        params["model"]["use_vsl"] = False
    use_vsl = params["model"].get("use_vsl", False)
    if use_vsl:
        params["train_input"]["input_pad_id"] = params["train_input"].get(
            "input_pad_id", 0
        )
        params["eval_input"]["input_pad_id"] = params["eval_input"].get(
            "input_pad_id", 0
        )
        params["model"]["input_pad_id"] = params["train_input"]["input_pad_id"]
    else:
        # Hard-code the padding values to turn off VSL.
        params["train_input"]["input_pad_id"] = -1
        params["eval_input"]["input_pad_id"] = -1
        params["model"]["input_pad_id"] = -1

    params["model"]["use_vsl"] = use_vsl
    params["train_input"]["use_vsl"] = use_vsl
    params["eval_input"]["use_vsl"] = use_vsl

    if is_cs(params["runconfig"]):
        # always shard if running on CS
        params["train_input"]["use_multiple_workers"] = params[
            "train_input"
        ].get("use_multiple_workers", True)
        params["eval_input"]["use_multiple_workers"] = params["eval_input"].get(
            "use_multiple_workers", True
        )

    # Runconfig.
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
    if (
        params["runconfig"]["enable_distributed"]
        and params["optimizer"]["loss_scaling_factor"] == "dynamic"
    ):
        raise ValueError(
            "Cannot use 'dynamic' loss scaling for distributed training,"
            + " use 'tf_dynamic' instead."
        )
    params["runconfig"]["multiple_workers"] = params["runconfig"].get(
        "multiple_workers", False
    )

    # Handling eval / infer parameters.
    params["runconfig"]["throttle_secs"] = params["runconfig"].get(
        "throttle_secs", 0
    )
    params["runconfig"]["predict_steps"] = params["runconfig"].get(
        "predict_steps", 1
    )
    params["runconfig"]["log_step_count_steps"] = params["runconfig"].get(
        "log_step_count_steps", params["runconfig"]["save_summary_steps"]
    )
    # Disable standard estimator logs and use a custom
    # logging hook when trained with gradient accumulation.
    params["runconfig"]["disable_standard_logs"] = (
        params["optimizer"]["grad_accum_steps"] > 1
    )

    # CSConfig.
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

    return params


def validate_input(param_name, param_value, valid_inputs):
    if param_value not in valid_inputs:
        raise ValueError(
            f"Invalid input {param_value} for {param_name}. Allowed values are {valid_inputs}"
        )


def get_custom_stack_params(params):
    stack_params = dict()
    stack_params["ir_mode"] = "mlir-cirh"
    runconfig_params = params["runconfig"]
    if (
        is_cs(runconfig_params)
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        stack_params["config"] = set_custom_config(FullConfig(), params)
    return stack_params


def set_custom_config(config, params):

    config.matching.kernel.enable_pipelined_mlm_loss = True
    config.matching.kernel.numerics.layernorm_compute_dtype = cb_types.T_F16
    # Performance hack: For 2 of 3 perf critical variants - transformer-large, g42-base -
    # we become fabric bound if we disable enc/dec embedding sharing, in which case
    # setting inc_pwt_estimate=True leads to worse runtime performance
    # Note that fabric utilization is a function of many params, mainly hidden size, MSL,
    # num encoders/decoders, share enc/dec embedding. So this is a crude short term hack
    if params["model"]["share_encoder_decoder_embedding"]:
        config.matching.kernel.inc_pwt_estimate = True

    use_vsl = params["model"].get("use_vsl", False)
    config.matching.kernel.use_legacy_vsl = use_vsl

    if params["runconfig"].get("mode", "train") == "eval":
        config.matching.add_pack_and_unpack.max_egress_per_pack = 1
        config.placement.prep_recolor_kernels.wrap_pack_kernel = True

    if params["runconfig"].get("multireplica", False):
        config.target_num_replicas = -1
        config.placement.pathfinder_inter_replica.fix_existing = TS_DISABLED
        config.placement.pathfinder_inter_replica.allow_infeasible_initial_state = (
            TS_ENABLED
        )
        # Always turns off pwt knob for multireplica for better perf since we get close
        # to max fabric utilization with multireplica enabled
        config.matching.kernel.inc_pwt_estimate = False

    return config
