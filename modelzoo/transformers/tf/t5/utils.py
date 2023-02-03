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
    # Model.

    assert not params["model"]["num_heads"] % 2, (
        f"Until the release 1.0, please specify an even number of heads for T5 model. "
        f"Got number of heads equal to {params['model']['num_heads']}. "
    )

    params["model"]["use_relative_attention_bias"] = params["model"].get(
        "use_relative_attention_bias", True
    )
    params["model"]["num_relative_attention_buckets"] = params["model"].get(
        "num_relative_attention_buckets", 32
    )

    factor = params["model"].get("initializer_factor", 1.0)
    # Mesh TensorFlow embedding initialization
    # See: https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1635
    params["model"]["embedding_initializer"] = params["model"].get(
        "embedding_initializer",
        {"name": "normal", "mean": 0.0, "stddev": factor * 1.0,},
    )

    # Mesh TensorFlow attention initialization to avoid scaling before softmax
    # See: https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
    params["model"]["query_layer_initializer"] = params["model"].get(
        "query_layer_initializer",
        {
            "name": "normal",
            "mean": 0.0,
            "stddev": factor
            * ((params["model"]["d_model"] * params["model"]["d_kv"]) ** -0.5),
        },
    )
    params["model"]["key_layer_initializer"] = params["model"].get(
        "key_layer_initializer",
        {
            "name": "normal",
            "mean": 0.0,
            "stddev": factor * (params["model"]["d_model"] ** -0.5),
        },
    )
    params["model"]["value_layer_initializer"] = params["model"].get(
        "value_layer_initializer",
        {
            "name": "normal",
            "mean": 0.0,
            "stddev": factor * (params["model"]["d_model"] ** -0.5),
        },
    )

    params["model"]["output_layer_initializer"] = params["model"].get(
        "output_layer_initializer",
        {
            "name": "normal",
            "mean": 0.0,
            "stddev": factor
            * (
                (params["model"]["num_heads"] * params["model"]["d_kv"]) ** -0.5
            ),
        },
    )
    # Mesh TensorFlow RPE initalization
    # See: https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L287-L289
    # and https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
    params["model"]["relative_attention_bias_weight_initializer"] = params[
        "model"
    ].get(
        "relative_attention_bias_weight_initializer",
        {"name": "glorot_uniform",},
    )

    params["model"]["weight_initialization_seed"] = params["model"].get(
        "weight_initialization_seed", None
    )

    # Mesh TensorFlow FF initialization
    # See: https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
    # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
    params["model"]["feed_forward_input_layer_initializer"] = params[
        "model"
    ].get(
        "feed_forward_input_layer_initializer",
        {
            "name": "normal",
            "mean": 0.0,
            "stddev": factor * ((params["model"]["d_model"]) ** -0.5),
        },
    )
    params["model"]["feed_forward_output_layer_initializer"] = params[
        "model"
    ].get(
        "feed_forward_output_layer_initializer",
        {
            "name": "normal",
            "mean": 0.0,
            "stddev": factor * ((params["model"]["d_ff"]) ** -0.5),
        },
    )
    # Dropout rate is used in embedding, attention, feed forward layers.
    params["model"]["dropout_rate"] = params["model"].get("dropout_rate", 0.1)
    params["model"]["attention_type"] = params["model"].get(
        "attention_type", "dot_product"
    )
    params["model"]["mlm_loss_scaling"] = params["model"].get(
        "mlm_loss_scaling", "precomputed_num_masked"
    )
    params["model"]["lm_loss_weight"] = params["model"].get(
        "lm_loss_weight", 0.015
    )
    params["model"]["use_ffn_bias"] = params["model"].get("use_ffn_bias", True)
    params["model"]["mixed_precision"] = params["model"].get(
        "mixed_precision", True
    )
    params["model"]["layer_norm_epsilon"] = params["model"].get(
        "layer_norm_epsilon", 1e-5
    )

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

    # Inputs
    params["train_input"]["do_lower_case"] = params["train_input"].get(
        "do_lower_case", False
    )
    params["train_input"]["extra_ids"] = params["train_input"].get(
        "extra_ids", 100
    )

    # Train vocabulary size.
    params["train_input"]["vocab_size"] = (
        get_vocab_size(
            params["train_input"].get("vocab_file"),
            params["train_input"].get("vocab_size"),
        )
        + params["train_input"]["extra_ids"]
    )

    params["model"]["share_encoder_decoder_embedding"] = params["model"].get(
        "share_encoder_decoder_embedding", True
    )

    # Pass along the model's mixed_precision flag to the data processors
    params["train_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )

    # `eval_input` required parameters.
    if "eval_input" in params:
        params["eval_input"]["extra_ids"] = params["eval_input"].get(
            "extra_ids", 100
        )

        params["eval_input"]["do_lower_case"] = params["eval_input"].get(
            "do_lower_case", False
        )
        params["eval_input"]["mixed_precision"] = params["model"].get(
            "mixed_precision", False
        )
        params["eval_input"]["src_max_sequence_length"] = params[
            "eval_input"
        ].get(
            "src_max_sequence_length",
            params["train_input"]["src_max_sequence_length"],
        )
        params["eval_input"]["tgt_max_sequence_length"] = params[
            "eval_input"
        ].get(
            "tgt_max_sequence_length",
            params["train_input"]["tgt_max_sequence_length"],
        )

        params["eval_input"]["vocab_size"] = (
            get_vocab_size(
                params["eval_input"].get("vocab_file"),
                params["eval_input"].get("vocab_size"),
            )
            + params["eval_input"]["extra_ids"]
        )
    else:
        # In case no `eval_input` is specified, set the params from `train_input`.
        params["eval_input"] = deepcopy(params["train_input"])

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

    # Handling eval / infer parameters
    params["runconfig"]["throttle_secs"] = params["runconfig"].get(
        "throttle_secs", 0
    )
    params["runconfig"]["predict_steps"] = params["runconfig"].get(
        "predict_steps", 1
    )
    params["runconfig"]["log_step_count_steps"] = params["runconfig"].get(
        "log_step_count_steps", params["runconfig"]["save_summary_steps"]
    )
    # Diasable standard estimator logs and use a custom
    # logging hook when trained with gradient accumulation.
    params["runconfig"]["disable_standard_logs"] = (
        params["optimizer"]["grad_accum_steps"] > 1
    )

    # CSConfig
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


def get_custom_stack_params(params):
    stack_params = dict()
    stack_params["ir_mode"] = "mlir-cirh"
    runconfig_params = params["runconfig"]
    use_cs = is_cs(runconfig_params)
    if (
        use_cs
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        stack_params["config"] = set_custom_config(FullConfig(), params)
    return stack_params


def set_custom_config(config, params):
    runconfig_params = params["runconfig"]
    csconfig = params.get("csconfig", {})
    use_cbfloat16 = csconfig.get("use_cbfloat16", False)
    use_vsl = params["model"].get("use_vsl", False)
    config.matching.kernel.use_legacy_vsl = use_vsl
    if (
        runconfig_params.get("multireplica", False)
        and runconfig_params.get("mode", "train") == "train"
    ):
        config.target_num_replicas = -1
        config.placement.pathfinder_inter_replica.fix_existing = TS_DISABLED
        config.placement.pathfinder_inter_replica.allow_infeasible_initial_state = (
            TS_ENABLED
        )
        config.matching.match_lair.disabled_converters.append(
            "AttentionCIRHConverter"
        )
    config.matching.kernel.numerics.layernorm_compute_dtype = (
        cb_types.T_CB16 if use_cbfloat16 else cb_types.T_F16
    )

    if runconfig_params.get("mode", "train") == "train":
        # config.placement.pathfinder.deltat_relative_margin = 0.7
        config.matching.kernel.enable_pipelined_mlm_loss = True
        config.matching.kernel.inc_pwt_estimate = True
    else:
        config.matching.add_pack_and_unpack.max_egress_per_pack = 1
        config.placement.prep_recolor_kernels.wrap_pack_kernel = True
        config.matching.match_lair.disabled_converters.append(
            "UnsortedGatherConverter"
        )
    return config
