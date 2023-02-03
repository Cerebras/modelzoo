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

import os
from copy import deepcopy

import yaml

from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.tf.model_utils.vocab_utils import get_vocab_size
from modelzoo.common.tf.run_utils import is_cs

if CSOFT_PACKAGE == CSoftPackage.SRC:
    from cerebras.pb.common.tri_state_pb2 import TS_ENABLED
    from cerebras.pb.stack.autogen_pb2 import AP_ENABLED
    from cerebras.pb.stack.full_pb2 import FullConfig
elif CSOFT_PACKAGE == CSoftPackage.WHEEL:
    from cerebras_appliance.pb.common.tri_state_pb2 import TS_ENABLED
    from cerebras_appliance.pb.stack.autogen_pb2 import AP_ENABLED
    from cerebras_appliance.pb.stack.full_pb2 import FullConfig
elif CSOFT_PACKAGE == CSoftPackage.NONE:
    pass
else:
    assert False, f"Invalid value for `CSOFT_PACKAGE`: {CSOFT_PACKAGE}"


def get_params(params_file, mode=None):

    # Load yaml into params
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    set_defaults(params, mode=mode)
    return params


def set_defaults(params, mode=None):
    for section in ["train_input", "eval_input"]:
        for key in ["vocab_file"]:
            if params.get(section, {}).get(key):
                params[section][key] = os.path.abspath(params[section][key])

    # Linformer specific
    # Embeddings
    params["model"]["use_segment_embedding"] = params["model"].get(
        "use_segment_embedding", False
    )
    params["model"]["position_embedding_type"] = (
        params["model"].get("position_embedding_type", "learned").lower()
    )
    params["model"]["embedding_size"] = params["model"].get(
        "embedding_size", params["model"]["hidden_size"]
    )
    params["model"]["share_embedding_weights"] = params["model"].get(
        "share_embedding_weights", True
    )
    params["model"]["max_position_embeddings"] = params["model"].get(
        "max_position_embeddings", None
    )

    # Initialization
    params["model"]["initializer"] = params.get(
        "initializer", {"name": "truncated_normal", "mean": 0.0, "stddev": 0.02}
    )
    params["model"]["embedding_initializer"] = params["model"].get(
        "embedding_initializer", None
    )
    params["model"]["weight_initialization_seed"] = params["model"].get(
        "weight_initialization_seed", None
    )

    # Encoder
    params["model"]["use_pre_normalization"] = params["model"].get(
        "use_pre_normalization", False
    )
    params["model"]["dropout_seed"] = params["model"].get("dropout_seed", None)
    params["model"]["attention_dropout_rate"] = params["model"].get(
        "attention_dropout_rate", 0.0
    )

    # Task-specific
    params["model"]["mlm_loss_weight"] = params["model"].get(
        "mlm_loss_weight", 1.0
    )

    params["model"]["disable_nsp"] = params["model"].get("disable_nsp", False)
    params["model"]["num_cls_classes"] = params["model"].get(
        "num_cls_classes", 2
    )
    params["model"]["layer_norm_epsilon"] = float(
        params["model"].get("layer_norm_epsilon", 1e-8)
    )

    # Flag to enable optimizations for GPU training
    params["model"]["enable_gpu_optimizations"] = params["model"].get(
        "enable_gpu_optimizations", False
    )

    # Train vocabulary size.
    params["train_input"]["vocab_size"] = get_vocab_size(
        params["train_input"].get("vocab_file"),
        params["train_input"].get("vocab_size"),
    )

    # `train_input` required parameters.
    params["train_input"]["shuffle"] = params["train_input"].get(
        "shuffle", True
    )
    params["train_input"]["repeat"] = params["train_input"].get("repeat", True)

    # Eval vocabulary size.
    if "eval_input" in params:
        params["eval_input"]["vocab_size"] = get_vocab_size(
            params["eval_input"].get("vocab_file"),
            params["eval_input"].get("vocab_size"),
        )
    else:
        # In case no `eval_input` is specified, set the params from `train_input`.
        params["eval_input"] = deepcopy(params["train_input"])

    # eval_input required parameters
    params["eval_input"]["shuffle"] = params["eval_input"].get("shuffle", False)
    params["eval_input"]["repeat"] = params["eval_input"].get("repeat", False)

    # Pass along the model's mixed_precision flag to the data processors
    params["train_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )
    params["eval_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )

    params["train_input"]["scale_mlm_weights"] = params["train_input"].get(
        "scale_mlm_weights", True
    )
    params["eval_input"]["scale_mlm_weights"] = params["eval_input"].get(
        "scale_mlm_weights", True
    )

    if params["model"]["attention_style"] == "linformer-shared-layers":
        params["model"]["use_projection_bias_in_attention"] = False

    # Set defaults for VSL padding parameters
    if params["model"].get("use_vsl") and mode != "train":
        params["model"]["use_vsl"] = False
    use_vsl = params["model"].get("use_vsl", False)
    if use_vsl:
        vocab_size = params["train_input"]["vocab_size"]
        for key, default in zip(
            ["input_pad_id", "segment_pad_id", "mlm_pad_id"],
            [vocab_size + 1, 2, -1],
        ):
            params["train_input"][key] = params["train_input"].get(key, default)
            params["model"][key] = params["train_input"][key]
        if -1 <= params["train_input"]["segment_pad_id"] <= 1:
            raise ValueError(
                f"segment pad id must be less than -1 or greater than 1"
                f" got {params['model']['segment_pad_id']}."
            )
        input_pad_id = params["train_input"]["input_pad_id"]
        if 0 < input_pad_id < vocab_size or input_pad_id == -1:
            raise ValueError(
                "input_pad_id must be the pad token, out of vocab, or a "
                f"negative index less than -1. Got {input_pad_id}."
            )
        if not params["train_input"]["mlm_pad_id"] == -1:
            raise ValueError(
                f"mlm_pad_id must be -1. Got {params['train_input']['mlm_pad_id']}"
            )
    else:
        # hard-code padding values to turn off VSL
        params["model"]["input_pad_id"] = -1
        params["model"]["segment_pad_id"] = -1

    # Optimizer
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

    # Runconfig
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


def load_pretrain_model_params(params, relative_path_dir):
    """
    When fine-tuning pre-trained weights, we want to use the same model params
    as were used during pre-training. To accomplish this, we allow fine-tuning
    yamls to provide a pretrain_params_path to load model params from.

    This function loads params from params["model"]["pretrain_params_path"],
    and merges the model section of those params with params["model"].
    pretrain_params_path can be absolute or relative to relative_path_dir.
    If a flag is provided both in params["model"] and pretrain_params_path,
    then params["model"] takes precedence.

    :param dict params: params from the fine-tuning yaml
    :param str relative_path_dir: directory to use as starting point if
        pretrain_params_path is a relative path.
    """
    pretrain_params_path = params["model"]["pretrain_params_path"]

    if not os.path.isabs(pretrain_params_path):
        pretrain_params_path = os.path.join(
            relative_path_dir, pretrain_params_path,
        )

    with open(pretrain_params_path, "r") as stream:
        model_params = yaml.safe_load(stream)["model"]

    # Avoid propagating VSL settings to the fine-tuning models.
    if "use_vsl" in model_params:
        del model_params["use_vsl"]

    # Model settings present in the FT model's yaml are used to override
    # the corresponding settings in the pretrain yaml.
    model_params.update(params["model"])

    params["model"] = model_params


def get_custom_stack_params(params):
    stack_params = dict()
    runconfig_params = params["runconfig"]
    use_cs = is_cs(runconfig_params)
    if (
        use_cs
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        stack_params["config"] = set_custom_config(FullConfig(), params)
        stack_params["ir_mode"] = "mlir-cirh"
    return stack_params


def set_custom_config(config, params):
    runconfig_params = params["runconfig"]
    linformer_attn_style = params["model"]["attention_style"]
    if runconfig_params.get("mode", "train") == "train":
        config.matching.autogen_policy = AP_ENABLED

        config.matching.remove_meta_kernels.remove_reshape_meta = TS_ENABLED

        if linformer_attn_style != "linformer-shared-layers":

            config.matching.match_lair.disabled_converters.append(
                "ReshapeMetaWithTransposeCIRHConverter"
            )

            if linformer_attn_style == "linformer":
                config.matching.match_lair.disabled_converters.append(
                    "ElementwiseCIRHConverter"
                )

            config.matching.kernel.autogen_config.always_store_inputs = (
                TS_ENABLED
            )

        config.matching.kernel.autogen_config.enable_task_based_estimator = (
            TS_ENABLED
        )

        config.matching.kernel.autogen_config.try_strip_outer_dim = TS_ENABLED

        config.matching.optimize_graph.placement_prediction.max_fill_rate = -1

        config.placement.pathfinder.deltat_relative_margin = 1.0

    return config
