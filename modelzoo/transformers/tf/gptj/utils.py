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

from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.model_utils.count_lines import count_lines
from modelzoo.common.tf.run_utils import is_cs
from modelzoo.transformers.tf.transformer_utils import get_bits_per_x_dataset

if CSOFT_PACKAGE == CSoftPackage.SRC:
    from cerebras.pb.stack.full_pb2 import FullConfig
elif CSOFT_PACKAGE == CSoftPackage.WHEEL:
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


def set_defaults(params):
    params = get_bits_per_x_dataset(params)

    tparams = params["train_input"]
    eparams = params["eval_input"]

    # Train vocabulary size
    vocab_size = tparams.get("vocab_size")
    if not vocab_size:
        vocab_size = count_lines(tparams["vocab_file"])
    tparams["vocab_size"] = vocab_size

    # train_input required parameters
    tparams["shuffle"] = tparams.get("shuffle", True)
    tparams["repeat"] = tparams.get("repeat", True)

    eparams = params.get("eval_input", {})
    # Eval vocabulary size
    if not eparams:
        eparams["vocab_size"] = tparams["vocab_size"]
    else:
        vocab_size = eparams.get("vocab_size")
        if not vocab_size:
            vocab_size = count_lines(eparams["vocab_file"])
        eparams["vocab_size"] = vocab_size

    # eval_input required parameters
    eparams["shuffle"] = eparams.get("shuffle", False)
    eparams["repeat"] = eparams.get("repeat", False)

    # Set up model defaults
    mparams = params["model"]
    mparams["num_hidden_layers"] = mparams.get("num_hidden_layers", 1)
    mparams["embedding_dropout_rate"] = mparams.get(
        "embedding_dropout_rate", 0.0
    )

    init_keys = [
        "embedding_initializer",
        "initializer",
        "output_layer_initializer",
    ]
    for _key in init_keys:
        params = set_initializers(params, _key)

    mparams["weight_initialization_seed"] = mparams.get(
        "weight_initialization_seed", None
    )

    # position embeddings
    mparams["max_position_embeddings"] = mparams.get(
        "max_position_embeddings", None
    )
    mparams["rotary_dim"] = mparams.get("rotary_dim", -1)

    mparams["dropout_seed"] = mparams.get("dropout_seed", None)
    mparams["embedding_dropout_rate"] = mparams.get(
        "embedding_dropout_rate", 0.0
    )
    mparams["attention_dropout_rate"] = mparams.get(
        "attention_dropout_rate", 0.0
    )
    mparams["residual_dropout_rate"] = mparams.get("residual_dropout_rate", 0.0)
    mparams["share_embedding_weights"] = mparams.get(
        "share_embedding_weights", True
    )
    mparams["layer_norm_epsilon"] = float(
        mparams.get("layer_norm_epsilon", 1e-8)
    )
    mparams["use_untied_layer_norm"] = mparams.get(
        "use_untied_layer_norm", False
    )
    mparams["backbone_only"] = mparams.get("backbone_only", False)
    mparams["use_bias_in_output"] = mparams.get("use_bias_in_output", True)
    mparams["use_cache"] = mparams.get("use_cache", False)
    mparams["attention_softmax_fp32"] = mparams.get(
        "attention_softmax_fp32", True
    )

    # Optimizer parameters
    oparams = params["optimizer"]
    oparams["loss_scaling_factor"] = oparams.get("loss_scaling_factor", 1.0)
    oparams["max_gradient_norm"] = oparams.get("max_gradient_norm", None)
    oparams["grad_accum_steps"] = oparams.get("grad_accum_steps", 1)
    oparams["log_summaries"] = oparams.get("log_summaries", True)

    # Runconfig parameters
    rparams = params["runconfig"]
    rparams["save_summary_steps"] = rparams.get("save_summary_steps", 100)
    rparams["save_checkpoints_steps"] = rparams.get(
        "save_checkpoints_steps", 1000
    )
    rparams["keep_checkpoint_max"] = rparams.get("keep_checkpoint_max", 5)
    rparams["tf_random_seed"] = rparams.get("tf_random_seed", None)
    rparams["enable_distributed"] = rparams.get("enable_distributed", False)
    rparams["throttle_secs"] = rparams.get("throttle_secs", 0)
    rparams["multiple_workers"] = rparams.get("multiple_workers", False)
    rparams["max_steps"] = rparams.get("max_steps", 100)
    rparams["eval_steps"] = rparams.get("eval_steps", 0)

    # Pass along the TFBaseModel's mixed_precision flag to the data processors
    tparams["mixed_precision"] = mparams.get("mixed_precision", False)
    eparams["mixed_precision"] = mparams.get("mixed_precision", False)

    rparams["log_step_count_steps"] = rparams.get(
        "log_step_count_steps", rparams["save_summary_steps"]
    )

    # Diasable standard estimator logs and use a custom
    # logging hook when trained with gradient accumulation.
    rparams["disable_standard_logs"] = rparams.get(
        "disable_standard_logs", oparams["grad_accum_steps"] > 1
    )

    # set default weight_servers
    params["runconfig"]["num_wgt_servers"] = params["runconfig"].get(
        "num_wgt_servers", 12
    )


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
    return stack_params


def set_custom_config(config, params):
    return config


def configure_variance_scaling_parameters(
    params,
    scale=None,
    scale_type=None,
    mode="fan_in",
    distribution="truncated_normal",
):
    """Configure the Variance Scaling parameters as per given arguments.
    Support for `small_init` from `Transformers without Tears:\
         Improving the Normalization of Self-Attention <https://arxiv.org/abs/1910.05895>`
    and `wang_init` as defined by `<https://github.com/kingoflolz/mesh-transformer-jax>`

    Args:
        params (dict): Model parameters for runtime.
        scale (float): Scale to be passed to the TensorFlow variance scaling
            initializer. Defaults to None.
        scale_type (float): Type of scaling to be done using variance scaling.
            Support for `small_init` and `wang_init` for now. Defaults to None.
        mode (str): Mode to run variance_scaling with. Defaults to `fan_in`.
        distribution (str): Distribution to run variance scaling with.
            Defaults to `truncated_normal`.

    Returns:
        A dictionary, which will be passed to the ``create_initializer`` function
        in the model code, with mode and distribution enabled correctly.
    """

    if scale and scale_type:
        raise ValueError(
            "Cannot define both scale and scale_type, please check arguments!!"
        )
    elif not scale and not scale_type:
        raise ValueError(
            "Cannot leave both scale and scale_type undefined, please check arguments!!"
        )

    if scale_type == "small_init":
        scale = 2 / 5
    elif scale_type == "wang_init":
        num_hidden_layers = params["num_hidden_layers"]
        scale = 4 / num_hidden_layers ** 2

    return {
        "name": "variance_scaling",
        "scale": scale,
        "mode": mode,
        "distribution": distribution,
    }


def configure_scaled_init_normal_parameters(
    params, key=None, mean=0.0, sigma=1.0, distribution="normal",
):
    """Configure the scaled init (truncated) normal distribution parameters.

    Args:
        params (dict): Parameters for runtime.
        key (dict): Optional scaling factor to use. Defaults to `None`.
        mean (float): Mean for normal distribution. Defaults to `0.0`.
        sigma (float): Standard deviation for normal distribution.
            Defaults to `1.0`.
        distribution (str): Distribution for sampling weights. One of
            `truncated_normal` or `normal`. Defaults to `normal`.
    """

    assert distribution in ["normal", "truncated_normal"]

    scaling_key = 1.0  # corresponds to no scaling
    if key:
        if key == "num_hidden_layers":
            scaling_key = params["model"].get(key, None) * 2
        elif key == "vocab_size":
            scaling_key = params["train_input"].get(key, None)
        else:
            raise ValueError(
                f"expected one of num_hidden_layers or vocab_size, got {key}."
            )

    stddev = sigma / math.sqrt(scaling_key)
    return {
        "name": distribution,
        "mean": mean,
        "stddev": stddev,
    }


def set_initializers(params, key):
    """Set initializations as specified by the user. This ensures flexibility in
    user inputs, as most models have the same architecture, but use
    different initializations to stabilize the training runs.

    Args:
        params (dict): Parameters for runtime.
        key (str): String specifying which initialization to set in the params
            dictionary.

    Returns:
        A dictionary containing the specified initialization scheme for given key.
    """

    mparams = params["model"]
    initializer = mparams.get(key, None)
    if not initializer:
        mparams[key] = get_default_initializations(params, key)
        return params

    initializer = initializer[0]
    if initializer.get("name") == "variance_scaling":
        mparams[key] = configure_variance_scaling_parameters(
            params=mparams,
            scale=initializer.get("scale"),
            scale_type=initializer.get("scale_type"),
            mode=initializer.get("mode", "fan_in"),
            distribution=initializer.get("distribution", "truncated_normal"),
        )
    elif initializer.get("name") == "scaled_init_normal":
        mparams[key] = configure_scaled_init_normal_parameters(
            params=params,
            key=initializer.get("key", None),
            mean=initializer.get("mean", 0.0),
            sigma=initializer.get("sigma", 1.0),
            distribution=initializer.get("distribution", "truncated_normal"),
        )

    return params


def get_default_initializations(params, key):
    """Get default initializations when nothing is specified from the user. This
    is to enable the 6B Gpt-J model, with the initializations specified in
    `<https://github.com/kingoflolz/mesh-transformer-jax>`_.

    Args:
        params (dict): Parameters for runtime.
        key (str): String specifying which initialization to set in the params
            dictionary.

    Returns:
        A dictionary containing the associated default initialization scheme for
        given key.
    """

    if key == "embedding_initializer":
        # 50257 is the default from Gpt-2 / J tokenization
        vocab_size = params["train_input"].get("vocab_size", 50257)
        return {
            "name": "truncated_normal",
            "mean": 0.0,
            "stddev": 1.0 / math.sqrt(vocab_size),
        }
    elif key == "initializer":
        return {
            "name": "variance_scaling",
            "scale": 1.0,
            "mode": "fan_in",
            "distribution": "truncated_normal",
        }
    elif key == "output_layer_initializer":
        num_hidden_layers = params["model"].get("num_hidden_layers")
        return {
            "name": "variance_scaling",
            "scale": 4.0 / num_hidden_layers ** 2,
            "mode": "fan_in",
            "distribution": "truncated_normal",
        }
    else:
        raise ValueError(
            f"{key} currently not supported, check value passed in."
        )
