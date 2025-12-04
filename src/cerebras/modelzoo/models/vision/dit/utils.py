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
from enum import Enum

# Pass model settings into data loader.
_model_to_input_map = [
    # latent shape
    "label_dropout_rate",
    "latent_size",
    "latent_channels",
    # diffusion & related params for performing gd
    "schedule_name",
]


class BlockType(Enum):
    ADALN_ZERO = "adaln_zero"

    @classmethod
    def values(cls):
        return [b.value for b in BlockType]

    @classmethod
    def get(cls, blk):
        if isinstance(blk, str):
            return BlockType(blk)
        elif isinstance(blk, Enum):
            return blk
        else:
            raise ValueError(
                f"Unsupported type {type(blk)}, supported are `str` and `Enum`"
            )


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values
    Args:
        params: The dictionary/object containing the params.
    """
    runconfig = params["runconfig"]

    # train input_required parameters
    _set_input_defaults(params)
    _set_model_defaults(params)
    _copy_params_across(params)
    if params.model.fp16_type is None:
        params.model.fp16_type = "bfloat16"
    # Runconfig related
    if runconfig["checkpoint_steps"] == 0:
        logging.warning(
            "Setting `runconfig.checkpoint_steps` to `runconfig.max_steps`. Setting to 0 only saves initial checkpoint"
        )
        runconfig["checkpoint_steps"] = runconfig["max_steps"]

    return params


def _set_model_defaults(params):
    # model related parameters
    mparams = params["model"]
    tparams = params["train_input"]
    mparams["num_diffusion_steps"] = tparams["num_diffusion_steps"]
    mparams["num_classes"] = tparams["num_classes"]
    mparams["beta_start"] = mparams.get("beta_start", 0.0001)
    mparams["beta_end"] = mparams.get("beta_end", 0.02)

    tparams["vae_scaling_factor"] = params["model"]["vae"]["scaling_factor"]
    params["eval_input"]["vae_scaling_factor"] = params["model"]["vae"][
        "scaling_factor"
    ]

    mparams["vae"]["in_channels"] = tparams["image_channels"]
    mparams["vae"]["out_channels"] = tparams["image_channels"]

    mparams["vae"]["scaling_factor"] = mparams["vae"].get(
        "scaling_factor", 0.18215
    )
    mparams["latent_channels"] = mparams.get(
        "latent_channels", mparams["vae"]["latent_channels"]
    )

    # Slight change to accommodate None handling due to config classes
    latent_size = mparams.get("latent_size", None)
    if latent_size is None:
        mparams["latent_size"] = mparams["vae"]["latent_size"]

    latent_dims = [mparams["latent_channels"]] + mparams["latent_size"]
    logging.info(f"Using VAE output Dimensions (C, H, W): {latent_dims}")
    mparams["block_type"] = mparams.get(
        "block_type", BlockType.ADALN_ZERO.value
    )
    if mparams["block_type"] not in BlockType.values():
        raise ValueError(
            f"Unsupported DiT block type {mparams['block_type']}. ",
            f"Supported values are {BlockType.values()}.",
        )
    logging.info(f"Using DiT block type : {mparams['block_type']}")

    if mparams["fp16_type"] == "bfloat16":
        params["optimizer"]["loss_scaling_factor"] = 1.0

    # Regression Head
    # False -> linear + unpatchify for regression head
    mparams["use_conv_transpose_unpatchify"] = mparams.get(
        "use_conv_transpose_unpatchify", True
    )
    if not mparams["use_conv_transpose_unpatchify"]:
        raise ValueError(
            f"Using linear layer + unpatchify in RegressionHead is unsupported at this time, "
            f"please set `model.use_conv_transpose_unpatchify` to True"
        )

    _set_layer_initializer_defaults(params)
    _set_reverse_process_defaults(params)


def _set_reverse_process_defaults(params):
    mparams = params["model"]
    rparams = mparams.get("reverse_process", {})
    if rparams:
        rparams["sampler"]["num_diffusion_steps"] = rparams["sampler"].get(
            "num_diffusion_steps", mparams["num_diffusion_steps"]
        )
        rparams["batch_size"] = rparams.get("batch_size", 32)
        rparams["pipeline"]["num_classes"] = rparams["pipeline"].get(
            "num_classes", mparams["num_classes"]
        )
        rparams["pipeline"]["custom_labels"] = rparams["pipeline"].get(
            "custom_labels", None
        )
        # For DDPM Sampler only
        if rparams["sampler"]["name"] == "ddpm":
            rparams["sampler"]["variance_type"] = "fixed_small"


def _set_layer_initializer_defaults(params):
    # Modifies in-place
    mparams = params["model"]

    # Patch Embedding
    mparams["projection_initializer"] = {"name": "xavier_uniform", "gain": 1.0}
    mparams["init_conv_like_linear"] = mparams.get(
        "init_conv_like_linear", mparams["use_conv_patchified_embedding"]
    )

    # Timestep Embedding MLP
    mparams["timestep_embedding_initializer"] = {
        "name": "normal",
        "mean": 0.0,
        "std": mparams["initializer_range"],
    }

    # Label Embedding table
    mparams["label_embedding_initializer"] = {
        "name": "normal",
        "mean": 0.0,
        "std": mparams["initializer_range"],
    }

    # Attention
    mparams["attention_initializer"] = {"name": "xavier_uniform", "gain": 1.0}

    # ffn
    mparams["ffn_initializer"] = {"name": "xavier_uniform", "gain": 1.0}

    # Regression Head FFN
    mparams["head_initializer"] = {"name": "zeros"}


def _set_input_defaults(params):
    # Modifies in place
    # train input_required parameters
    tparams = params["train_input"]
    tparams["shuffle"] = tparams.get("shuffle", True)
    tparams["shuffle_seed"] = tparams.get("shuffle_seed", 4321)
    tparams["num_classes"] = tparams.get("num_classes", 1000)
    tparams["noaugment"] = tparams.get("noaugment", False)
    tparams["drop_last"] = tparams.get("drop_last", True)
    tparams["num_workers"] = tparams.get("num_workers", 0)
    tparams["prefetch_factor"] = tparams.get("prefetch_factor", 10)
    tparams["persistent_workers"] = tparams.get("persistent_workers", True)

    if tparams["noaugment"]:
        tparams["transforms"] = None
        logging.info(
            f"Since `noaugment`={tparams['noaugment']}, the transforms are set to None"
        )
    tparams["use_worker_cache"] = tparams.get("use_worker_cache", False)

    # eval input_required parameters
    eparams = params["eval_input"]
    eparams["shuffle"] = eparams.get("shuffle", False)
    eparams["shuffle_seed"] = eparams.get("shuffle_seed", 4321)
    eparams["noaugment"] = eparams.get("noaugment", False)
    eparams["drop_last"] = eparams.get("drop_last", True)
    eparams["num_workers"] = eparams.get("num_workers", 0)
    eparams["prefetch_factor"] = eparams.get("prefetch_factor", 10)
    eparams["persistent_workers"] = eparams.get("persistent_workers", True)
    if eparams["noaugment"]:
        eparams["transforms"] = None
        logging.info(
            f"Since `noaugment`={eparams['noaugment']}, the transforms are set to None"
        )
    eparams["use_worker_cache"] = eparams.get("use_worker_cache", False)


def _copy_params_across(params):
    for _key_map in _model_to_input_map:
        if isinstance(_key_map, tuple):
            assert len(_key_map) == 2, f"Tuple {_key_map} does not have len=2"
            model_key, input_key = _key_map
        else:
            model_key = input_key = _key_map

        for section in ["train_input", "eval_input"]:
            params[section][input_key] = params["model"][model_key]
