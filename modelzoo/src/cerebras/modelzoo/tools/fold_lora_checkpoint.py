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
import logging
import os

import yaml
from tqdm import tqdm

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.lora import (
    LoRA_Embedding,
    make_model_lora,
)
from cerebras.modelzoo.registry import registry
from cerebras.modelzoo.tools.checkpoint_converters.streaming_checkpoints import (
    StreamingCSWriter,
)


def fold_lora_model_from_file(
    model, config_path, checkpoint_path, progress_bar=False
):
    def insert_suffix_into_file(file, suffix):
        "Inserts suffix before a file extension."

        base, ext = os.path.splitext(file)
        return base + suffix + ext

    folded_config_path = insert_suffix_into_file(config_path, "_folded")
    folded_checkpoint_path = insert_suffix_into_file(checkpoint_path, "_folded")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint = cstorch.load(checkpoint_path)

    lora_params = config["model"].pop("lora_params")
    assert (
        lora_params is not None
    ), "lora_params must be included in model config"

    # Enable merge_weights for lora_params:
    def enable_merge_weights(params):
        params["merge_weights"] = True

    if isinstance(lora_params, list):
        for param_group in lora_params:
            enable_merge_weights(param_group)
    else:
        enable_merge_weights(lora_params)

    model_class = registry.get_model_class(model)
    model = model_class(config)
    if hasattr(model, "_post_device_transfer"):
        model._post_device_transfer()
    model = make_model_lora(model, lora_params)

    # Folding the embedding layer's weights breaks weight tying as the lm_head
    # and embedding layer would need different weights
    if (
        config["model"].get("share_embedding_weights")
        and hasattr(model, "model")
        and hasattr(model.model, "get_input_embeddings")
        and isinstance(model.model.get_input_embeddings(), LoRA_Embedding)
    ):
        logging.warning(
            "Your model enables weight tying between the embedding & language "
            "model head, while simultaneously updating the embeddings via LoRA."
            " In order to maintain the intended behavior, the folded checkpoint"
            " disables weight tying (i.e. share_embedding_weights=False)"
        )
        config["model"]["share_embedding_weights"] = False
        model = model_class(config)
        if hasattr(model, "_post_device_transfer"):
            model._post_device_transfer()
        model = make_model_lora(model, lora_params)

    model.load_state_dict(checkpoint["model"])

    # Flipping the model into eval mode automatically folds weights. Note that the
    # .lora_A and .lora_B keys are retained and so we'll need to prune them out
    # when building our final checkpoint later.
    model.eval()
    state_dict = model.state_dict()

    folded_checkpoint = StreamingCSWriter(folded_checkpoint_path)
    folded_checkpoint["model"] = {}

    pbar = (
        tqdm(total=len(state_dict), desc="Folding LoRA weights")
        if progress_bar
        else None
    )

    for k, v in state_dict.items():
        if not (k.endswith(".lora_A") or k.endswith(".lora_B")):
            folded_checkpoint["model"][k] = v
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    folded_checkpoint.save()
    with open(folded_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return folded_config_path, folded_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fold LoRA checkpoints')
    parser.add_argument(
        '--model',
        type=str,
        choices=registry.list_models(),
        help='Model name',
    )

    parser.add_argument(
        '-p',
        '--params',
        type=str,
        required=True,
        help='Path to .yaml file with model parameters',
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Checkpoint path',
    )

    args = parser.parse_args()

    folded_config_path, folded_checkpoint = fold_lora_model_from_file(
        args.model, args.params, args.checkpoint_path, progress_bar=True
    )

    print("Saved folded config to:", folded_config_path)
    print("Saved folded checkpoint to:", folded_checkpoint)
