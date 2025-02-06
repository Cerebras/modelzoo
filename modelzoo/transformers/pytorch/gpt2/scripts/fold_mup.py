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

"""
This script takes a path to a muP GPT checkpoint and folds muP constants into
the weights of the model to create an sP checkpoint that has approximately
equivalent behavior when used for non-training workloads.
"""
import argparse
import math
import os

import torch
import yaml

import cerebras_pytorch as cstorch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=str, required=True, help="path to source ckpt"
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="destination checkpoint file path",
    )
    parser.add_argument(
        "-p", "--params", required=True, help="path to a muP config file"
    )
    args = parser.parse_args()

    def load_model_state(ckpt_path):
        state_dict = cstorch.load(ckpt_path)
        return {k: v for (k, v) in state_dict["model"].items()}

    state_dict = load_model_state(args.src)
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)

    d_model = params["model"]["hidden_size"]
    n_heads = params["model"]["num_heads"]
    d_sqrt = math.sqrt(d_model // n_heads)
    output_scale = params["model"].pop("output_logits_scale")
    emb_scale = params["model"].pop("embeddings_scale")
    params["model"].pop("scale_qk_dot_by_d")
    params["model"]["share_embedding_weights"] = False

    # there are two different checkpoint formats in recent monolith, so we need to
    # figure out which one the current checkpoint uses
    if "model.model.lm_head.weight" in state_dict:
        prefix = "model.model"
    else:
        assert "model.lm_head.weight" in state_dict, next(iter(state_dict))
        prefix = "model"

    x = state_dict[f"{prefix}.lm_head.weight"]
    assert x is not None
    state_dict[f"{prefix}.embedding_layer.word_embeddings.weight"] = (
        x * emb_scale
    )
    if f"{prefix}.embedding_layer.position_embeddings.weight" in state_dict:
        state_dict[
            f"{prefix}.embedding_layer.position_embeddings.weight"
        ] *= emb_scale
    state_dict[f"{prefix}.lm_head.weight"] = x * output_scale

    for k in state_dict:
        if "proj_k_dense_layer" in k:
            state_dict[k] /= d_sqrt

    torch.save({"model": state_dict}, args.dest)
    out_params_path = os.path.join(os.path.dirname(args.dest), "params.yaml")
    with open(out_params_path, "w") as f:
        yaml.dump(params, f)
