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


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values
    Args:
        params: The dictionary containing the params
    """
    # Model
    params["model"]["to_float16"] = params["model"].get("to_float16", False)
    params["model"]["share_embedding_weights"] = params["model"].get(
        "share_embedding_weights", True
    )
    params["model"]["mlm_loss_weight"] = params["model"].get(
        "mlm_loss_weight", True
    )
    params["model"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )
    params["model"]["layer_norm_epsilon"] = params["model"].get(
        "layer_norm_epsilon",
        1e-6 if params["model"]["mixed_precision"] else 1e-8,
    )

    params["model"]["max_position_embeddings"] = params["model"].get(
        "max_position_embeddings", params["train_input"]["max_sequence_length"],
    )

    # Train
    params["train_input"]["mixed_precision"] = params["model"][
        "mixed_precision"
    ]
    params["train_input"]["shuffle"] = params["train_input"].get(
        "shuffle", True
    )
    params["train_input"]["shuffle_seed"] = params["train_input"].get(
        "shuffle_seed", 1
    )
    params["train_input"]["vocab_size_dna"] = params["model"].get(
        "vocab_size_dna"
    )
    params["train_input"]["vocab_size_ideas"] = params["model"].get(
        "vocab_size_ideas"
    )
    # Eval
    params["eval_input"] = deepcopy(params["train_input"])
    params["eval_input"]["shuffle"] = params["eval_input"].get("shuffle", False)
