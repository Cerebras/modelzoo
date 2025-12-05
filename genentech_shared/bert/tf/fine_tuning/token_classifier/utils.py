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

import yaml
from genentech_shared.bert.tf.utils import get_pfam_vocab

from modelzoo.transformers.tf.bert.utils import load_pretrain_model_params
from modelzoo.transformers.tf.bert.utils import (
    set_defaults as set_bert_defaults,
)


def get_params(params_file):
    """
    Reads in params from yaml, fills in bert pretrain params if provided,
    uses defaults from bert's utils.py.
    :param  params_file: path to yaml with bert token classifier params.
    :returns: params dict.
    """
    # Load yaml into params.
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)

    vocab_size = len(get_pfam_vocab()[0])
    params["train_input"]["vocab_size"] = vocab_size
    if "eval_input" in params:
        params["eval_input"]["vocab_size"] = vocab_size

    if "pretrain_params_path" in params["model"]:
        load_pretrain_model_params(
            params, os.path.dirname(os.path.abspath(__file__))
        )

    set_defaults(params)

    return params


def set_defaults(params):
    set_bert_defaults(params)

    params["model"]["include_padding_in_loss"] = params["model"].get(
        "include_padding_in_loss", False
    )
    params["model"]["loss_weight"] = params["model"].get("loss_weight", 1.0)

    # Encoder output dropout layer params.
    params["model"]["encoder_output_dropout_rate"] = params["model"].get(
        "encoder_output_dropout_rate", 0.1
    )

    # Check number of classes.
    assert (
        "num_classes" in params["train_input"]
    ), "No number of classes specified in `train_input` field."
    num_classes = params["train_input"]["num_classes"]
    assert num_classes in {
        3,
        8,
    }, f"Wrong number of classes specified, available number of classes are `3` and `8`. Got: {num_classes}"
    params["eval_input"]["num_classes"] = num_classes
    params["model"]["num_classes"] = num_classes

    # Getting the path of label vocab file to use during predict mode in `model_fn`.
    params["model"]["label_vocab_file"] = params["train_input"].get(
        "label_vocab_file", None
    )

    # Use `eval_input` params as defaults for `predict_input`.
    predict_input_params = params["eval_input"].copy()
    if "predict_input" in params:
        predict_input_params.update(params["predict_input"])

    params["predict_input"] = predict_input_params
