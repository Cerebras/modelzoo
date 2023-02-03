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

from modelzoo.transformers.tf.bert.utils import load_pretrain_model_params
from modelzoo.transformers.tf.bert.utils import (
    set_defaults as set_bert_defaults,
)


def get_params(params_file):
    """
    Reads in params from yaml, fills in bert pretrain params if provided,
    uses defaults from bert's utils.py.

    :param  params_file: path to yaml with classifier params
    :returns: params dict
    """
    # Load yaml into params
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)

    set_defaults(params)

    return params


def set_defaults(params):
    if "pretrain_params_path" in params["model"]:
        load_pretrain_model_params(
            params, os.path.dirname(os.path.abspath(__file__))
        )

    set_bert_defaults(params)

    # cls-layer params
    params["model"]["cls_dropout_rate"] = params["model"].get(
        "cls_dropout_rate", 0.0
    )
    # use eval_input params as defaults for predict_input
    predict_input_params = params["eval_input"].copy()
    if "predict_input" in params:
        predict_input_params.update(params["predict_input"])
    params["predict_input"] = predict_input_params
