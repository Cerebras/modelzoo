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
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from modelzoo.transformers.tf.bert.fine_tuning.qa.data import (
    eval_input_fn,
    predict_input_fn,
    train_input_fn,
)
from modelzoo.transformers.tf.bert.fine_tuning.qa.model import model_fn
from modelzoo.transformers.tf.bert.fine_tuning.qa.utils import get_params
from modelzoo.transformers.tf.bert.run import create_arg_parser, run

CS1_MODES = ["train", "predict"]


def main():
    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_dir"
    )
    parser = create_arg_parser(default_model_dir)
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)
    run(
        args,
        params,
        model_fn,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        output_layer_name="qa_logits",
        cs1_modes=CS1_MODES,
    )


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
