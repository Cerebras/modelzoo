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
This script uses user provided SP GPT-2/3 yaml config to calculate and 
then generate a correspoding muP yaml config. 

This script requires arguments for the base model for which you tuned the 
hyperparameters: lr_base, std_base and m_embed
1) (required) input SP yaml config
2) (optional) base model hidden dimension
3) (optional) base learning rate. Ensure that the config has two 
sequential Linear schedulers. First lr scheduler should be the linear warm-up
and second scheduler should do linear decay.
4) (optional) base initialization standard deviation
5) (optional) embedding output multiplier
6) (optional) Output path to store the muP yaml config

The default values for the optional arguments base_lr, base_init_std and m_embed 
are set with the "Empirically Tuned Values"
in the Cerebras-GPT paper: https://arxiv.org/abs/2304.03208. 
Also, the default base_layer_width is set to 256 as used in this paper. 

Example usage:
python convert_config_to_mup.py -i <path/to/yaml>/params_gpt3_tiny.yaml -d_base 256 -lr_base 6.e-3 -std_base 0.08 -m_base 10.
"""

import argparse
import logging
import os
import sys

import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from cerebras.modelzoo.common.utils.run.cli_parser import get_params

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S',
)


def parse_argument():
    parser = argparse.ArgumentParser(
        description="Parsing arguments to generate muP-specific hyperparamters."
    )
    parser.add_argument(
        '--input_yaml',
        '-i',
        required=True,
        type=str,
        help='path to the input yaml file specifying the '
        'configuration for an SP GPT-2/3 model. '
        'Currently only `Constant` and `Linear` lr schdules are supported for conversion',
    )
    parser.add_argument(
        '--base_layer_width',
        '-d_base',
        required=False,
        type=int,
        default=256,
        help='base model layer width',
    )
    parser.add_argument(
        '--base_lr',
        '-lr_base',
        required=False,
        type=float,
        default=6e-3,
        help='base learning rate; ensure that the config has two '
        'sequential schedulers. First lr scheduler should be `Linear` warm-up '
        'and second scheduler should be `Linear` or `CosineDecay`.',
    )
    parser.add_argument(
        '--base_init_std',
        '-std_base',
        required=False,
        type=float,
        default=0.08,
        help='base initialization standard deviation',
    )
    parser.add_argument(
        '--m_embed',
        '-m_base',
        required=False,
        type=float,
        default=10.0,
        help='embedding output multiplier',
    )
    parser.add_argument(
        '--output_yaml',
        '-o',
        required=False,
        type=str,
        help='path to store the generated ouptut yaml file, '
        'if not provided, config will be stored under the same path '
        'as the input but with a `_mup` tag',
    )
    args = parser.parse_args()
    return args


def check_default_args(args):
    cmd_line_args = sys.argv
    if (
        '-d_base' not in cmd_line_args
        and '--base_layer_width' not in cmd_line_args
    ):
        logging.warning(
            f'base layer width not provided; using default Cerebras-GPT base model width '
            ' = {args.base_layer_width}.'
        )
    else:
        logging.info(
            f'base layer width is set to {args.base_layer_width}; ensure that you have run '
            'hyper-parameter sweep for this base model.'
        )

    if '-lr_base' not in cmd_line_args and '--base_lr' not in cmd_line_args:
        logging.warning(
            f'base lr not provided; using default Cerebras-GPT base lr = {args.base_lr}.'
        )
    else:
        logging.info(
            f'base lr is set to {args.base_lr}; ensure that you have run hyper-parameter '
            'sweep to find the optimal lr with the base model.'
        )

    if (
        '-std_base' not in cmd_line_args
        and '--base_init_std' not in cmd_line_args
    ):
        logging.warning(
            f'base initialization std not provided; using default Cerebras-GPT base init std'
            ' = {args.base_init_std}.'
        )
    else:
        logging.info(
            f'base init std is set to {args.base_init_std}; ensure that you have run '
            'hyper-parameter sweep to find the optimal base init std with the base model.'
        )

    if '-m_base' not in cmd_line_args and '--m_embed' not in cmd_line_args:
        logging.warning(
            f'base embeddings scaling factor not provided; using default Cerebras-GPT '
            'base embeddings scale = {args.m_embed}.'
        )
    else:
        logging.info(
            f'base embeddings scaling factor is set to {args.m_embed}; ensure that you '
            'have run hyper-parameter sweep to find the optimal embeddings scale with the base model.'
        )


def generate_mup_config(args):
    input_yaml = os.path.abspath(args.input_yaml)
    params = get_params(input_yaml)
    # check if required args are provided
    if "model" not in params or "hidden_size" not in params["model"]:
        raise RuntimeError(
            f"The input yaml must be a valid gpt configuration, \
        but {args.input_yaml} didn't contain "
            "the `hidden_size` key"
        )

    if "num_hidden_layers" not in params["model"]:
        raise RuntimeError(
            f"The input yaml must be a valid gpt configuration, \
        but {args.input_yaml} didn't contain "
            "the `num_hidden_layers` key"
        )

    # calcualte values for muP transferable parameters
    d_model = params['model']['hidden_size']
    num_hidden_layers = params['model']['num_hidden_layers']
    width_mult = float(d_model / args.base_layer_width)
    embedding_init_std = float(args.base_init_std)
    init_std = float(args.base_init_std / np.sqrt(width_mult))
    output_init_std = float(
        args.base_init_std / np.sqrt(2 * width_mult * num_hidden_layers)
    )
    base_lr = float(args.base_lr)
    lr_adjustment = float(1 / width_mult)
    output_logits_scale = float(1 / width_mult)
    embeddings_scale = float(args.m_embed)
    scale_qk_dot_by_d = True

    # Fill muP Transferred params in the yaml

    # update initializations
    params['model']['embedding_initializer'] = {
        'mean': 0.0,
        'name': 'truncated_normal',
        'std': embedding_init_std,
        'a': (-2.0) * embedding_init_std,
        'b': 2.0 * embedding_init_std,
    }
    params['model']['initializer'] = {
        'mean': 0.0,
        'name': 'truncated_normal',
        'std': init_std,
        'a': (-2.0) * init_std,
        'b': 2.0 * init_std,
    }

    params['model']['output_layer_initializer'] = {
        'mean': 0.0,
        'name': 'truncated_normal',
        'std': output_init_std,
        'a': (-2.0) * output_init_std,
        'b': 2.0 * output_init_std,
    }

    # update scaling multipliers
    params['model']['output_logits_scale'] = output_logits_scale
    params['model']['embeddings_scale'] = embeddings_scale
    # Attention logits scaling by d_head
    params['model']['scale_qk_dot_by_d'] = scale_qk_dot_by_d
    # Add optimizer lr and scaling factor for attention block weights
    #  linear warmup + linear decay lr schedule is recommended for
    # GPT-2/3 models.
    if params['optimizer'].get('learning_rate'):
        if not isinstance(params['optimizer'].get('learning_rate'), list):
            raise RuntimeError(
                f'{params["optimizer"].get("learning_rate")} is not supported for config generation. '
                'Linear Warm-up + Linear/Cosine Decay lr schedule should be used for GPT-2/3'
            )
        else:
            for lr_schedule in params['optimizer'].get('learning_rate'):
                if lr_schedule.get('scheduler'):
                    if lr_schedule.get('scheduler') not in [
                        'Linear',
                        'CosineDecay',
                    ]:
                        raise RuntimeError(
                            f'{lr_schedule.get("scheduler")} is not supported for config generation'
                        )
                    else:
                        if lr_schedule.get('initial_learning_rate') != 0:
                            current_initial_lr = lr_schedule.get(
                                'initial_learning_rate'
                            )
                            lr_schedule['initial_learning_rate'] = base_lr
                            if (
                                not lr_schedule.get('end_learning_rate')
                                and lr_schedule.get('end_learning_rate') != 0
                            ):
                                raise RuntimeError(
                                    f'`end_learning_rate` needs to be provided'
                                )
                            if lr_schedule['end_learning_rate'] != 0.0:
                                lr_schedule['end_learning_rate'] = base_lr * (
                                    lr_schedule['end_learning_rate']
                                    / current_initial_lr
                                )
                        else:
                            lr_schedule['end_learning_rate'] = base_lr
    else:
        params['optimizer']['learning_rate'] = base_lr
    params['optimizer']['adjust_learning_rate'] = {
        'decoder_kernel': lr_adjustment
    }

    output_file_name = (
        args.output_yaml
        if args.output_yaml
        else os.path.splitext(input_yaml)[0] + '_muP.yaml'
    )
    with open(output_file_name, 'w') as f:
        yaml.dump(params, f, sort_keys=False)
    logging.info(f'muP config saved to {output_file_name}')


def main():
    args = parse_argument()
    check_default_args(args)
    generate_mup_config(args)


if __name__ == '__main__':
    main()
