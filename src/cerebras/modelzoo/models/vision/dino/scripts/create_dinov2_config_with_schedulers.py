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
import math
from copy import deepcopy

import yaml


def get_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    """
    layer_id = num_layers + 1  # default
    if "lr_emb_params" in name or "lr_proj_params" in name:
        layer_id = 0
    elif "layer" in name:
        layer_id = int(name.split("_")[2]) + 1

    lr_multiplier = lr_decay_rate ** (num_layers + 1 - layer_id)
    if "lr_proj_params" in name:
        lr_multiplier *= patch_emb_multiplier
    print(f"Layer name {name}, id: {layer_id}: {lr_multiplier}")
    return lr_multiplier


def make_lr_schedule(
    base_config_lr_schedule, lr_decay_rate, num_layers, layer_name
):
    """
    Make lr schedule for different ViT blocks.
    """
    lr_schedule = deepcopy(base_config_lr_schedule)
    for i, scheduler in enumerate(lr_schedule["SequentialLR"]["schedulers"]):
        for key, value in scheduler.items():
            for k, v in value.items():
                if "end_learning_rate" in k and i == 0:
                    value[k] = v * get_lr_decay_rate(
                        name=layer_name,
                        lr_decay_rate=lr_decay_rate,
                        num_layers=num_layers,
                    )
                elif "initial_learning_rate" in k and i == 1:
                    value[k] = v * get_lr_decay_rate(
                        name=layer_name,
                        lr_decay_rate=lr_decay_rate,
                        num_layers=num_layers,
                    )
    lr_schedule["SequentialLR"]["param_group_tags"] = layer_name
    print(f"{lr_schedule}")
    return lr_schedule


def get_linear_lr_value(total_iters, initial_lr, end_lr, current_iter):
    """
    Calculate linear learning rate value.
    """
    return initial_lr + (end_lr - initial_lr) * (current_iter / total_iters)


def make_last_layer_lr_schedule(
    base_config_lr_schedule, epochs_to_freeze=1, tag="lr_last_layer_params"
):
    """
    Make lr schedule for different ViT blocks.
    """
    lr_schedule = deepcopy(base_config_lr_schedule)
    # for last layer, freeze for 1 epoch, so set lr to 0, then start
    # the warmup from the value from get_linear_lr_value and complete
    # remaining warmup epochs
    n_iters_to_freeze = epochs_to_freeze * total_iters // 100
    phase1_initial_lr = 0.0
    phase1_final_lr = get_linear_lr_value(
        total_iters * 10 // 100, 0.0, lr_val, n_iters_to_freeze
    )
    phase2_total_iters = (10 * total_iters) // 100 - n_iters_to_freeze
    for i, scheduler in enumerate(lr_schedule["SequentialLR"]["schedulers"]):
        for key, value in scheduler.items():
            for k, v in value.items():
                if "initial_learning_rate" in k and i == 0:
                    value[k] = phase1_final_lr
                if "end_learning_rate" in k and i == 0:
                    value[k] = lr_val
                    value["total_iters"] = phase2_total_iters

    lr_schedule["SequentialLR"]["schedulers"].insert(
        0,
        {
            "LinearLR": {
                "initial_learning_rate": phase1_initial_lr,
                "end_learning_rate": phase1_initial_lr,
                "total_iters": n_iters_to_freeze,
            }
        },
    )
    lr_schedule["SequentialLR"]["param_group_tags"] = tag
    return lr_schedule


if __name__ == "__main__":
    # output_file_name = "dinov2_large_224_bs1024.yaml"
    # input_file_name = "params_dinov2_large_patch14_ref_not_for_cs_run.yaml"

    parser = argparse.ArgumentParser(
        description="Create learning rate schedulers"
    )
    parser.add_argument(
        "--input_file_name",
        type=str,
        required=True,
        help="Path to the input YAML file containing the base configuration",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        required=True,
        help="Path to the output YAML file to save the generated learning rate schedulers",
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.004, help="Base learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size"
    )
    parser.add_argument(
        "--total_iters",
        type=int,
        default=125000,  # 100 epochs
        help="Total number of iterations",
    )
    parser.add_argument(
        "--lr_decay_rate",
        type=float,
        default=0.9,
        help="Learning rate decay rate per layer",
    )
    parser.add_argument(
        "--patch_emb_multiplier",
        type=float,
        default=0.2,
        help="Patch embedding multiplier",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="End value for weight decay",
    )

    args = parser.parse_args()

    base_lr = args.base_lr
    batch_size = args.batch_size
    total_iters = args.total_iters
    lr_decay_rate = args.lr_decay_rate
    patch_emb_multiplier = args.patch_emb_multiplier

    input_file_name = args.input_file_name
    output_file_name = args.output_file_name

    with open(input_file_name, "r") as f:
        config = yaml.safe_load(f)

    num_layers = config["trainer"]["init"]["model"]["image_model_trunks"][0][
        "image_model"
    ]["num_hidden_layers"]

    train_batch_size_config = config["trainer"]["fit"]["train_dataloader"][
        "batch_size"
    ]
    assert train_batch_size_config == batch_size, (
        f"Batch size in config: {train_batch_size_config} is not same as input "
        f"batch size: {batch_size}. Please specify the same batch size in the "
        f"config for proper LR calculation."
    )

    max_steps = config["trainer"]["init"]["loop"]["max_steps"]
    assert max_steps == total_iters, (
        f"Total iterations in config: {max_steps} is not same as input "
        f"total iterations: {total_iters}. Please specify the same total "
        f"iterations in the config for proper LR calculation."
    )

    lr_val = math.sqrt(batch_size / 1024) * base_lr

    wd_schedule = {
        "CosineDecayWD": {
            "initial_val": 0.04,
            "end_val": args.weight_decay_end,
            "total_iters": total_iters,
            "param_group_tags": "wd_params",
        }
    }

    base_config_lr_schedule = {
        "SequentialLR": {
            "schedulers": [
                {
                    "LinearLR": {
                        "initial_learning_rate": 0.0,
                        "end_learning_rate": lr_val,
                        "total_iters": int(0.1 * total_iters),  # 10 epochs
                    }
                },
                {
                    "CosineDecayLR": {
                        "initial_learning_rate": lr_val,
                        "end_learning_rate": 1.0e-06,
                        "total_iters": int(0.9 * total_iters),  # 90 epochs
                    }
                },
            ],
            "param_group_tags": "default_lr_params",
        }
    }

    all_tags = [
        "wd_params",
        "lr_emb_params",
        "lr_proj_params",
        "lr_last_layer_params",
        "default_lr_params",
    ]
    all_tags.extend([f"lr_layer_{i}_params" for i in range(num_layers)])

    all_param_groups = [
        # there's no wd applied to norm layes anywhere ,
        [
            "*.mask_token",  # mask token
            "*.cls_embedding",  # cls token
            "*.embedding_layer.*.weight",  # embedding layer weights
            "*.mlp.*.weight",  # for heads except last layer
            "*.last_layer.weight*",  # for last layer, not usage of * at the end
            "*.encoder.*.self_attn.*.weight",  # all attention weights in encoder layers
            "*.encoder.*.ffn.*.weight",  # all ffn weights in encoder layers
        ],
        ["*position_embeddings*", "*.mask_token*", "*.cls_embedding*"],
        ["*.linear_proj.*"],
        ["*last_layer.*"],
        ["*.mlp.*", "*.transformer_encoder.norm.*"],
    ]

    all_param_groups.extend(
        [
            [
                f"*.transformer_encoder.layers.{i}.*",
            ]
            for i in range(num_layers)
        ]
    )

    assert len(all_param_groups) == len(all_tags)

    groups_to_tags = dict(zip(all_tags, all_param_groups))

    output_config = deepcopy(config)
    optimizer_config = output_config["trainer"]["init"]["optimizer"]
    # delete existing scheduler
    del output_config["trainer"]["init"]["schedulers"]

    param_groups = []

    for tag, param_group in groups_to_tags.items():
        param_groups.append(
            {
                "params": param_group,
                "tag": tag,
            }
        )

    optimizer_config["AdamW"]["params"] = param_groups

    whole_schedule = []

    for tag in all_tags:
        if tag == "wd_params":
            whole_schedule.append(wd_schedule)
        elif tag == "lr_last_layer_params":
            whole_schedule.append(
                make_last_layer_lr_schedule(base_config_lr_schedule, tag=tag)
            )
        else:
            whole_schedule.append(
                make_lr_schedule(
                    base_config_lr_schedule, lr_decay_rate, num_layers, tag
                )
            )

    output_config["trainer"]["init"]["schedulers"] = whole_schedule

    with open(output_file_name, "w") as f:
        f.write(
            "# This file is generated by create_dinov2_config_with_schedulers.py\n"
        )
        f.write("# --input_file_name: {}\n".format(input_file_name))
        yaml.dump(
            output_config,
            f,
            sort_keys=False,
            indent=2,
            default_flow_style=False,
        )
    print("Output config has been saved successfully.")
