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
import re
from typing import Any, Dict

import torch
import yaml

import cerebras.pytorch as cstorch

logger = logging.getLogger(__name__)


###############################################################################
#                           CHECKPOINT CONVERSION                             #
###############################################################################
def convert_dinov2_to_vit(
    state_dict_in: Dict[str, torch.Tensor],
    use_bias_in_output: bool = True,
    num_classes: int = 1000,
    prefix: str = "image_model_trunks.model.0.",
) -> Dict[str, torch.Tensor]:
    """
    Convert a DINOv2 teacher model checkpoint to a ViTClassification-style state_dict.

    Args:
        state_dict_in (Dict[str, torch.Tensor]): The input DINOv2 checkpoint state_dict.
        use_bias_in_output (bool): Whether to include a bias term in the final classifier.
        num_classes (int): The output dimension of the classifier.
        prefix (str): The prefix in the original state_dict to replace.

    Returns:
        Dict[str, torch.Tensor]: A ViTClassification-style state_dict.
    """
    # Prepare the output state_dict
    state_dict_out: Dict[str, torch.Tensor] = {}

    # Exclude keys that do not map directly to the ViT model.
    exclude_keys = [
        prefix + "embedding_layer.mask_token",
        prefix + "embedding_layer.interpolation_matrix",
    ]

    # Convert backbone weights from the teacher trunk to ViT format
    for key, value in state_dict_in.items():
        if key.startswith(prefix) and key not in exclude_keys:
            # We only replace the initial prefix once to avoid accidental replacements
            # if the substring appears again in the key.
            new_key = re.sub(f"^{prefix}", "vit_model.", key)
            state_dict_out[new_key] = value

    # Verify the final layer norm bias exists so we can infer hidden size
    last_norm_bias_key = f"{prefix}encoder.transformer_encoder.norm.bias"
    if last_norm_bias_key not in state_dict_in:
        raise ValueError(
            f"Did not find the last layer norm bias in the state_dict. "
            f"Expected key: {last_norm_bias_key}"
        )
    hidden_size = state_dict_in[last_norm_bias_key].shape[0]

    # Build a new classifier with dimension = 2 * hidden_size
    # (Assumes use_dinov2_classifier=True in the model config,
    #  which can concatenate CLS+avg_pool features.)
    classifier_weight = torch.empty((num_classes, 2 * hidden_size))
    torch.nn.init.normal_(classifier_weight, mean=0.0, std=0.02)
    state_dict_out["classifier.classifier.ffn.0.linear_layer.weight"] = (
        classifier_weight
    )

    if use_bias_in_output:
        classifier_bias = torch.zeros(num_classes)
        state_dict_out["classifier.classifier.ffn.0.linear_layer.bias"] = (
            classifier_bias
        )

    return state_dict_out


###############################################################################
#                 CONFIG CONVERSION (DINOv2 -> ViTClassification)            #
###############################################################################
MODEL_CONFIG_FIELDS_TO_SKIP = [
    # model class name changes from MultiImageViTModel to vision_transformer
    "name",
    # dropouts are not needed
    "attention_dropout_rate",
    "dropout_rate",
    "embedding_dropout_rate",
    "stochastic_depth_drop_prob",
    "stochastic_depth_drop_prob_schedule",
    "stochastic_depth_mode",
    # initializers are not needed
    "attention_initializer",
    "initializer_range",
    "position_embedding_initializer",
    "projection_initializer",
    "cls_token_initializer",
    "default_initializer",
    "ffn_initializer",
    "pooler_initializer",
    # embedding settings for training are not needed
    "interpolate_position_embedding",
    "use_masked_patches",
    # not needed for ViTClassification
    "image_layer_idx",
    # stale fields to be removed from model config
    "mixed_precision",
    "fp16_type",
]


def convert_dinov2_teacher_config_to_vit_classification_config(
    dino_cfg: Dict[str, Any],
    dataset_path: str,
    use_bias_in_output: bool = True,
    num_classes: int = 1000,
) -> Dict[str, Any]:
    """
    Convert the teacher trunk (index 0) of a DINOv2 config to a minimal
    ViTClassification config. Training-related fields (dropouts, initializers, etc.)
    are omitted. Defaults for optimizer and data loaders are provided.

    Args:
        dino_cfg (Dict[str, Any]): The input DINOv2 config.
        use_bias_in_output (bool): Whether to include bias in the final classifier.
        num_classes (int): The output dimension of the classifier.

    Returns:
        Dict[str, Any]: A new config suitable for ViTClassification.
    """
    # Locate the teacher trunk in the DINOv2 config
    try:
        teacher_trunk = dino_cfg["trainer"]["init"]["model"][
            "image_model_trunks"
        ][0]
    except (KeyError, IndexError) as e:
        raise ValueError(
            "Could not find the teacher trunk at index 0 under "
            "'trainer.init.model.image_model_trunks'."
        ) from e

    # Validate the teacher trunk config
    if not teacher_trunk.get("stop_grad", False):
        raise ValueError(
            "Teacher trunk must have 'stop_grad: true' at index 0."
        )

    teacher_cfg = teacher_trunk.get("image_model")
    if teacher_cfg is None:
        raise ValueError("Missing 'image_model' field in teacher trunk config.")

    if teacher_cfg.get("name") != "MultiImageViTModel":
        raise ValueError(
            "Expected 'name: MultiImageViTModel' in teacher trunk; got "
            f"{teacher_cfg.get('name')}"
        )

    # Build minimal ViTClassification config from the teacher config
    model_cfg: Dict[str, Any] = {
        "name": "vision_transformer",
        "num_classes": num_classes,
        "compute_eval_metrics": True,
        "use_bias_in_output": use_bias_in_output,
        "use_dinov2_classifier": True,  # For teacher+student feature concatenation
        "freeze": ["^vit_model"],  # Freeze all vit_model layers
    }

    # Copy over relevant fields, skipping any that are not needed
    for field, value in teacher_cfg.items():
        if field not in MODEL_CONFIG_FIELDS_TO_SKIP:
            model_cfg[field] = value

    # Construct a new default config around the minimal model cfg
    new_cfg = {
        "trainer": {
            "init": {
                "backend": dino_cfg["trainer"]["init"].get("backend"),
                "model": model_cfg,
                "optimizer": {
                    "SGD": {
                        "momentum": 0.9,
                        "weight_decay": 0.0,
                    }
                },
                "schedulers": [
                    {
                        "SequentialLR": {
                            "schedulers": [
                                {
                                    "CosineAnnealingLR": {
                                        "initial_learning_rate": 0.025,
                                        "eta_min": 0.0,
                                        "T_max": 12500,
                                    }
                                }
                            ]
                        }
                    }
                ],
                "precision": {
                    "fp16_type": "bfloat16",
                },
                "loop": {
                    "max_steps": 12500,
                    "eval_frequency": 12500,
                },
                "checkpoint": {
                    "steps": 12500,
                },
                "logging": {
                    "log_steps": 1,
                },
            },
            "fit": {
                "train_dataloader": {
                    "data_processor": "ImageNet1KProcessor",
                    "data_dir": dataset_path,
                    "batch_size": 128,
                    "image_size": [224, 224],
                    "shuffle": True,
                    "shuffle_seed": 42,
                    "split": "train",
                    "transforms": [
                        {
                            "name": "random_resized_crop",
                            "size": [224, 224],
                            "scale": [0.08, 1.0],
                            "ratio": [0.75, 1.33],
                            "interpolation": "bicubic",
                        },
                        {"name": "random_horizontal_flip", "p": 0.5},
                        {"name": "to_tensor"},
                        {
                            "name": "normalize",
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    ],
                    "num_workers": 8,
                    "prefetch_factor": 2,
                    "persistent_workers": True,
                    "use_worker_cache": True,
                },
                "val_dataloader": {
                    "data_processor": "ImageNet1KProcessor",
                    "data_dir": dataset_path,
                    "batch_size": 128,
                    "image_size": [224, 224],
                    "split": "val",
                    "transforms": [
                        {
                            "name": "resize",
                            "size": [256, 256],
                            "interpolation": "bicubic",
                        },
                        {"name": "center_crop", "size": [224, 224]},
                        {"name": "to_tensor"},
                        {
                            "name": "normalize",
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    ],
                    "num_workers": 8,
                    "prefetch_factor": 2,
                    "persistent_workers": True,
                    "use_worker_cache": True,
                },
            },
        }
    }

    old_precision = dino_cfg["trainer"]["init"].get("precision")
    if old_precision is not None:
        new_cfg["trainer"]["init"]["precision"] = old_precision

    return new_cfg


###############################################################################
#                           MAIN FUNCTION                                     #
###############################################################################
def main() -> None:
    """
    CLI for converting a DINOv2 (teacher) checkpoint and config to ViTClassification
    formats. Will save a new .mdl file (checkpoint) and/or .yaml file (config)
    as requested by command-line arguments.
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=(
            "CLI for Cerebras DINOv2 -> ViTClassification conversion "
            "(both checkpoint and config)."
        )
    )
    parser.add_argument(
        "--input_ckpt",
        type=str,
        required=False,
        help="Path to the input DINOv2 checkpoint (.mdl).",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        required=False,
        help="Path to the output ViTClassification checkpoint (.mdl).",
    )
    parser.add_argument(
        "--input_config",
        type=str,
        required=False,
        help=(
            "Path to the input DINOv2 config (.yaml). If provided, the script "
            "will also convert and save a new config."
        ),
    )
    parser.add_argument(
        "--output_config",
        type=str,
        required=False,
        help=(
            "Path to the output ViTClassification config (.yaml). Required if "
            "--input_config is provided."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory for the new config.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=False,
        default=1000,
        help="Number of classes in classification.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Convert checkpoint if requested
    # -------------------------------------------------------------------------
    if args.input_ckpt:
        if not args.output_ckpt:
            raise ValueError(
                "--output_ckpt must be provided when --input_ckpt is specified."
            )
        # Load checkpoint
        old_sd = cstorch.load(args.input_ckpt)

        # If the checkpoint is wrapped with a "model" key, use that
        if "model" in old_sd:
            old_sd = old_sd["model"]

        # Convert and save
        new_sd = convert_dinov2_to_vit(old_sd, num_classes=args.num_classes)
        final_out = {"model": new_sd}
        cstorch.save(final_out, args.output_ckpt)
        logger.info(f"Saved converted checkpoint to {args.output_ckpt}")

    # -------------------------------------------------------------------------
    # 2. Convert config if requested
    # -------------------------------------------------------------------------
    if args.input_config:
        if not args.output_config:
            raise ValueError(
                "--output_config must be provided when --input_config is specified."
            )
        with open(args.input_config, "r") as f:
            dino_cfg = yaml.safe_load(f)

        new_config = convert_dinov2_teacher_config_to_vit_classification_config(
            dino_cfg,
            args.dataset_path,
            num_classes=args.num_classes,
        )

        with open(args.output_config, "w") as f:
            f.write("# This file is generated by convert_dinov2_to_vit.py\n")
            f.write("# --input_config: {}\n".format(args.input_config))
            yaml.safe_dump(new_config, f, sort_keys=False)

        logger.info(f"Saved converted config to {args.output_config}")

    logger.info("Conversion completed.")


if __name__ == "__main__":
    main()
