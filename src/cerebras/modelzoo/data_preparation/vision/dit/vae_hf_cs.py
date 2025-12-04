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

# isort: off
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# isort: on

"""
command:
python modelzoo/tools/checkpoint_converters/internal/vae_hf_cs.py --dest_ckpt_path=<path to converted checkpoint>
"""
import argparse
import logging
import os
from typing import Tuple

import cerebras.pytorch as cstorch

LOGFORMAT = '%(asctime)s %(levelname)-4s[%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGFORMAT)

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter,
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
)


def get_parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--src_ckpt_path",
        type=str,
        required=False,
        default=None,
        help=f"Path to HF Pretrained VAE checkpoint .bin file. "
        f"If not provided, file is automatically downloaded from "
        f"https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",
    )
    parser.add_argument(
        "--dest_ckpt_path",
        type=str,
        required=False,
        default=os.path.join(
            os.path.dirname(__file__), "mz_stabilityai-sd-vae-ft-mse_ckpt.bin"
        ),
        help="Path to converted modelzoo compatible checkpoint",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=True,
        help="Path to VAE model params yaml",
    )
    args = parser.parse_args()
    return args


class Converter_VAEModel_HF_CS19(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [  # same keys
                    EquivalentSubkey("encoder.conv", "encoder.conv"),
                    ".*\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # same keys
                    EquivalentSubkey(
                        "encoder.down_blocks", "encoder.down_blocks"
                    ),
                    ".*",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # same keys
                    EquivalentSubkey(
                        "encoder.mid_block.resnets", "encoder.mid_block.resnets"
                    ),
                    ".*(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # encoder.mid_block.attentions.0.group_norm.weight -> encoder.mid_block.norms.0.weight
                    EquivalentSubkey(
                        "encoder.mid_block.attentions",
                        "encoder.mid_block.norms",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("group_norm.", ""),
                    ".*(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # encoder.mid_block.attentions.0.query.weight -> encoder.mid_block.attentions.0.proj_q_dense_layer.weight
                    EquivalentSubkey(
                        "encoder.mid_block.attentions",
                        "encoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("query", "proj_q_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # encoder.mid_block.attentions.0.key.weight -> encoder.mid_block.attentions.0.proj_k_dense_layer.weight
                    EquivalentSubkey(
                        "encoder.mid_block.attentions",
                        "encoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("key", "proj_k_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # encoder.mid_block.attentions.0.value.weight -> encoder.mid_block.attentions.0.proj_v_dense_layer.weight
                    EquivalentSubkey(
                        "encoder.mid_block.attentions",
                        "encoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("value", "proj_v_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # encoder.mid_block.attentions.0.proj_attn.weight -> encoder.mid_block.attentions.0.proj_output_dense_layer.weight
                    EquivalentSubkey(
                        "encoder.mid_block.attentions",
                        "encoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("proj_attn", "proj_output_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # same keys
                    EquivalentSubkey("decoder.conv", "decoder.conv"),
                    ".*\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # same keys
                    EquivalentSubkey("decoder.up_blocks", "decoder.up_blocks"),
                    ".*(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # same keys
                    EquivalentSubkey(
                        "decoder.mid_block.resnets", "decoder.mid_block.resnets"
                    ),
                    ".*(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # decoder.mid_block.attentions.0.group_norm.weight -> decoder.mid_block.norms.0.weight
                    EquivalentSubkey(
                        "decoder.mid_block.attentions",
                        "decoder.mid_block.norms",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("group_norm.", ""),
                    ".*(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # decoder.mid_block.attentions.0.query.weight -> decoder.mid_block.attentions.0.proj_q_dense_layer.weight
                    EquivalentSubkey(
                        "decoder.mid_block.attentions",
                        "decoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("query", "proj_q_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # decoder.mid_block.attentions.0.key.weight -> decoder.mid_block.attentions.0.proj_k_dense_layer.weight
                    EquivalentSubkey(
                        "decoder.mid_block.attentions",
                        "decoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("key", "proj_k_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # decoder.mid_block.attentions.0.value.weight -> decoder.mid_block.attentions.0.proj_v_dense_layer.weight
                    EquivalentSubkey(
                        "decoder.mid_block.attentions",
                        "decoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("value", "proj_v_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # decoder.mid_block.attentions.0.proj_attn.weight -> decoder.mid_block.attentions.0.proj_output_dense_layer.weight
                    EquivalentSubkey(
                        "decoder.mid_block.attentions",
                        "decoder.mid_block.attentions",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("proj_attn", "proj_output_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # same keys
                    EquivalentSubkey("quant_conv", "quant_conv"),
                    ".*(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
            ConversionRule(
                [  # same keys
                    EquivalentSubkey("post_quant_conv", "post_quant_conv"),
                    ".*(?:weight|bias)",
                ],
                action=BaseCheckpointConverter.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("vae_HF", "cs-1.9")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None


if __name__ == "__main__":
    import yaml

    from cerebras.modelzoo.models.vision.dit.layers.vae.VAEModel import (
        AutoencoderKL as CSAutoencoderKL,
    )

    args = get_parser_args()

    if args.src_ckpt_path is None:
        import requests

        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"
        logging.info(
            f"No `src_ckpt_path` provided, downloading the model from {url}"
        )
        response = requests.get(url)
        response.raise_for_status()

        args.src_ckpt_path = os.path.join(
            os.path.dirname(__file__), "hf_stabilityai-sd-vae-ft-mse_ckpt.bin"
        )
        with open(args.src_ckpt_path, "wb") as fh:
            fh.write(response.content)
        logging.info(
            f"Downloaded source pretrained ckpt at {args.src_ckpt_path}"
        )

    old_state_dict = cstorch.load(args.src_ckpt_path)

    # VAE Params for CS modelzoo
    with open(args.params_path, "r") as fh:
        vae_params = yaml.safe_load(fh)["model"]["vae"]

    # Initialize CS VAE model
    cs_vae = CSAutoencoderKL(**vae_params)
    new_state_dict = cs_vae.state_dict()

    logging.info(f"Converting checkpoint...")
    # Convert
    converter = Converter_VAEModel_HF_CS19()
    matched_all_keys = converter.convert_all_keys(
        old_state_dict=old_state_dict,
        new_state_dict=new_state_dict,
        from_index=0,
    )
    logging.info(f"matched_all_keys:{matched_all_keys}")
    cstorch.save(
        new_state_dict,
        args.dest_ckpt_path,
    )
    logging.info(f"DONE: Converting checkpoint, saved at {args.dest_ckpt_path}")
