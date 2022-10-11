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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Because GPU is not needed for this
import argparse
import json
import sys

import numpy as np
import tensorflow as tf
import torch
from transformers import GPTJForCausalLM


def get_runtime_args():
    """Create parser for command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Directory containing HuggingFace checkpoint. If it does not exist,"
            " this path is used to store the downloaded checkpoint."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Model directory where converted checkpoints and states will be"
            " written. If it does not exist, it is created during runtime."
        ),
    )
    parser.add_argument(
        "--share_embeddings",
        default=False,
        action='store_true',
        help=(
            "Remove the lm_head weights and creates a checkpoint with shared (aka. tied) embeddings."
            " Note: This will make the TF checkpoint differ from the original JAX model."
        ),
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help=("Whether to check model call from HuggingFace."),
    )
    return parser.parse_args(sys.argv[1:])


def create_tf_var(tensor: np.ndarray, name: str, session: tf.compat.v1.Session):
    """Takes a tensor and creates a TensorFlow Variable with same shape and
    dtype, initialized with the given tensor.

    Args:
        tensor (ndarray): Tensor to model as zeros
        name (str): Name to give the variable
        session (tf.Session): Session to create the variables

    Returns:
        A zero initialized TensorFlow variable
    """
    tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
    tf_var = tf.compat.v1.get_variable(
        dtype=tf_dtype,
        shape=tensor.shape,
        name=name,
        initializer=tf.compat.v1.zeros_initializer(),
    )
    session.run(tf.compat.v1.variables_initializer([tf_var]))
    session.run(tf_var)
    return tf_var


def dict_to_checkpoint(state_dict: dict, checkpoint_name: str):
    """Convert a dictionary of weights to a TF checkpoint to be used by the
    model during execution.

    Args:
        state_dict (dict): Dictionary containing weight names mapped to pytorch
        tensors from the HuggingFace checkpoint
        checkpoint_name (str): Path to save the TensorFlow checkpoint
    """
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
        for name, pt_tensor in state_dict.items():
            np_tensor = pt_tensor.numpy()
            if name:
                tf_var = create_tf_var(
                    tensor=np_tensor, name=name, session=session,
                )
                tf.keras.backend.set_value(tf_var, np_tensor)
                tf_weight = session.run(tf_var)

        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
        saver.save(session, checkpoint_name)


def save_mappings(tf_ckpt_path: str, variable_dict: dict, transpose_dict: dict):
    """Save the mapping of variables between PyTorch and TensorFlow.

    Args:
        tf_ckpt_path (str): Path for storing checkpoints and associated mappings
        variable_dict (dict): A mapping of variables between PyTorch and
            TensorFlow checkpoints names.
        transpose_dict (dict): A mapping of whether variables between PyTorch
            and TensorFlow checkpoints are transposed.
    """
    with open(os.path.join(tf_ckpt_path, 'keys.json'), 'w') as fout:
        json.dump(variable_dict, fout, sort_keys=False)

    with open(os.path.join(tf_ckpt_path, 'shapes.json'), 'w') as fout:
        json.dump(transpose_dict, fout, sort_keys=False)


def map_embeddings_pt_to_tf(key: str):
    """Map embedding layer to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    transpose = False
    dict_map = {"transformer.wte.weight": "input_embedding/embedding_weights"}
    return dict_map[key], transpose


def map_outputs_pt_to_tf(key: str, shared_embeddings: bool = True):
    """Map final layer norm and output head bias to weights. Output weights for
    sharedweight layer comes from emebdding layer mapping.

    Args:
        key (str): key to map from PyTorch to TensorFlow
        shared_embeddings (bool): whether the model is trained with shared
            embeddings. Defaults to `True`

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    transpose = False
    if key == "lm_head.weight":
        transpose = True

    dict_map = {
        "transformer.ln_f.weight": "post_decoder_layer_norm/post_decoder_layer_norm/gamma",
        "transformer.ln_f.bias": "post_decoder_layer_norm/post_decoder_layer_norm/beta",
    }

    embeddings_map = {}
    if shared_embeddings:
        embeddings_map = {
            "lm_head.bias": "bias",
        }
    else:
        embeddings_map = {
            "lm_head.weight": "lm_head/lm_head/kernel",
            "lm_head.bias": "lm_head/lm_head/bias",
        }
    dict_map.update(embeddings_map)

    return dict_map[key], transpose


def map_ln_pt_to_tf(key: str):
    """Map LayerNorm Layer in decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """
    transpose = False
    split_key = key.split('.')
    pt_layer_num = int(split_key[2])

    final_key_dict = {
        "bias": "beta",
        "weight": "gamma",
    }
    final_key = final_key_dict[split_key[-1]]

    tf_key = f"gptj_decoder/{pt_layer_num}/ln_1/ln_1/{final_key}"
    return tf_key, transpose


def map_attn_pt_to_tf(key: str):
    """Map Attention Block in decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """
    transpose = False
    if "weight" in key:
        transpose = True

    split_key = key.split('.')
    pt_layer_num = int(split_key[2])

    last_two = ".".join([split_key[-2], split_key[-1]])
    if last_two in ["attn.bias", "attn.masked_bias"]:
        # Note: "attn.bias" is a 2028x2048 autoregressive mask,
        # and "attn.masked_bias" is a large negative number (-1.0e+09).
        # Neither need to be saved in the checkpoint. They are generated when they are needed.
        return None, False

    dict_map = {
        f"transformer.h.{pt_layer_num}": f"gptj_decoder/{pt_layer_num}",
        "attn": "attn",
        "q_proj": "q_proj/q_proj",
        "k_proj": "k_proj/k_proj",
        "v_proj": "v_proj/v_proj",
        "weight": "kernel",
        "bias": "bias",
        "out_proj": "out_proj/out_proj",
    }

    for k in dict_map.keys():
        if k in key:
            key = key.replace(k, dict_map[k])

    tf_key = key.replace(".", "/")
    return tf_key, transpose


def map_mlp_pt_to_tf(key: str):
    """Map MLP Block in decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    transpose = False
    if "weight" in key:
        transpose = True

    split_key = key.split('.')
    pt_layer_num = int(split_key[2])

    final_key_dict = {"bias": "bias", "weight": "kernel"}
    final_key = final_key_dict[split_key[-1]]

    if "fc_in" in key:
        tf_layer_num = int(pt_layer_num * 2)
    elif "fc_out" in key:
        tf_layer_num = int(pt_layer_num * 2 + 1)

    tf_key = f"gptj_decoder/{pt_layer_num}/mlp/dense_layer_{tf_layer_num}/dense_layer_{tf_layer_num}/{final_key}"
    tf_key = tf_key.replace("_0", "")
    return tf_key, transpose


def map_decoder_pt_to_tf(key: str):
    """Map decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    if "ln_1" in key:
        output_ = map_ln_pt_to_tf(key)
    elif "attn" in key:
        output_ = map_attn_pt_to_tf(key)
    elif "mlp" in key:
        output_ = map_mlp_pt_to_tf(key)
    else:
        raise ValueError(
            f"expected key to have one of ln_1, attn, mlp substrings"
            f" got {key} instead. Check that the model definition is correct!!"
        )

    return output_


def convert_pt_checkpoint_to_tf(
    input_dir: str, output_dir: str, share_embeddings: bool, debug: bool,
):
    """Main function to convert PyTorch weights to TensorFlow.

    Args:
        pt_ckpt_path (str): Path to PyTorch checkpoint
        tf_ckpt_path (str): Path to TensorFlow checkpoint directory
        share_embeddings (bool): Specifies whether to share embeddings for
            checkpoint conversion.
        debug (bool): Enable debug for model creation
    """
    # Create PT model
    pt_checkpoint_path = os.path.join(input_dir, "pytorch_model.bin")
    if not os.path.exists(pt_checkpoint_path):
        print(
            f"{pt_checkpoint_path} does not exist, downloading checkpoint from HuggingFace, or loading from HugginFace cache."
        )
        pt_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        print(f"Model loaded. Saving checkpoint at {input_dir}")
        pt_model.save_pretrained(input_dir, max_shard_size="30GB")
        print("Checkpoint saved")
    else:
        pt_model = GPTJForCausalLM.from_pretrained(input_dir)
    print("Created HuggingFace model!!")

    if debug:
        # print number of parameters for debugging
        model_parameters = filter(
            lambda p: p.requires_grad, pt_model.parameters()
        )
        num_params = sum([p.numel() for p in model_parameters])
        print(f"Initialized model with param count: {num_params}")

    pt_weight_dict = pt_model.state_dict()
    weight_keys = sorted(list(pt_weight_dict.keys()))

    state_dict_update = {}
    keys_mapping = {}
    shape_mapping = {}

    # perform mapping
    for key in weight_keys:
        if "transformer.wte" in key:
            tf_name, transpose = map_embeddings_pt_to_tf(key)
        elif "transformer.h" in key:
            tf_name, transpose = map_decoder_pt_to_tf(key)
        elif "transformer.ln_f" or "lm_head" in key:
            if share_embeddings and key == "lm_head.weight":
                print(
                    f"{key} not needed when embeddings are shared with"
                    f" classifier. Continuing mapping for keys."
                )
                continue
            tf_name, transpose = map_outputs_pt_to_tf(key, share_embeddings)

        if tf_name is None:
            if "attn.bias" or "attn.masked_bias" in key:
                continue

            raise ValueError(
                f"{key} mapped as None. Ignore if this is desired behavior."
                f" Else exercise the checkpoint saving with caution!!"
            )

        try:
            val = pt_weight_dict[key]
            update_val = val.T if transpose else val
            update_val = torch.Tensor(update_val)
            # if valid tensor, insert to mapping
            state_dict_update[tf_name] = update_val
            keys_mapping[key] = tf_name
            shape_mapping[key] = transpose
        except TypeError:
            print(
                f"tried to map {key} to {tf_name}, but got wrong TensorType"
                f". Ignoring mapping for now, to be fixed later !!"
            )

    # save mappings for verification
    save_mappings(output_dir, keys_mapping, shape_mapping)

    # convert finally dictionary to checkpoints
    tf_ckpt_path = os.path.join(output_dir, "tf_model.ckpt")
    dict_to_checkpoint(state_dict_update, tf_ckpt_path)


def validate_args(args):
    """Validate the user specified arguments.

    Args:
        args (namespace): Argparse arguments
    """
    if not os.path.isdir(args.output_dir):
        print(
            "Output directory does not exist. Creating it for runtime execution."
        )
        os.makedirs(args.output_dir, exist_ok=False)


def main():
    args = get_runtime_args()
    validate_args(args)
    # Convert to TF checkpoint
    convert_pt_checkpoint_to_tf(
        args.input_dir, args.output_dir, args.share_embeddings, args.debug,
    )
    print("Converted HuggingFace checkpoint successfully!!")


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    main()
