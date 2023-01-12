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

import pickle

import numpy as np


def extract_mask_from_weight(sparse_wgt: np.ndarray, sparse_val: float):
    """
    Given a sparsified weight in a dense format, return the mask used
    to generate said weight as a dense tensor. Sparsified weights are
    assumed to be a float type, otherwise NaN sparse_vals are not well
    defined.
    """
    mask = np.ones_like(sparse_wgt)  # Assume everything is dense
    if np.isnan(sparse_val):
        sparse_entries = np.isnan(sparse_wgt)
    else:
        sparse_entries = sparse_wgt == sparse_val
    if np.any(
        sparse_entries
    ):  # Special case to exclude dense non-floats (ex. global_step)
        mask[sparse_entries] = sparse_val
    return mask


def should_sparsify_weight(weight_name, weight):
    """Function to handle which weights to sparsify. Currently, we do not
    sparsify the following parameters: scalars, biases in weights, norm weights
    and emebdding weights

    Args:
        weight_name (str): Name of the weights to sparsify
        weight (numpy.ndarray): Weight to sparsify

    Returns:
        A boolean indicating whether the weight should be sparsified or not
    """
    # handle scalars and biases, respectively
    if not weight.shape or len(weight.shape) <= 1:
        return False

    # handle embeddings
    if "embedding" in weight_name.lower():
        return False

    # handle norm weights. the check for biases should handle this, but adding
    # it for ensuring coverage.
    if "norm" in weight_name.lower():
        return False

    return True


def erdos_renyi_distribution(
    params, names, sparsity, erk_power_scale=1.0, is_kernel=True
):
    """
    Get Erdos-Renyi [Kernel] based distribution for weights based on parameter
    dimensions from `"Rigging the Lottery: Making All Tickets Winners"
    <https://arxiv.org/abs/1911.11134>`_.

    We will start with all layers and try to find right epsilon. However if
    any probablity exceeds 1, we will make that layer dense and repeat the
    process (finding epsilon) with the non-dense layers.

    We want the total number of connections to be the same. Let say we have
    four layers with N_1, N_2, N_3, N_4 parameters each. Lets say after some
    iterations, probability of some dense layers (3, 4) exceeded 1 and
    therefore we added them to the dense_layers set. Those layers will not
    scale with erdos_renyi, however we need to count them so that target
    paratemeter count is achieved. Hence, we solve for this:
    --------------------------------------------------------------------------
    eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
    --------------------------------------------------------------------------
    eps * (p_1 * N_1 + p_2 * N_2) =
    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
    --------------------------------------------------------------------------
    eps = rhs / ([for all_i] sum_i p_i * N_i) = rhs / divisor.

    Args:
        params (iterable): iterable of parameters to sparsify
        names (iterable): iterable of names to sparsify
        sparsity (float): Sparsity value to use for weights
        erk_power_scale (float): Gamma factor for ERK distribution, 0.0 corresponds
            to `uniform` distribution and 1.0 corresponds to `erk` distribution,
            defaults to 1.0
        is_kernel (bool): Specifies to use ERK distirbution over ER distribution
            from the Sparse Evolutionary Training (SET) paper, defaults to `True`
    Returns:
        A dictionary `sparsity_level_dict` containing a mapping of names to
        sparsities per layer
    """
    is_epsilon_valid = False
    dense_layers = set()
    # density of network
    density = 1.0 - sparsity

    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for n, p in zip(names, params):
            n_param = p.size
            n_zeros = n_param * sparsity
            n_ones = n_param * density

            if n in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                if is_kernel:
                    raw_probabilities[n] = (
                        sum(p.shape) / n_param
                    ) ** erk_power_scale
                else:
                    n_in, n_out = p.shape[:2]
                    raw_probabilities[n] = (n_in + n_out) / (n_in * n_out)

                divisor += raw_probabilities[n] * n_param

        epsilon = rhs / divisor
        max_prob = max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for p_id, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    dense_layers.add(p_id)
        else:
            is_epsilon_valid = True

    sparsity_level_dict = {}
    for n in names:
        if n not in dense_layers:
            p_density = epsilon * raw_probabilities[n]
            sparsity_level_dict[n] = 1.0 - p_density
        else:
            sparsity_level_dict[n] = 0.0

    return sparsity_level_dict


def uniform_distribution(names, sparsity):
    """
    Get Uniform (all layers get the same sparsity) distribution for weights
    based on parameter dimensions.

    Args:
        names (iterable): iterable of names to sparsify
        sparsity (float): Sparsity value to use for weights
    Returns:
        A dictionary `sparsity_level_dict` containing a mapping of names to
        sparsities per layer
    """
    sparsity_level_dict = {}
    for n in names:
        sparsity_level_dict[n] = sparsity

    return sparsity_level_dict


def extract_mask_from_file(mask_file: str, sparse_val: float):
    """
    Given a file from which masks can be extracted for an entire
    model, load the file and extract masks. Sparsified weights, if
    present in the file, are assumed to be a float type, otherwise
    NaN sparse_vals are not well defined
    """
    from modelzoo.common.tf.run_utils import get_weight_dict

    mask_dict = {}

    if mask_file.endswith(".pkl"):
        with open(mask_file, 'rb') as f_saved_masks:
            mask_dict = pickle.load(f_saved_masks)
    elif mask_file.endswith('.npz'):
        mask_dict = dict(np.load(mask_file, allow_pickle=True))
        if len(mask_dict.keys()) == 1:  # need to unwrap
            _, mask_dict = mask_dict.popitem()
            mask_dict = mask_dict.item()
    else:  # checkpoint file
        wgts = get_weight_dict(mask_file)
        mask_dict = {}
        for wgt_name, sparse_wgt in wgts.items():
            if sparse_val is not None:
                inferred_sparse_val = sparse_val
            else:
                inferred_sparse_val = (
                    float('nan') if np.any(np.isnan(sparse_wgt)) else 0
                )
            mask_dict[wgt_name] = extract_mask_from_weight(
                sparse_wgt, inferred_sparse_val
            )

    return mask_dict


__all__ = [
    "extract_mask_from_weight",
    "extract_mask_from_file",
    "should_sparsify_weight",
    "erdos_renyi_distribution",
    "uniform_distribution",
]
