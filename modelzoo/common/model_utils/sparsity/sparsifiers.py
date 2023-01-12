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

"""Module for sparsifiers in weight streaming"""
import abc
import logging
import math
import os
import re
from enum import Enum
from functools import reduce

import numpy as np
import scipy
import scipy.sparse

from modelzoo.common.model_utils.sparsity.utils import (
    erdos_renyi_distribution,
    extract_mask_from_file,
    should_sparsify_weight,
    uniform_distribution,
)

LOW_SPARSITY_THRESHOLD = 0.5


def get_sparsity_level_dict(
    weights_dict,
    sparsity_level,
    sparsity_distribution='uniform',
    erk_power_scale=1.0,
):
    """
    Get a per layer mapping of sparsity levels for input to checkpoint manipulation.

    Args:
        weights_dict (dict): A dictionary mapping for weight names to parameters
        sparsity_level (float): Sparsity value to use for weights
        sparsity_distribution (str): Sparsity distribution to use for getting
            per layer sparsity, defaults to `uniform`
        erk_power_scale (float): Gamma factor for ERK distribution, 0.0 corresponds
            to `uniform` distribution and 1.0 corresponds to `erk` distribution,
            defaults to 1.0
    Returns:
        A dictionary `sparsity_level_dict` containing a mapping of names to
            sparsities per layer
    """
    params = []
    names = []

    for v_name, v in weights_dict.items():
        if should_sparsify_weight(v_name, v):
            params.append(v)
            names.append(v_name)

    if sparsity_distribution in ['er', 'erk']:
        is_kernel = sparsity_distribution == 'erk'
        sparsity_level_dict = erdos_renyi_distribution(
            params,
            names,
            sparsity_level,
            erk_power_scale=erk_power_scale,
            is_kernel=is_kernel,
        )
    elif sparsity_distribution == 'uniform':
        sparsity_level_dict = uniform_distribution(names, sparsity_level)
    else:
        raise ValueError(
            f'Expected one of Uniform, ER, ERK for sparsity distribution'
            f' got {sparsity_distribution}'
        )

    return sparsity_level_dict


class BaseSparsifier(abc.ABC):
    """ This is an abstract class that can be passed to a training object
    to trigger sparsity. We use a class instead of a simple callback function
    to allow storage of specific state information

    Any particular Sparsifier should use this as a base class and implement
    the following functions.
    """

    @abc.abstractmethod
    def apply_sparsity(self, step):
        pass

    @abc.abstractmethod
    def get_masked_weights(self, step, weights_dict, sparse_val):
        pass

    @abc.abstractmethod
    def get_num_sparsified_values(self, v_name, sparse_val):
        pass


class ConstantMaskSparsifier(BaseSparsifier):
    """Apply a fixed mask to enduce sparsity"""

    # Given we don't always use the same type of mask, this is here for
    # auditing which kind of mask is used.
    class MaskType(Enum):
        """ Sparsity type
        """

        DENSE = 1
        HIGH_SPARSITY = 2
        LOW_SPARSITY = 3

    def __init__(
        self,
        n_iter,
        sparsity_level=0.5,
        sparsity_distribution='uniform',
        erk_power_scale=1.0,
        epsilon=None,
        zeta=None,
        seed=None,
        mask_file=None,
    ):
        self.n_iter = n_iter
        assert isinstance(sparsity_level, float)
        self.sparsity_level = sparsity_level
        self.sparsity_distribution = sparsity_distribution
        self.erk_power_scale = erk_power_scale
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        # this will be populated first time we generate masks
        self.mask_dict = {}

    def apply_sparsity(self, step):
        if step % self.n_iter == 0:
            return True
        return False

    def get_masked_weights(self, step, weights_dict, sparse_val):
        """ We want to use scipy sparse tensors to generate the masked weights
        in a more efficient manner. However, there are a few caveats:
            (1): scipy sparse matricies only work for 2d matrix. For arbitrary
                 dimensioned tensors, need to rely on np.dense

            (2): High sparsity: In this case, we want to generate a mask where
                 the set values correspond to the set values in the actual
                 weight. The issue with this appoach is once we do a sparse-
                 dense multiply/and, we are left with 0s and non-zeros. We then
                 need to (a) distinguish between 0s that came from the mask and
                 0s that were originally part of the weight and (b) make the 0s
                 correspoinding to the mask set to NaN. This involves some
                 np.where magic.

            (2): High sparsity: In this case, we want to generate a mask where
                 the set values correspond to the unset/nz values in the actual
                 weight. The issue with this appoach is if the sparse_val is 0
                 (and not 'nan'), then there is no nice efficient way to mask
                 using scipy sparse tensors. In scipy sparse tensors, the sparse
                 value is always 0, so 0 can't be used as the set value in the
                 mask. We would have to make the set value -1, and then
                 subtract it from a an all ones tensor to get a usable mask.
                 As this doesn't provide any more savings than using the above
                 approach, for sparse val 0, we will use the high sparsity
                 approach even at low sparsity.
                 Afaik, this is the only usage of 0 as sparse val - so this
                 performance hit should not be a huge issue as it's localized
                 to testing.

        In the dense case, we simply create a tensor of the same shape filled
        with 1s and sparse_values as the mask. We then multiply this with our
        weight tensor to get the value.

        In the sparse case, we will create a sparse tensor with 'nan' as the
        non-sparse element, and do sparse-dense element-wise addition with the
        weight tensor to get the masked value. If the masking val is 0, we will
        first create a sparse matrix with 1 as the set value, then flip the
        values (subtract a all ones tensor) and sparse-dense multiply this
        with the weight to get the masked weight.
        """

        def get_mask_dense(shape, sparsity_level):
            mask = np.random.choice(
                [1.0, sparse_val],
                shape,
                p=[1 - sparsity_level, sparsity_level],
            )
            return mask

        def apply_mask_dense(mask, weight):
            return weight * mask.astype(weight.dtype)

        def get_mask_low_sparsity(shape, sparsity_level):
            def data_rvs(k):
                return np.nan * np.ones(k)

            assert math.isnan(sparse_val)
            assert len(shape) == 2
            mask = scipy.sparse.random(
                shape[0], shape[1], density=sparsity_level, data_rvs=data_rvs,
            )
            return mask

        def apply_mask_low_sparsity(mask, weight):
            output_dtype = weight.dtype
            assert mask.shape == weight.shape
            if not math.isnan(sparse_val):
                # pylint: disable=protected-access
                mask = mask._add_dense(np.ones_like(mask))
            # pylint: disable=protected-access
            weight = mask._add_dense(weight).A
            return weight.astype(output_dtype)

        def get_mask_high_sparsity(shape, sparsity_level):
            def data_rvs(k):
                return np.ones(k)

            assert len(shape) == 2
            mask = scipy.sparse.random(
                shape[0],
                shape[1],
                density=(1 - sparsity_level),
                data_rvs=data_rvs,
            )
            return mask

        def apply_mask_high_sparsity(mask, weight):
            """ We need to be able to distinguish between 0s that are part of
            the weight (and haven't been sparsed out) vs the sparse values.
            Both will be 0, before we do the nan conversion process. Using
            the sparse mask in np.where will involve us expanding it and thus
            defeating it's purpose.

            So we will first set all 0s in the weight as inf as a way of
            marking them, and then at the end set them to 0 again.
            """
            output_dtype = weight.dtype
            assert mask.shape == weight.shape
            weight = np.where(weight == 0, np.inf, weight)
            weight = mask.multiply(weight).A
            if sparse_val != 0:
                weight = np.where(weight == 0, sparse_val, weight)
            weight = np.where(weight == np.inf, 0, weight)
            return weight.astype(output_dtype)

        assert math.isnan(sparse_val) or sparse_val == 0
        sparse_weights = {}

        sparsity_level_dict = get_sparsity_level_dict(
            weights_dict,
            sparsity_level=self.sparsity_level,
            sparsity_distribution=self.sparsity_distribution,
            erk_power_scale=self.erk_power_scale,
        )
        for v_name, v in weights_dict.items():
            if not should_sparsify_weight(v_name, v):
                sparse_weights[v_name] = v
                continue

            sparsity_level = sparsity_level_dict[v_name]
            use_dense, use_low_sparsity = False, False
            if sparsity_level <= LOW_SPARSITY_THRESHOLD and sparse_val != 0:
                use_low_sparsity = True
            if len(v.shape) > 2:
                use_dense = True

            if use_dense:
                if v_name not in self.mask_dict:
                    self.mask_dict[v_name] = (
                        get_mask_dense(v.shape, sparsity_level).astype(v.dtype),
                        self.MaskType.DENSE,
                    )
                sparse_weights[v_name] = apply_mask_dense(
                    self.mask_dict[v_name][0], weights_dict[v_name]
                )
            else:
                # sparse masks need 2d tensors. So make them 2d and then
                # reshape them as needed.
                og_shape = v.shape
                if use_low_sparsity:
                    if v_name not in self.mask_dict:
                        self.mask_dict[v_name] = (
                            get_mask_low_sparsity(
                                v.shape, sparsity_level
                            ).astype(v.dtype),
                            self.MaskType.LOW_SPARSITY,
                        )

                    v = apply_mask_low_sparsity(self.mask_dict[v_name][0], v)
                else:
                    if v_name not in self.mask_dict:
                        self.mask_dict[v_name] = (
                            get_mask_high_sparsity(
                                v.shape, sparsity_level
                            ).astype(v.dtype),
                            self.MaskType.HIGH_SPARSITY,
                        )
                    v = apply_mask_high_sparsity(self.mask_dict[v_name][0], v)

                if og_shape:
                    v = v.reshape(og_shape)
                sparse_weights[v_name] = v

        return sparse_weights

    def get_num_sparsified_values(self, v_name, sparse_val):
        if v_name not in self.mask_dict:
            return 0

        mask, mask_type = self.mask_dict[v_name]
        if mask_type == self.MaskType.LOW_SPARSITY:
            original_sparse_count = mask.count_nonzero()
        elif mask_type == self.MaskType.DENSE and np.isnan(sparse_val):
            original_sparse_count = np.count_nonzero(np.isnan(mask))
        elif mask_type == self.MaskType.DENSE:  # sparse_val is 0
            mask_shape = mask.shape
            original_sparse_count = reduce(
                lambda a, b: a * b, mask_shape
            ) - np.count_nonzero(mask)
        else:  # high sparsity
            mask_shape = mask.shape
            original_sparse_count = (
                reduce(lambda a, b: a * b, mask_shape) - mask.count_nonzero()
            )
        return original_sparse_count


class SETSparsifier(BaseSparsifier):
    """ Implements the SET sparsification procedure outlined in
    https://www.nature.com/articles/s41467-018-04316-3.pdf. Note that SET can
    only be applied to FC layers and this implementation assumes currently that
    all the layers are FC. Can change this later ...

    The algorithm is pretty simple - Each FC connection is first initialized based
    on the Erdos-Renyi random graph. Thereafter, at every epoch, we drop a percentage
    of the lowest positive and highest negative weights and add random connections.
    """

    def __init__(
        self,
        n_iter,
        sparsity_level=0.5,
        sparsity_distribution='er',
        erk_power_scale=1.0,
        epsilon=None,
        zeta=None,
        seed=None,
        mask_file=None,
        sparse_val=None,
    ):
        self.n_iter = n_iter
        self.epsilon = epsilon  # For initialization
        self.zeta = zeta  # For sparsity while training
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.sparsity_level = sparsity_level
        self.sparsity_distribution = sparsity_distribution
        self.erk_power_scale = erk_power_scale
        self.last_mask = {}
        self.mask_dict = None

    def apply_sparsity(self, step):
        if step % self.n_iter == 0:
            return True
        return False

    def get_initial_masks(self, weights_dict, sparse_val):
        """Get initial masks"""
        sparse_weights = {}
        mask_dict = {}

        sparsity_level_dict = get_sparsity_level_dict(
            weights_dict,
            sparsity_level=self.sparsity_level,
            sparsity_distribution=self.sparsity_distribution,
            erk_power_scale=self.erk_power_scale,
        )
        for v_name, v in weights_dict.items():
            # check if weight should be sparsified
            if not should_sparsify_weight(v_name, v):
                sparse_weights[v_name] = v
            else:
                # The expected % of sparseness at initialization out is given by
                # (1 - epsilon*(n_curr + n_prev)/(n_curr*n_prev)
                # No need to actually sample - this in expectation is the same thing
                n_curr, n_prev = v.shape
                sparsity_level = sparsity_level_dict[v_name]
                mask = np.random.choice(
                    [1.0, sparse_val],
                    size=v.shape,
                    p=[1 - sparsity_level, sparsity_level],
                ).astype(v.dtype)
                mask_dict[v_name] = mask
                sparse_weights[v_name] = v * mask
        return mask_dict, sparse_weights

    def get_training_masks(self, weights_dict, sparse_val):
        """Get sparsity masks for training"""
        sparse_weights = {}
        mask_dict = {}

        for v_name, v in weights_dict.items():
            # check if weight should be sparsified
            if not should_sparsify_weight(v_name, v):
                sparse_weights[v_name] = v
            else:
                curr_mask = self.mask_dict[v_name].flatten()
                curr_weights = v.flatten()
                new_weights = np.zeros_like(curr_weights)

                non_mask_indices = np.argwhere(curr_mask == 1)
                mask_indices = np.argwhere(curr_mask != 1)

                # The number of trainable weights we will keep (the large amplitude ones)
                # the rest we will mask out
                n_keep = int((1 - self.zeta) * non_mask_indices.size)
                new_weights[non_mask_indices] = curr_weights[non_mask_indices]
                sorted_flattened_indices = np.argsort(np.abs(new_weights))[::-1]
                new_weights[sorted_flattened_indices[n_keep:]] = sparse_val
                curr_mask[sorted_flattened_indices[n_keep:]] = sparse_val

                # chose a random set of the previously masked values and grow them by
                # randomly initializing them
                to_grow = int(self.zeta * mask_indices.size)
                grow_values = np.random.randn(to_grow).astype(v.dtype)
                grow_indices = np.random.choice(
                    mask_indices.flatten(), to_grow, replace=False
                ).astype(v.dtype)
                new_weights[grow_indices] = grow_values
                curr_mask[grow_indices] = np.ones_like(grow_values)

                # reshape everything as need
                new_weights = new_weights.reshape(v.shape)
                curr_mask = curr_mask.reshape(v.shape)
                sparse_weights[v_name] = new_weights
                mask_dict[v_name] = curr_mask

        return mask_dict, sparse_weights

    def get_masked_weights(self, step, weights_dict, sparse_val):
        if step == 0:
            # return the initialization pattern
            self.mask_dict, sparse_weights = self.get_initial_masks(
                weights_dict, sparse_val
            )
        else:
            self.mask_dict, sparse_weights = self.get_training_masks(
                weights_dict, sparse_val
            )
        return sparse_weights

    def get_num_sparsified_values(self, v_name, sparse_val):
        if self.mask_dict is None or v_name not in self.mask_dict:
            return 0

        mask = self.mask_dict[v_name]
        mask_shape = mask.shape
        num_elements = reduce(lambda a, b: a * b, mask_shape)
        if sparse_val == 0:
            original_sparse_count = num_elements - mask.count_nonzero()
        else:  # must be nans
            assert np.isnan(sparse_val)
            original_sparse_count = np.count_nonzero(np.isnan(mask))
        return original_sparse_count


class TopKSparsifier(BaseSparsifier):
    """ Apply Top-K sparsity mask that keeps only the largest k% weights by magnitude """

    def __init__(
        self,
        n_iter,
        sparsity_level=0.5,
        sparsity_distribution='uniform',
        erk_power_scale=1.0,
        epsilon=None,
        zeta=None,
        seed=None,
        mask_file=None,
    ):
        self.n_iter = n_iter
        assert isinstance(
            sparsity_level, float
        ), f'Expected sparsity level to be a float but got {type(sparsity_level)}'
        assert (
            0.0 <= sparsity_level <= 1.0
        ), f'Expected sparsity level between 0 and 1 but got {sparsity_level}.'
        self.sparsity_level = sparsity_level
        self.sparsity_distribution = sparsity_distribution
        self.erk_power_scale = erk_power_scale
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.mask_dict = {}

    def apply_sparsity(self, step):
        if step % self.n_iter == 0:
            return True
        return False

    @staticmethod
    def get_num_dense_to_keep(sparsity, weight_size):
        """ Use TopK to compute the number of dense elements that should remain
        in the weight matrix, rounded to the floor value.
        """
        return int((1.0 - sparsity) * weight_size)

    def get_masked_weights(self, step, weights_dict, sparse_val):
        sparse_weights = {}

        sparsity_level_dict = get_sparsity_level_dict(
            weights_dict,
            sparsity_level=self.sparsity_level,
            sparsity_distribution=self.sparsity_distribution,
            erk_power_scale=self.erk_power_scale,
        )
        for v_name, v in weights_dict.items():
            # check if weight should be sparsified
            if not should_sparsify_weight(v_name, v):
                sparse_weights[v_name] = v
            else:
                # Convert NaNs to zeros before applying the mask.
                # Needed when pruning an already sparse checkpoint
                # with sparse_val = NaN
                v = np.where(np.isnan(v), 0, v)
                curr_weights = v.flatten()
                mask = np.ones_like(curr_weights)

                # Find the trainable weights (of the largest magnitude) to keep
                # and discard the rest.
                n_keep = self.get_num_dense_to_keep(
                    sparsity_level_dict[v_name], curr_weights.size
                )

                sorted_flattened_indices = np.argsort(np.abs(curr_weights))[
                    ::-1
                ]
                mask[sorted_flattened_indices[n_keep:]] = sparse_val

                # Reshape mask to match weights in weights_dict
                mask = mask.reshape(v.shape)
                self.mask_dict[v_name] = mask
                sparse_weights[v_name] = weights_dict[v_name] * mask

        return sparse_weights

    def get_num_sparsified_values(self, v_name, sparse_val):
        if self.mask_dict is None or v_name not in self.mask_dict:
            return 0

        mask = self.mask_dict[v_name]
        mask_shape = mask.shape
        num_elements = reduce(lambda a, b: a * b, mask_shape)
        if sparse_val == 0:
            original_sparse_count = num_elements - np.count_nonzero(mask)
        else:  # must be nans
            assert np.isnan(sparse_val)
            original_sparse_count = np.count_nonzero(np.isnan(mask))
        return original_sparse_count


class BalancedTopKSparsifier(BaseSparsifier):
    """ Apply a modified version of the Top-K sparsity mask where Top-K is
    applied to each column separately rather than to the entire matrix.

    Note that this modification results in a slightly lower sparsity compared
    to per-layer Top-K due to round-off errors. We include a correction for
    such error so that the final number of sparse values is the same as the
    original per-layer Top-K sparsification.
    """

    def __init__(
        self,
        n_iter,
        sparsity_level=0.5,
        sparsity_distribution='uniform',
        erk_power_scale=1.0,
        epsilon=None,
        zeta=None,
        seed=None,
        mask_file=None,
        sparse_val=None,
    ):
        self.n_iter = n_iter
        assert isinstance(
            sparsity_level, float
        ), f'Expected sparsity level to be a float but got {type(sparsity_level)}'
        assert (
            0.0 <= sparsity_level <= 1.0
        ), f'Expected sparsity level between 0 and 1 but got {sparsity_level}.'
        self.sparsity_level = sparsity_level
        self.sparsity_distribution = sparsity_distribution
        self.erk_power_scale = erk_power_scale
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.mask_dict = {}

    def apply_sparsity(self, step):
        if step % self.n_iter == 0:
            return True
        return False

    def get_masked_weights(self, step, weights_dict, sparse_val):
        sparse_weights = {}

        sparsity_level_dict = get_sparsity_level_dict(
            weights_dict,
            sparsity_level=self.sparsity_level,
            sparsity_distribution=self.sparsity_distribution,
            erk_power_scale=self.erk_power_scale,
        )
        for v_name, v in weights_dict.items():
            # check if weight should be sparsified
            if not should_sparsify_weight(v_name, v):
                sparse_weights[v_name] = v
            else:
                assert len(v.shape) == 2, "Weight must be 2D"
                nrow, ncol = v.shape
                num_sparse_per_col = int(sparsity_level_dict[v_name] * nrow)
                sorted_indices = np.argsort(np.abs(v), axis=0)[
                    :num_sparse_per_col
                ]
                mask = np.ones_like(v)
                np.put_along_axis(mask, sorted_indices, sparse_val, axis=0)

                # Compare number of sparse values with TopK sparsifier
                # and apply correction as necessary.
                n_keep = (nrow - num_sparse_per_col) * ncol
                n_keep_topk = TopKSparsifier.get_num_dense_to_keep(
                    sparsity_level_dict[v_name], v.size
                )
                if n_keep < n_keep_topk:
                    logging.warning(
                        "Balanced TopK is more sparse than TopK."
                        "Using the sparse weight from balanced TopK."
                    )

                if n_keep > n_keep_topk:
                    # Column-wise TopK is denser than per-layer TopK. Apply
                    # correction to match with per-layer Top-K.
                    curr_weights = v.copy()
                    np.put_along_axis(curr_weights, sorted_indices, 0, axis=0)
                    curr_weights = curr_weights.flatten()
                    mask = np.ones_like(curr_weights)
                    sorted_flattened_indices = np.argsort(np.abs(curr_weights))[
                        ::-1
                    ]
                    mask[sorted_flattened_indices[n_keep_topk:]] = sparse_val
                    # Reshape mask to match weights in weights_dict
                    mask = mask.reshape(v.shape)

                self.mask_dict[v_name] = mask
                sparse_weights[v_name] = weights_dict[v_name] * mask

        return sparse_weights

    def get_num_sparsified_values(self, v_name, sparse_val):
        if self.mask_dict is None or v_name not in self.mask_dict:
            return 0

        mask = self.mask_dict[v_name]
        mask_shape = mask.shape
        num_elements = reduce(lambda a, b: a * b, mask_shape)
        if sparse_val == 0:
            original_sparse_count = num_elements - np.count_nonzero(mask)
        else:  # must be nans
            assert np.isnan(sparse_val)
            original_sparse_count = np.count_nonzero(np.isnan(mask))
        return original_sparse_count


class FileSparsifier(BaseSparsifier):
    """ Load saved sparsity mask from disk and apply them on dense weights """

    def __init__(
        self,
        n_iter,
        sparsity_level=0.5,
        sparsity_distribution='uniform',
        erk_power_scale=1.0,
        epsilon=None,
        zeta=None,
        seed=None,
        mask_file=None,
        sparse_val=None,
    ):
        self.n_iter = n_iter
        # make sure this file exists
        assert mask_file is not None, "must provide a mask file or checkpoint!"
        mask_file_name = os.path.basename(mask_file)
        pattern = re.compile(fr"^{mask_file_name}(-\d+)?(.meta)?$")
        mask_dir = os.path.dirname(mask_file)
        file_exists = False
        for fname in os.listdir(mask_dir):
            if pattern.match(fname):
                file_exists = True
                break
        assert (
            file_exists
        ), f"cannot find a masks file corresponding to name {mask_file}"

        self.mask_dict = extract_mask_from_file(mask_file, sparse_val)

        assert isinstance(
            self.mask_dict, dict
        ), f"Expected <class 'dict'> in {mask_file} but got {type(self.mask_dict)}"

    def apply_sparsity(self, step):
        if step % self.n_iter == 0:
            return True
        return False

    def get_masked_weights(self, step, weights_dict, sparse_val):
        sparse_weights = {}

        for v_name, v in weights_dict.items():
            # check if weight should be sparsified
            if not should_sparsify_weight(v_name, v):
                sparse_weights[v_name] = v
            else:
                assert (
                    v_name in self.mask_dict.keys()
                ), f"Missing mask for {v_name}"

                # Populate the mask such as non-sparse elements are 1 and sparse
                # elements are sparse_val.
                mask = np.where(self.mask_dict[v_name] != 1, sparse_val, 1.0)
                self.mask_dict[v_name] = mask

                sparse_weights[v_name] = v * mask
        return sparse_weights

    def get_num_sparsified_values(self, v_name, sparse_val):
        if self.mask_dict is None or v_name not in self.mask_dict:
            return 0

        mask = self.mask_dict[v_name]
        mask_shape = mask.shape
        num_elements = reduce(lambda a, b: a * b, mask_shape)
        if sparse_val == 0:
            original_sparse_count = num_elements - mask.count_nonzero()
        else:  # must be nans
            assert np.isnan(sparse_val)
            original_sparse_count = np.count_nonzero(np.isnan(mask))
        return original_sparse_count


class CheckerboardSparsifier(BaseSparsifier):
    """ Apply Checkerboard sparsity mask that keeps strict sparsity
        pattern across rows and columns """

    def __init__(
        self,
        n_iter,
        sparsity_level=0.5,
        sparsity_distribution='uniform',
        erk_power_scale=1.0,
        epsilon=None,
        zeta=None,
        seed=None,
        mask_file=None,
        sparse_val=None,
    ):
        self.n_iter = n_iter
        assert isinstance(
            sparsity_level, float
        ), f'Expected sparsity level to be a float but got {type(sparsity_level)}'
        assert (
            0.0 <= sparsity_level <= 1.0
        ), f'Expected sparsity level between 0 and 1 but got {sparsity_level}.'
        self.sparsity_level = sparsity_level
        self.sparsity_distribution = sparsity_distribution
        self.erk_power_scale = erk_power_scale
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.mask_dict = {}

    def apply_sparsity(self, step):
        if step % self.n_iter == 0:
            return True
        return False

    def get_masked_weights(self, step, weights_dict, sparse_val):
        sparse_weights = {}

        sparsity_level_dict = get_sparsity_level_dict(
            weights_dict,
            sparsity_level=self.sparsity_level,
            sparsity_distribution=self.sparsity_distribution,
            erk_power_scale=self.erk_power_scale,
        )
        for v_name, v in weights_dict.items():
            # check if weight should be sparsified
            if not should_sparsify_weight(v_name, v):
                sparse_weights[v_name] = v
            else:
                mask = np.ones_like(v)
                # Create a row with a strictly balanced sparsity pattern
                #   This algorithm sets a range of evenly spaced values with step size = density
                #    and uses the transitions across integer thresholds (ie; 1.8 -> 2.0) to place
                #    the dense values.
                sparse_row = np.floor(
                    np.arange(0, v.shape[1] + 1)
                    * (1 - sparsity_level_dict[v_name])
                    + 1e-5
                )
                for i in range(sparse_row.shape[0] - 1):
                    sparse_row[i] = (
                        sparse_val if sparse_row[i] == sparse_row[i + 1] else 1
                    )
                sparse_row = sparse_row[:-1]
                assert len(sparse_row) == v.shape[1]

                # Apply the sparse row to each row in the weight tensor, then shift the
                #   sparse row to create the checkerboard pattern
                for y in range(v.shape[0]):
                    mask[y, :] *= sparse_row
                    sparse_row = np.roll(sparse_row, -1, 0)

                self.mask_dict[v_name] = mask
                sparse_weights[v_name] = weights_dict[v_name] * mask

        return sparse_weights

    def get_num_sparsified_values(self, v_name, sparse_val):
        if self.mask_dict is None or v_name not in self.mask_dict:
            return 0

        mask = self.mask_dict[v_name]
        mask_shape = mask.shape
        num_elements = reduce(lambda a, b: a * b, mask_shape)
        if sparse_val == 0:
            original_sparse_count = num_elements - mask.count_nonzero()
        else:  # must be nans
            assert np.isnan(sparse_val)
            original_sparse_count = np.count_nonzero(np.isnan(mask))
        return original_sparse_count


SPARSIFIER_MAP = {
    "file": FileSparsifier,
    "constant": ConstantMaskSparsifier,
    "topk": TopKSparsifier,
    "set": SETSparsifier,
    "balanced-topk": BalancedTopKSparsifier,
    "checkerboard": CheckerboardSparsifier,
}

__all__ = [
    "BalancedTopKSparsifier",
    "BaseSparsifier",
    "CheckerboardSparsifier",
    "ConstantMaskSparsifier",
    "FileSparsifier",
    "SETSparsifier",
    "SPARSIFIER_MAP",
    "TopKSparsifier",
]
