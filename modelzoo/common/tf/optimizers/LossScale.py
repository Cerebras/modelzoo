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
Loss scaling
"""
import inspect
import logging

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.train.experimental import DynamicLossScale
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import (
    loss_scale_optimizer as loss_scale_optimizer_v1,
)


# CSDynamicLossScale derives from DynamicLossScale because TF does a check
# against isinstance(obj, DynamicLossScale) that we must pass
class CSDynamicLossScale(DynamicLossScale):
    """Loss scale that dynamically adjusts itself.

    Dynamic loss scaling works by adjusting the loss scale as training progresses.
    The goal is to keep the loss scale as high as possible without overflowing the
    gradients. As long as the gradients do not overflow, raising the loss scale
    never hurts.

    The algorithm starts by setting the loss scale to an initial value. Every N
    steps that the gradients are finite, the loss scale is increased by some
    factor. However, if a NaN or Inf gradient is found, the gradients for that
    step are not applied, and the loss scale is decreased by the factor. This
    process tends to keep the loss scale as high as possible without gradients
    overflowing.
    """

    def __init__(
        self,
        initial_loss_scale=2.0 ** 15,  # See docstring for why this is big.
        increment_period=2000,
        multiplier=2.0,
        min_loss_scale=2.0 ** -14,
        max_loss_scale=2.0 ** 15,
        overflow_tolerance=0.05,
    ):
        """Creates the dynamic loss scale.

        Args:
          :param float initial_loss_scale: The loss scale to use at the
              beginning. It's better to start this at a very high number, because a
              loss scale that is too high gets lowered far more quickly than a loss
              scale that is to low gets raised. The default is 2 ** 15, which is
              approximately half the maximum float16 value.
          :param int increment_period: Increases loss scale every `increment_period`
              consecutive steps that finite gradients are encountered. If a nonfinite
              gradient is encountered, the count is reset back to zero.
          :param float multiplier: The multiplier to use when increasing or
              decreasing the loss scale.
          :param float min_loss_scale: Smallest possible loss scale value.
          :param float max_loss_scale: Largest possible loss scale value.
          :param float overflow_tolerance: Overflow tolerance.
        """
        if not np.isclose(multiplier, 2.0):
            logging.warn(
                f"Only a value of 2.0 is supported for dynamic loss "
                f"scaling multiplier, but {multiplier} was provided. "
                f"Adjusting value to 2.0 instead."
            )
            multiplier = 2

        # Call super init
        (
            getattr(self, "__super__init__", None)
            or super(CSDynamicLossScale, self).__init__
        )(
            initial_loss_scale=initial_loss_scale,
            increment_period=increment_period,
            multiplier=multiplier,
        )

        # Set up additional state variables
        # counter of steps since last LS increase
        self._steps_since_rescale = self._add_weight(
            name="steps_since_rescale", initial_value=0, dtype=tf.int64,
        )

        # counter of steps since last Inf or Nan
        self._overflows_since_rescale = self._add_weight(
            name="overflows_since_rescale", initial_value=0, dtype=tf.int64,
        )

        # A global norm tensor to use for Inf/NaN checking. If not set before
        # update() is called, the norm will be computed in update().
        self.global_norm = None

        # Leaving these variables public in order to set manually in a
        # drop-in replacement situation via direct reference or setattr
        self.min_loss_scale = min_loss_scale
        if self.min_loss_scale < 2.0 ** -14:
            raise ValueError(
                f"`min_loss_scale` of {self.min_loss_scale} is too small"
            )

        self.max_loss_scale = max_loss_scale
        if self.max_loss_scale > tf.float32.max:
            raise ValueError(
                f"Provided max_loss_scale value of {self.max_loss_scale} "
                f"exceeds FP32_MAX of {tf.float32.max}"
            )

        self.overflow_tolerance = overflow_tolerance
        if self.overflow_tolerance < 0:
            raise ValueError("loss scaling coutner threshold must be set >= 0")

    def update(self, unscaled_grads_vars):
        """
        dynamically update the loss scaling
        """

        if self.global_norm is None:
            self.global_norm = tf.linalg.global_norm(
                [g for g in unscaled_grads_vars if g is not None]
            )

        isfinite = tf.math.is_finite(self.global_norm)
        # int rep of isfinite
        isfinite_int = tf.cast(isfinite, tf.int64)

        # inc step cntr
        steps_since_rescale = self._steps_since_rescale.assign_add(1)

        # if overflow, inc overflow cntr
        overflows_since_rescale = self._overflows_since_rescale.assign_add(
            1 - isfinite_int
        )

        ratio = tf.cast(overflows_since_rescale, tf.float32) / tf.cast(
            steps_since_rescale, tf.float32
        )

        # decrease LS condition
        dec_cond = tf.cast(
            tf.math.greater(ratio, self.overflow_tolerance), tf.int64
        )

        # reset cntrs on dec_cond
        overflows_since_rescale = overflows_since_rescale.assign(
            overflows_since_rescale * (1 - dec_cond)
        )

        steps_since_rescale = steps_since_rescale.assign(
            steps_since_rescale * (1 - dec_cond)
        )

        # update the loss scaling factor on Inf/NaN
        ls_fac_ = tf.cast(1 + dec_cond, self._current_loss_scale.dtype)
        current_loss_scale = self._current_loss_scale.assign(
            self._current_loss_scale / ls_fac_
        )

        # increase loss scaling if steps_since_rescale > threshold
        ls_fac_ = tf.math.maximum(
            steps_since_rescale - self.increment_period + 1, 0
        )

        # increase loss 2x
        current_loss_scale = current_loss_scale.assign(
            current_loss_scale * tf.cast(ls_fac_ + 1, current_loss_scale.dtype)
        )

        current_loss_scale = current_loss_scale.assign(
            tf.clip_by_value(
                current_loss_scale, self.min_loss_scale, self.max_loss_scale
            )
        )

        # reset counters if > threshold
        steps_since_rescale = steps_since_rescale.assign(
            steps_since_rescale * (1 - ls_fac_)
        )
        overflows_since_rescale = overflows_since_rescale.assign(
            overflows_since_rescale * (1 - ls_fac_)
        )

        return (
            tf.group(
                current_loss_scale,
                overflows_since_rescale,
                steps_since_rescale,
            ),
            isfinite,
        )

    def get_config(self):
        config = (
            getattr(self, "__super__get_config__", None)
            or super(CSDynamicLossScale, self)
        ).get_config()
        config.update(
            {
                "overflow_tolerance": self.overflow_tolerance,
                "min_loss_scale": self.min_loss_scale,
                "max_loss_scale": self.max_loss_scale,
            }
        )
        return config


# Adapter for MixedPrecisionLossScaleOptimizer.
# Used to introduce the loss_scale property.
# This will make MixedPrecisionLossScaleOptimizer interface
# consistent with loss_scale_optimizer_v2.LossScaleOptimizer
# that has this property. Will be removed once we switch
# to v2 optimizers.
class MixedPrecisionLossScaleOptimizerAdapter(
    loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer
):
    def __init__(self, opt, loss_scale):
        super(MixedPrecisionLossScaleOptimizerAdapter, self).__init__(
            opt, loss_scale
        )

    @property
    def loss_scale(self):
        return self._loss_scale

    @property
    def optimizer(self):
        return self._optimizer


def wrap_optimizer(opt, loss_scale):
    """Wraps an optimizer with a LossScaleOptimizer."""

    if isinstance(
        opt, loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer
    ):
        raise ValueError(
            '"opt" must not already be an instance of a '
            'MixedPrecisionLossScaleOptimizer. '
            '`enable_mixed_precision_graph_rewrite` will '
            'automatically wrap the optimizer with a '
            'MixedPrecisionLossScaleOptimizer.'
        )
    # To avoid a circular dependency, we cannot depend on tf.keras. Because
    # LossScaleOptimizer is in Keras, we cannot use isinstance, so instead check
    # the class name.
    if opt.__class__.__name__ == 'LossScaleOptimizer':
        raise ValueError(
            '"opt" must not already be an instance of a '
            'LossScaleOptimizer. '
            '`enable_mixed_precision_graph_rewrite` will '
            'automatically wrap the optimizer with a '
            'LossScaleOptimizer.'
        )

    if isinstance(opt, optimizer.Optimizer):
        # For convenience, we allow the V2 version of this function to wrap the V1
        # optimizer, even though we do not document this.
        return MixedPrecisionLossScaleOptimizerAdapter(opt, loss_scale)

    # Because we cannot depend on tf.keras, we see if `opt` is an instance of the
    # Keras OptimizerV2 class by checking the subclass names.
    base_classes = inspect.getmro(opt.__class__)
    base_class_names = [cls.__name__ for cls in base_classes]
    is_loss_scale_optimizer_v2 = 'OptimizerV2' in base_class_names

    if is_loss_scale_optimizer_v2:
        # Because we cannot depend on tf.keras, we cannot unconditionally do this
        # import. But since `opt` is a Keras OptimizerV2, we know keras is
        # importable, so it is safe to do this import. (Technically, it's possible
        # to have a dependency on OptimizerV2 and not LossScaleOptimizer, but this
        # is not done in practice).
        from tensorflow.python.keras.mixed_precision.experimental import (
            loss_scale_optimizer as loss_scale_optimizer_v2,  # disable=g-import-not-at-top; pylint:
        )

        return loss_scale_optimizer_v2.LossScaleOptimizer(opt, loss_scale)

    raise ValueError(
        '"opt" must be an instance of a tf.train.Optimizer or a '
        'tf.keras.optimizers.Optimizer, but got: %s' % opt
    )
