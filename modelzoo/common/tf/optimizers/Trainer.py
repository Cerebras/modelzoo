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
The trainer class.
"""
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.train.experimental import DynamicLossScale
from tensorflow.python.keras.mixed_precision.experimental.loss_scale_optimizer import (
    LossScaleOptimizer,
)
from tensorflow.python.training.experimental.loss_scale_optimizer import (
    MixedPrecisionLossScaleOptimizer,
)

from modelzoo.common.tf.layers.utils import summary_layer
from modelzoo.common.tf.optimizers.AdamWOptimizer import AdamWOptimizer
from modelzoo.common.tf.optimizers.GradAccumOptimizer import GradAccumOptimizer
from modelzoo.common.tf.optimizers.LossScale import (
    CSDynamicLossScale,
    wrap_optimizer,
)


class Trainer:
    """
    The trainer class that builds
    train ops based on the given
    configuration parameters.

    :param dict params: Trainer configuration parameters.
    :param bool tf_summary: Summaries flag.
    :param bool mixed_precision: Mixed precision flag.
    """

    def __init__(self, params, tf_summary=False, mixed_precision=False):

        # Optimizer params
        self._optimizer = None
        self._optimizer_type = params["optimizer_type"].lower()
        self._momentum = params.get("momentum", 0.9)
        self._beta1 = params.get("beta1", 0.9)
        self._beta2 = params.get("beta2", 0.999)
        self._epsilon = float(
            params.get("epsilon", 1e-05 if mixed_precision else 1e-08)
        )
        self._use_bias_correction = params.get("use_bias_correction", False)

        self._weight_decay_rate = params.get("weight_decay_rate", 0.0)
        self._exclude_from_weight_decay = params.get(
            "exclude_from_weight_decay",
            ["LayerNorm", "layer_norm", "bias", "Bias"],
        )

        self._rmsprop_decay = params.get("rmsprop_decay", 0.9)
        self._rmsprop_momentum = params.get("rmsprop_momentum", 0.0)

        # Learning rate params
        self._lr_params = params["learning_rate"]
        if not isinstance(self._lr_params, (float, str, dict, list)):
            raise ValueError(
                f"Learning rate must be a float, a dict, or a list of dicts. "
                f"Got {type(self._lr_params)}"
            )

        # Loss scaling
        self._loss_scaling_factor = params.get("loss_scaling_factor", 1.0)
        if isinstance(
            self._loss_scaling_factor, str
        ) and self._loss_scaling_factor not in ['dynamic', 'tf_dynamic']:
            raise ValueError(
                "Loss scaling factor must be either numeric or "
                "one of the string values ['dynamic, 'tf_dynamic']"
            )

        # Dynamic loss scaling (DLS) params required by
        # CS-supported and tf native DLS optimizers
        self._initial_loss_scale = params.get("initial_loss_scale", 2.0 ** 15)
        self._steps_per_increase = params.get("steps_per_increase", 2000)

        # Extra DLS params required only by CS-supported
        # DLS optimizer (loss_scaling_factor=='dynamic')
        self._min_loss_scale = params.get("min_loss_scale", 2.0 ** -14)
        self._max_loss_scale = params.get("max_loss_scale", 2.0 ** 15)
        self._overflow_tolerance = params.get("overflow_tolerance", 0.05)

        # Gradient clipping params
        self._max_gradient_norm = params.get("max_gradient_norm", 0)
        self._max_gradient_value = params.get("max_gradient_value", 0)

        # Gradient accumulation params
        self._grad_accum_steps = params.get("grad_accum_steps", 1)

        # Util params
        self._log_summaries = params.get("log_summaries", False)
        self._log_grads = params.get("log_grads", False)
        self._log_hists = params.get("log_hists", False)
        self._disable_lr_steps_reset = params.get(
            "disable_lr_steps_reset", False
        )
        self._denormal_range = 2 ** -14 if mixed_precision else 2 ** -126

        self._gradient_global_norm = None
        self._loss_scale_value = None
        self.tf_summary = tf_summary
        self._ws_summary = params.get("ws_summary", False)

    def build_train_ops(self, loss):
        """
        Setup optimizer and build train ops.

        :param Tensor loss: The loss tensor
        :return: Train ops
        """
        self._optimizer = self.build_optimizer()
        grads_and_vars = self._optimizer.compute_gradients(
            tf.cast(loss, tf.float32)
        )

        if self._log_summaries:
            self._gradient_global_norm = tf.linalg.global_norm(
                [g for (g, v) in grads_and_vars]
            )
            if not self.is_grad_accum():
                tf.compat.v1.summary.scalar(
                    'train/unclipped_grad_norm', self._gradient_global_norm
                )

                if self._ws_summary:
                    gradient_num_zeros = tf.reduce_sum(
                        [
                            tf.reduce_sum(tf.cast(tf.equal(g, 0.0), tf.float32))
                            for (g, v) in grads_and_vars
                        ],
                    )
                    tf.compat.v1.summary.scalar(
                        "train/grad_num_zeros", gradient_num_zeros
                    )

                    self._params_global_norm = tf.linalg.global_norm(
                        [v for (g, v) in grads_and_vars]
                    )
                    tf.compat.v1.summary.scalar(
                        'train/params_norm', self._params_global_norm
                    )

        # This code mimics the CS1 dynamic loss scaling
        # kernel implementation, where the global norm of
        # weight gradients is computed to detect NaN/Inf and
        # is reused in gradient clipping. This saves compute
        # and simplifies kernel matching.
        if isinstance(
            self._optimizer,
            (LossScaleOptimizer, MixedPrecisionLossScaleOptimizer),
        ):
            # CSDynamicLossScale checks the global norm of the weight
            # gradients to determine whether a NaN/Inf has occurred when its
            # update() method is called. If its global_norm field is set, then
            # it will just check that value instead of recomputing the norm.
            if hasattr(self._optimizer.loss_scale, 'global_norm'):
                if self._gradient_global_norm is None:
                    self._gradient_global_norm = tf.linalg.global_norm(
                        [g for (g, v) in grads_and_vars]
                    )
                self._optimizer.loss_scale.global_norm = (
                    self._gradient_global_norm
                )

        clipped_grads_and_vars = self.clip_gradients(
            grads_and_vars, global_norm=self._gradient_global_norm
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()

        train_op = self._optimizer.apply_gradients(
            clipped_grads_and_vars, global_step,
        )

        if self._log_summaries and self._log_grads and not self.is_grad_accum():
            # Log the scaled gradients
            for (g, v) in grads_and_vars:
                if "kernel" in v.name:
                    self.log_training_summaries(
                        self._rescale(g), v.name, f"kernel_grads"
                    )
                elif "bias" in v.name:
                    self.log_training_summaries(
                        self._rescale(g), v.name, f"bias_grads"
                    )

        return train_op

    def build_optimizer(self):
        """
        Setup the optimizer.

        :returns: The optimizer
        """
        lr = self.get_learning_rate()
        if self._log_summaries and not self.is_grad_accum():
            tf.compat.v1.summary.scalar('train/lr', lr)

        optimizer = None
        if self._optimizer_type == "sgd":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=lr, name="SGD",
            )
        elif self._optimizer_type == "momentum":
            optimizer = tf.compat.v1.train.MomentumOptimizer(
                learning_rate=lr, momentum=self._momentum, name="SGDM",
            )
        elif self._optimizer_type == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=lr,
                beta1=self._beta1,
                beta2=self._beta2,
                epsilon=self._epsilon,
                name="Adam",
            )
        elif self._optimizer_type == "adamw":
            optimizer = AdamWOptimizer(
                learning_rate=lr,
                weight_decay_rate=self._weight_decay_rate,
                beta1=self._beta1,
                beta2=self._beta2,
                epsilon=self._epsilon,
                exclude_from_weight_decay=self._exclude_from_weight_decay,
                use_bias_correction=self._use_bias_correction,
                name="AdamW",
            )
        elif self._optimizer_type == "rmsprop":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=lr,
                use_locking=False,
                centered=False,
                decay=self._rmsprop_decay,
                momentum=self._rmsprop_momentum,
                name="RMSProp",
            )
        else:
            raise ValueError(f'Unsupported optimizer {self._optimizer_type}')

        # Set up loss scale
        loss_scale = None
        if self.uses_dynamic_loss_scaling():
            if self._loss_scaling_factor == 'dynamic':
                # Explicit Cerebras System optimized dynamic loss scaling
                loss_scale = CSDynamicLossScale(
                    initial_loss_scale=self._initial_loss_scale,
                    increment_period=self._steps_per_increase,
                    multiplier=2.0,
                    min_loss_scale=self._min_loss_scale,
                    max_loss_scale=self._max_loss_scale,
                    overflow_tolerance=self._overflow_tolerance,
                )
            else:
                # For any Cerebras System run, DynamicLossScale will be
                # automatically replaced with CSDynamicLossScale
                loss_scale = DynamicLossScale(
                    initial_loss_scale=self._initial_loss_scale,
                    increment_period=self._steps_per_increase,
                    multiplier=2.0,
                )

            self._loss_scale_value = loss_scale()
            if self._log_summaries and not self.is_grad_accum():
                tf.compat.v1.summary.scalar(
                    'train/loss_scale', self._loss_scale_value
                )
            if self.tf_summary:
                summary_layer(tf.cast(self._loss_scale_value, tf.float16))
        elif self.uses_static_loss_scaling():
            loss_scale = self._loss_scaling_factor
            self._loss_scale_value = self._loss_scaling_factor

        # Wraps optimizer with:
        # V1 optimizer:
        #   loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer
        # V2 optimizer (i.e. Keras):
        #   loss_scale_optimizer_v2.LossScaleOptimizer
        # MixedPrecisionLossScaleOptimizer returns unscaled grads
        # Some may be NaNs, in which case, apply_gradients() won't apply
        # them and may adjust the loss scaling factor
        if loss_scale is not None:
            optimizer = wrap_optimizer(optimizer, loss_scale=loss_scale)

        # Wraps optimizer with GradAccumOptimizer
        # for gradient accumulation
        if self.is_grad_accum():
            optimizer = GradAccumOptimizer(
                optimizer, grad_accum_steps=self._grad_accum_steps
            )

        return optimizer

    def get_learning_rate(self):
        """
        Define the learning rate schedule.
        Currently supports:
        - constant
        - exponential
        - linear
        - polynomial
        - piecewise constant
        - inverse exponential time decay (not supported natively)

        learning_rate can be specified in yaml as:
        - a single float for a constant learning rate
        - a dict representing a single decay schedule
        - a list of dicts (for a series of decay schedules)

        :returns: the learning rate tensor
        """

        def _get_scheduler(schedule_params, step):
            """
            Parses a dict of learning rate scheduler specifications and
            returns a learning rate tensor.

            :param dict schedule_params:
                    A dict with a "scheduler" key (e.g.,
                    schedule_params["scheduler"] = "Exponential") and all
                    params schedulers of that type need.
            :param tf.Tensor step:
                    The step that the scheduler should use to calculate the
                    learning rate.

            :returns: The learning rate tensor.
            """
            scheduler = schedule_params["scheduler"]
            if scheduler == "Constant":
                return tf.constant(
                    schedule_params["learning_rate"], dtype=tf.float32
                )
            elif scheduler == "Exponential":
                return tf.compat.v1.train.exponential_decay(
                    schedule_params["initial_learning_rate"],
                    step,
                    schedule_params["decay_steps"],
                    schedule_params["decay_rate"],
                    staircase=schedule_params.get("staircase", False),
                )
            elif scheduler == "PiecewiseConstant":
                return tf.compat.v1.train.piecewise_constant(
                    step,
                    boundaries=schedule_params["boundaries"],
                    values=schedule_params["values"],
                )
            elif scheduler == "Polynomial" or scheduler == "Linear":
                power = (
                    1.0
                    if scheduler == "Linear"
                    else schedule_params.get("power", 1.0)
                )
                return tf.compat.v1.train.polynomial_decay(
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    global_step=step,
                    decay_steps=schedule_params["steps"],
                    end_learning_rate=schedule_params["end_learning_rate"],
                    power=power,
                    cycle=schedule_params.get("cycle", False),
                )
            elif scheduler == "Cosine":
                return tf.compat.v1.train.cosine_decay(
                    learning_rate=schedule_params["initial_learning_rate"],
                    global_step=step,
                    decay_steps=schedule_params["decay_steps"],
                    alpha=schedule_params.get("alpha", 0.0),
                )
            else:
                raise ValueError(f"Unsupported LR scheduler {scheduler}")

        # handle a constant learning rate
        # scientific notation (e.g. "1e-5") parsed as string in yaml
        if isinstance(self._lr_params, (float, str)):
            return tf.constant(float(self._lr_params), dtype=tf.float32)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        # handle a single decay schedule
        if isinstance(self._lr_params, dict):
            return _get_scheduler(self._lr_params, global_step)

        # handle a list of decay schedules
        assert isinstance(self._lr_params, list)
        if len(self._lr_params) == 1:
            return _get_scheduler(self._lr_params[0], global_step)

        total_steps = 0
        schedule_sequence = []
        # if disable_lr_steps_reset is True, global_step will not be offset,
        # meaning that schedules will overlap rather than occur sequentially.
        # helps replicate Google's LR schedules on BERT.
        step_reset_mask = 1 - int(self._disable_lr_steps_reset)
        for i, schedule_params in enumerate(self._lr_params):
            # default argument needed so that schedule is captured in for loop
            # see https://docs.python.org/3/faq/programming.html#id10
            schedule_fn = lambda sp=schedule_params, ts=total_steps: _get_scheduler(
                sp, global_step - (ts * step_reset_mask)
            )
            if i == len(self._lr_params) - 1:
                break
            # all schedules except final become cases, `decay_steps` is used
            # by cosine decay schedule currently
            if (
                "steps" not in schedule_params
                and "decay_steps" not in schedule_params
            ):
                raise ValueError(
                    "Non-final LR schedules must specify number of steps."
                )
            # one of two cases to enable schedules
            if "steps" in schedule_params:
                total_steps += schedule_params["steps"]
            elif (
                "decay_steps" in schedule_params
                and schedule_params["scheduler"] == "Cosine"
            ):
                # add this case for cosine decay schedule
                total_steps += schedule_params["decay_steps"]
            schedule_sequence.append(
                (tf.less(global_step, total_steps), schedule_fn)
            )
        # final schedule becomes the default
        return tf.case(schedule_sequence, default=schedule_fn)

    def clip_gradients(self, grads_vars, global_norm=None):
        """
        Performs basic gradient clipping:
            - by global norm if self._max_gradient_norm is set
            - by value if self._max_gradient_value is set, to the symmetric
              range (-self._max_gradient_value, self._max_gradient_value)

        :param Tensor grads_vars: List of ``(grad, var)`` tuples
        """
        # clip by norm
        if self._max_gradient_norm:
            if self._max_gradient_value:
                raise ValueError(
                    "Gradients can be clipped by norm or by value, but not both. "
                    "Do not set both max_gradient_norm and max_gradient_value."
                )
            if self._max_gradient_norm < 0:
                raise ValueError(
                    f"max_gradient_norm cannot be negative. Got "
                    f"{self._max_gradient_norm}"
                )
            gradients = [g for (g, v) in grads_vars]
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self._max_gradient_norm, use_norm=global_norm,
            )
            grads_vars = [
                (clipped_gradients[i], grads_vars[i][1])
                for i in range(len(gradients))
            ]
        # clip by value
        elif self._max_gradient_value:
            if self._max_gradient_value < 0:
                raise ValueError(
                    f"max_gradient_value cannot be negative. Got "
                    f"{self._max_gradient_value}"
                )
            for i, (g, v) in enumerate(grads_vars):
                clipped_gradient = tf.clip_by_value(
                    g, -self._max_gradient_value, self._max_gradient_value
                )
                grads_vars[i] = (clipped_gradient, v)
        return grads_vars

    def log_training_summaries(self, tensor, name, family):
        """
        Make summaries for training. Plotting summaries for
        - Sparsity of tensor
        - Histogram of tensor (on log scale)
        - Denormals in tensor
        - Norm of tensor

        :param Tensor tensor: tensor to plot summaries for
        :param str name: name of the tensor to plot summaries for
        :param str family: family that the tensor belongs to (kernel / bias)
        """

        tf.compat.v1.summary.scalar(
            f"sparsity_{family}/{name}",
            (
                1.0
                - tf.math.count_nonzero(tensor, dtype=tf.float32)
                / tf.size(tensor, out_type=tf.float32)
            ),
        )
        tf.compat.v1.summary.scalar(
            f"denormal_{family}/{name}",
            (
                tf.reduce_sum(
                    tf.cast(
                        tf.math.logical_and(
                            tf.math.less(tf.abs(tensor), self._denormal_range),
                            tf.math.not_equal(tensor, 0),
                        ),
                        tf.float32,
                    )
                )
                / tf.size(tensor, out_type=tf.float32)
            ),
        )
        tf.compat.v1.summary.scalar(
            f"norm_{family}/{name}", tf.linalg.global_norm([tensor])
        )
        if self._log_hists:
            tf.compat.v1.summary.histogram(
                f"{family}/{name}",
                tf.math.log(tf.cast(tf.abs(tensor), tf.float32) + 2.0 ** -50)
                / tf.math.log(2.0),
            )

    def _rescale(self, g):
        """
        Scale the gradients for plotting

        :param Tensor g: tensor to scale
        """
        try:
            output = g * tf.cast(self._loss_scale_value, g.dtype)
        except Exception as e:
            tf.compat.v1.logging.debug(e)
            output = g
        return output

    def is_grad_accum(self):
        return True if self._grad_accum_steps > 1 else False

    def uses_loss_scaling(self):
        return (
            self.uses_dynamic_loss_scaling() or self.uses_static_loss_scaling()
        )

    def uses_dynamic_loss_scaling(self):
        return self._loss_scaling_factor in ['dynamic', 'tf_dynamic']

    def uses_static_loss_scaling(self):
        return (
            not isinstance(self._loss_scaling_factor, str)
            and np.isscalar(self._loss_scaling_factor)
            and not np.isclose(self._loss_scaling_factor, 1.0)
        )

    @property
    def gradient_global_norm(self):
        return self._gradient_global_norm

    @property
    def loss_scale_value(self):
        return self._loss_scale_value

    @property
    def grad_accum_steps(self):
        return self._grad_accum_steps

    @property
    def log_summaries(self):
        return self._log_summaries

    @property
    def optimizer(self):
        return self._optimizer
