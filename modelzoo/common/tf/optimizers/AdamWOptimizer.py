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

import re

import tensorflow as tf


class AdamWOptimizer(tf.compat.v1.train.Optimizer):
    """
    Adam Weight Decay optimizer (AdamW).
    Based on: https://github.com/google-research/bert/blob/master/optimization.py
    """

    def __init__(
        self,
        learning_rate,
        weight_decay_rate=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        use_bias_correction=False,
        exclude_from_weight_decay=None,
        use_locking=False,
        name="AdamW",
    ):

        super(AdamWOptimizer, self).__init__(use_locking, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        if exclude_from_weight_decay and (
            not (
                isinstance(exclude_from_weight_decay, list)
                or isinstance(exclude_from_weight_decay, tuple)
            )
            or not all(isinstance(i, str) for i in exclude_from_weight_decay)
        ):
            raise ValueError(
                "If specified, exclude_from_weight_decay must be a list, or a "
                f"tuple. Got '{exclude_from_weight_decay}' of type "
                f"{type(exclude_from_weight_decay)}"
            )
        # Whether to use Adam's bias correction given differences in
        # the decay of momentum and velocity terms
        self.use_bias_correction = use_bias_correction
        self.bias_correction = None

    def _create_slots(self, var_list):
        if self.use_bias_correction:
            # Create the beta1 and beta2 accumulators on the same device as
            # the first variable. Sort the var_list to make sure this device
            # is consistent across workers (these need to go on the same PS,
            # otherwise some updates are silently ignored).
            first_var = min(var_list, key=lambda x: x.name)
            self._create_non_slot_variable(
                initial_value=self.beta1,
                name="beta1_power",
                colocate_with=first_var,
            )
            self._create_non_slot_variable(
                initial_value=self.beta2,
                name="beta2_power",
                colocate_with=first_var,
            )

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        if grad is None or var is None:
            return None

        param_name = self._get_variable_name(var.name)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        next_m = tf.multiply(self.beta1, m) + tf.multiply(
            1.0 - self.beta1, grad
        )
        next_v = tf.multiply(self.beta2, v) + tf.multiply(
            1.0 - self.beta2, tf.square(grad)
        )
        update = next_m / (tf.sqrt(next_v) + self.epsilon)
        if self.use_bias_correction:
            update *= self._get_bias_correction()

        if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * var

        update_with_lr = self.learning_rate * update
        next_var = var - update_with_lr

        assignments = [var.assign(next_var), m.assign(next_m), v.assign(next_v)]

        return tf.group(*assignments, name=self._name)

    def _resource_apply_dense(self, grad, var):
        return self._apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        # IndexedSlices have issues at XLA compile.
        # However, this sparse code won't be triggered
        # as long as we multiply the embedding
        # table by 1.0.
        out_grad = tf.IndexedSlices(
            grad, indices, dense_shape=var.shape.as_list()
        )
        return self._apply_dense(out_grad, var)

    def _get_variable_name(self, param_name):
        """
        Get the variable name from the tensor name.
        """
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _do_use_weight_decay(self, param_name):
        """
        Whether to use L2 weight decay for `param_name`.
        """
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_beta_accumulators(self):
        assert self.use_bias_correction
        with tf.init_scope():
            graph = tf.compat.v1.get_default_graph()
            return (
                self._get_non_slot_variable("beta1_power", graph=graph),
                self._get_non_slot_variable("beta2_power", graph=graph),
            )

    def _get_bias_correction(self):
        assert self.use_bias_correction
        if self.bias_correction is None:
            beta1_power, beta2_power = self._get_beta_accumulators()
            self.bias_correction = tf.sqrt(1.0 - beta2_power) / (
                1.0 - beta1_power
            )
        return self.bias_correction

    def _finish(self, update_ops, name_scope):
        if not self.use_bias_correction:
            return tf.group(update_ops)

        # Update the power accumulators.
        with tf.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with tf.compat.v1.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self.beta1, use_locking=self._use_locking
                )
                update_beta2 = beta2_power.assign(
                    beta2_power * self.beta2, use_locking=self._use_locking
                )
        return tf.group(
            *update_ops + [update_beta1, update_beta2], name=name_scope
        )
