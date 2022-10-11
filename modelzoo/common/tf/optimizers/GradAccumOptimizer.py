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
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import ops
from tensorflow.python.keras.mixed_precision.experimental.loss_scale_optimizer import (
    LossScaleOptimizer,
)
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer as opt
from tensorflow.python.training.experimental.loss_scale_optimizer import (
    MixedPrecisionLossScaleOptimizer,
)


class GradAccumOptimizer(tf.compat.v1.train.Optimizer):
    """
    Gradient accumulation optimizer.
    Wraps the provided optimizer by adding
    gradient accumulation functionality.

    This functionality enables a training mode
    where gradients are accumulated over every
    grad_accum_steps. At every grad_accum_steps
    step, the accumulated gradients are used to
    update the weights. After that the gradients
    are set to zero, and the process continues.
    This is equivalent to training with the
    batch size of grad_accum_steps*batch_size.

    This gives a possibility to train models with
    any batch size on a single GPU, provided
    that they fit into memory with batch size 1.
    """

    def __init__(self, optimizer, grad_accum_steps):
        """
        Initializes a `GradAccumOPtimizer`.

        :param tf.compat.v1.train.Optimizer optimizer: Optimizer to be wrapped
            into gradient accumulation.
        :param int grad_accum_steps: Number of gradient accumulation steps.
        """

        super(GradAccumOptimizer, self).__init__(
            use_locking=False, name="GradAccumOptimizer"
        )

        if not isinstance(optimizer, tf.compat.v1.train.Optimizer):
            raise TypeError(
                "optimizer must be an instance of "
                "tf.compat.v1.train.Optimizer, "
                f"but got: {optimizer}"
            )
        if grad_accum_steps < 1:
            raise ValueError(
                "grad_accum_steps must be positive, "
                f"but got {grad_accum_steps}"
            )
        self._optimizer = optimizer
        self._grad_accum_steps = grad_accum_steps
        # Step counter variable
        self._accum_step_counter = tf.compat.v1.get_variable(
            "accum_step_counter",
            shape=[],
            dtype=tf.int32,
            initializer=tf.compat.v1.constant_initializer(0),
            trainable=False,
        )
        self._accum_grads = None  # Variables holding accumulated gradients

    def compute_gradients(
        self,
        loss,
        var_list=None,
        gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None,
    ):
        """
        Computes and accumulates gradients.
        Returns a list of (gradient, variable)
        pairs, where the new gradient is a sum
        of the gradient for the current batch and
        gradient values accumulated so far. The
        accumulated values are set to zero every
        grad_accum_steps. See `tf.compat.v1.train.Optimizer`
        for description of input arguments. 
        """
        grads_and_vars = self._optimizer.compute_gradients(
            loss / self._grad_accum_steps,
            var_list,
            gate_gradients,
            aggregation_method,
            colocate_gradients_with_ops,
            grad_loss,
        )
        if tf.distribute.has_strategy():
            if tf.distribute.in_cross_replica_context():
                raise RuntimeError(
                    "`_reduce_and_accumulate_gradients` "
                    "should be called in replica context."
                )
            return tf.distribute.get_replica_context().merge_call(
                self._reduce_and_accumulate_gradients, args=(grads_and_vars,),
            )
        return self._accumulate_gradients(grads_and_vars)

    def _accumulate_gradients(self, grads_and_vars, distribution=None):
        """
        Accumulate gradients. Current accumulated gradients are
        stored in self._accum_grads (list of non-trainable variables).

        :param grads_and_vars: List of (gradient, variable) tuples
            returned from self._optimizer.compute_gradients. If
            run in distributed, the gradients need to be reduced
            across replicas before the function call.
        :params tf.distribute.Strategy distribution: Distribution strategy.

        :returns: List of (gradient, variable) tuples with accumulated
            gradient values.
        """
        self._accum_grads = [
            tf.compat.v1.get_variable(
                self._get_variable_name(v.name) + f"_grad_accum_{k}",
                shape=distribution.experimental_local_results(g)[0].shape
                if distribution
                else g.shape,
                dtype=distribution.experimental_local_results(g)[0].dtype
                if distribution
                else g.dtype,
                initializer=tf.compat.v1.constant_initializer(0.0),
                trainable=False,
            )
            for k, (g, v) in enumerate(grads_and_vars)
        ]
        next_accum_grads = [
            self._accum_grads[k].assign_add(g)
            for k, (g, v) in enumerate(grads_and_vars)
        ]
        grads_and_vars = [
            (next_accum_grads[k], v) for k, (_, v) in enumerate(grads_and_vars)
        ]
        return grads_and_vars

    def _reduce_and_accumulate_gradients(self, distribution, grads_and_vars):
        """
        Reduce gradients from all replicas
        and accumulate. Used for distributed runs.

        :params tf.distribute.Strategy distribution: Distribution strategy.
        :param grads_and_vars: List of (gradient, variable) tuples
            returned from self._optimizer.compute_gradients.

        :returns: List of (gradient, variable) tuples where
            the resulting gradients were first reduced
            across replicas and then accumulated.
        """
        reduced_grads = distribution.extended.batch_reduce_to(
            ds_reduce_util.ReduceOp.SUM, grads_and_vars
        )
        grads_and_vars = [
            (reduced_grads[k], v) for k, (_, v) in enumerate(grads_and_vars)
        ]
        return self._accumulate_gradients(grads_and_vars, distribution)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Apply gradients to variables.
        See `tf.compat.v1.train.Optimizer`
        for description of input arguments. 
        """
        if tf.distribute.has_strategy():
            if tf.distribute.in_cross_replica_context():
                raise RuntimeError(
                    "`_distributed_apply` should be called in replica context."
                )
            return tf.distribute.get_replica_context().merge_call(
                self._distributed_apply,
                args=(grads_and_vars, global_step, name),
            )
        return self._apply_gradients(grads_and_vars, global_step, name)

    def _apply_gradients(
        self, grads_and_vars, global_step=None, name=None, distribution=None
    ):
        """
        Apply gradients to variables every self._grad_accum_steps steps.

        :param grads_and_vars: List of (gradient, variable) tuples.
        :param Tensor global_step: Global step.  
        :param str name: Optional name for the returned operation.  
        :param tf.distribute.Strategy distribution: Distribution strategy (optional).

        :returns: Train ops.
        """
        next_accum_step = self._accum_step_counter.assign_add(1)

        def _apply_grads_and_reset_accum(
            grads_and_vars, accum_step_counter, accum_grads
        ):
            if tf.distribute.has_strategy():

                def _apply_fn():
                    return self._apply_reduced_gradients(
                        distribution, grads_and_vars, global_step, name,
                    )

                if isinstance(
                    self._optimizer, MixedPrecisionLossScaleOptimizer
                ):
                    grads = [g for g, _ in grads_and_vars]
                    with tf.compat.v1.get_default_graph().control_dependencies(
                        grads
                    ):
                        (
                            loss_scale_update_op,
                            should_apply_grads,
                        ) = self._optimizer._loss_scale.update(grads)
                    maybe_apply_op = tf.cond(
                        should_apply_grads,
                        _apply_fn,
                        # Workaround to create a "void op" when
                        # an infinite batch discovered. Need
                        # this to ensure that tf.cond returns
                        # tensors of same type on both branches.
                        lambda: tf.constant(0, dtype=tf.int64),
                    )
                    train_op = tf.group(
                        maybe_apply_op, loss_scale_update_op, name=name
                    )
                else:
                    train_op = _apply_fn()
            else:
                train_op = self._optimizer.apply_gradients(
                    grads_and_vars, global_step, name
                )

            with tf.compat.v1.get_default_graph().control_dependencies(
                [train_op]
            ):
                zero_accum_step = accum_step_counter.assign(0)
                zero_accum_grads = [
                    g.assign(tf.zeros_like(g)) for g in accum_grads
                ]
            return tf.group(train_op, zero_accum_step, zero_accum_grads)

        return tf.cond(
            pred=tf.equal(next_accum_step, self._grad_accum_steps),
            true_fn=lambda: _apply_grads_and_reset_accum(
                grads_and_vars, self._accum_step_counter, self._accum_grads
            ),
            false_fn=lambda: tf.group(
                next_accum_step, [g for g, _ in grads_and_vars]
            ),
        )

    def _distributed_apply(
        self, distribution, grads_and_vars, global_step=None, name=None
    ):
        """
        Apply gradients to variables every self._grad_accum_steps steps.
        Distributed version of _apply_gradients. 
        """
        # Since the gradients were ensured to have same value on each
        # replica after `compute_gradients`, the mean reduction has no effect
        # on values and is only needed to convert PerReplica gradient
        # tensors to mirrored.
        next_accum_grads = distribution.extended.batch_reduce_to(
            ds_reduce_util.ReduceOp.MEAN, grads_and_vars
        )
        grads_and_vars = [
            (next_accum_grads[k], v) for k, (_, v) in enumerate(grads_and_vars)
        ]
        return self._apply_gradients(
            grads_and_vars, global_step, name, distribution
        )

    def _apply_reduced_gradients(
        self, distribution, grads_and_vars, global_step=None, name=None,
    ):
        """
        This is a truncated version of tf.compat.v1.Optimizer's
        apply_gradients, where cross-replica gradient reduction
        was removed. We perform the reduction in `compute_gradients`
        instead. This is needed to ensure that `compute_gradients`
        returns the actual accumulated gradient.
        """
        # Special handling to obtain optimizer
        # if training with loss scaling
        optimizer = (
            self._optimizer.optimizer
            if isinstance(self._optimizer, MixedPrecisionLossScaleOptimizer)
            else self._optimizer
        )

        # Note that this is called in a cross-replica context.
        var_list = [v for _, v in grads_and_vars]
        with ops.init_scope():
            optimizer._create_slots(var_list)

        def update(v, g):
            """Apply gradients to a replica variable."""
            assert v is not None

            try:
                # Convert the grad to Tensor or IndexedSlices if necessary.
                g = ops.convert_to_tensor_or_indexed_slices(g)
            except TypeError:
                raise TypeError(
                    "Gradient must be convertible to a Tensor"
                    " or IndexedSlices, or None: %s" % g
                )
            if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                raise TypeError(
                    "Gradient must be a Tensor, IndexedSlices, or None: %s" % g
                )
            p = opt._get_processor(v)
            scope_name = v.op.name

            # device_policy is set because non-mirrored tensors will be read in
            # `update_op`. `_resource_apply_dense`, `lr_t`, `beta1_t` and `beta2_t`
            # is an example.
            with ops.name_scope("update_" + scope_name):
                return p.update_op(optimizer, g)

        with ops.name_scope(name, optimizer.get_name()) as name:
            optimizer._prepare()

            update_ops = [
                op
                for grad, var in grads_and_vars
                for op in distribution.extended.update(
                    var, update, args=(grad,), group=False
                )
            ]

            def finish(self, update_ops):
                return optimizer._finish(update_ops, "update")

            non_slot_devices = distribution.extended.non_slot_devices(var_list)
            finish_updates = distribution.extended.update_non_slot(
                non_slot_devices, finish, args=(self, update_ops), group=False,
            )
            if global_step is None:
                apply_updates = distribution.group(finish_updates, name=name)
            else:
                with ops.control_dependencies(finish_updates):
                    apply_updates = distribution.extended.update(
                        global_step,
                        state_ops.assign_add,
                        args=(1,),
                        kwargs={"name": name},
                    )

            if isinstance(apply_updates, ops.Tensor):
                apply_updates = apply_updates.op
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
                train_op.append(apply_updates)

            return apply_updates

    def _get_variable_name(self, var_name):
        """
        Get the variable name from the tensor name.
        """
        m = re.match("^(.*):\\d+$", var_name)
        if m is not None:
            var_name = m.group(1)
        return var_name

    @property
    def loss_scale(self):
        loss_scale = None
        if isinstance(
            self._optimizer, MixedPrecisionLossScaleOptimizer
        ) or isinstance(self._optimizer, LossScaleOptimizer):
            loss_scale = self._optimizer.loss_scale
        return loss_scale
