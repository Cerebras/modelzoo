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

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.python.ops.control_flow_util_v2 import (
    CondBranchFuncGraph,
    WhileBodyFuncGraph,
    WhileCondFuncGraph,
)
from tensorflow.python.util import nest

RECOMPUTED_TENSOR_ATTR_NAME = '_recomputed_tensor'


class AbstractRecomputeWrapper(ABC):
    """Utility functions for the decorator `tf.custom_gradient
    <https://www.tensorflow.org/api_docs/python/tf/custom_gradient>`_,
    when used in training.

    An abstract class to handle many small requirements when using the
    decorator `tf.custom_gradient
    <https://www.tensorflow.org/api_docs/python/tf/custom_gradient>`_.
    This class is used to recompute the activations during the backward
    propagation part of a training step. This code acts as a backbone for
    recompute wrappers and reversible layers.

    The following utility functions are designed to make it easy to implement
    the recomputation:

        - ``_set_recomputed_tensor`` and ``_check_get_recomputed_tensor``.

            These functions to attach the recomputed tensors to the
            corresponding forward pass tensors. These functions are useful for
            passing the recomputed tensors between, for example, reversible
            layers, so that we do not need to save any tensors.

        - ``_block_recompute_and_gradients``.

            This function takes a forward block of the computation, recomputes
            the block, and then calculates and returns the gradients associated
            with the block.

        - **Scope handling functions**

            - ``tf.custom_gradient``.

                This structure names the scopes of the gradients. However, this
                naming is based on the ``IdentityN`` ops it attaches to the
                portion of the graph for which the user would like to add a
                custom gradient. This is not always convenient. Moreover, the
                ``tf.custom_gradient`` does not track the appropriate control
                flow contexts for the variables used in that portion of the
                graph. The scope handling functions in this class are helpful
                here.

            - ``_get_clean_grad_scope``

                This function cleans the named scope for clean graphs.

            - ``_update_variables_for_context``

                This function finds the correct variable tensors for the
                control flow contexts: for example, to use recomputation inside
                a while-loop).

    The basic structure for a recompute layer is as follows:

        - Define a custom gradient function using ``tf.custom_gradient`` inside\
        the ``__call__`` function of a recompute layer.

        - Inside the ``__call__`` function, call the forward propagation of the\
        layer and define the recompute+gradient function. We recommend you\
        use the ``_block_recompute_and_gradients`` function).

    """

    # Class variable to track whether a warning has been printed about
    # variables that cannot be found inside the desired control flow context.
    CtrlFlowWarnedOnce = False

    def _set_recomputed_tensor(self, tensor, recomputed_tensor):
        """Sets the RECOMPUTED_TENSOR_ATTR_NAME tensor on the input tensor.

        Sets the RECOMPUTED_TENSOR_ATTR_NAME tensor on the input tensor to
        point to the ``recomputed_tensor`` to communicate the
        ``recomputed_tensor`` to any layer that might be able to consume it.

        This approach allows, for example, an upstream layer in a reversible
        model to recompute using the existing recomputed tensor rather than its
        original outputs. Thus, the upstream layer does not need to save its
        original outputs for backprop.

        Args:
            tensor (Tensor): The forward propagation tensor to attach a
                recomputed tensor to.
            recomputed_tensor (Tensor): The recomputed forward tensor to
                attach to the forward tensor, so that other layers can get
                access to it, instead of saving the forward prop tensor.
        """
        if recomputed_tensor is None:
            return

        assert isinstance(tensor, tf.Tensor)
        assert isinstance(recomputed_tensor, tf.Tensor)
        tensor.__setattr__(RECOMPUTED_TENSOR_ATTR_NAME, recomputed_tensor)
        # Custom gradient wrappers pass all inputs and outputs through an
        # IdentityN op. If this input is passed through an IdentityN, then
        # we would also like to attach the recomputed_tensor to the original
        # tensor that was created. Get that tensor
        orig_tensor = tensor
        while (
            orig_tensor.op.type == 'IdentityN'
            or orig_tensor.op.type == 'Identity'
        ):
            # Get port of tensor, then get IdentityN input at that port
            port_id = None
            for idx, out_tens in enumerate(orig_tensor.op.outputs):
                if out_tens == orig_tensor:
                    port_id = idx
                    break
            assert port_id is not None
            # Set attribute of the original tensor
            orig_tensor = orig_tensor.op.inputs[port_id]
            orig_tensor.__setattr__(
                RECOMPUTED_TENSOR_ATTR_NAME, recomputed_tensor
            )

    def _check_get_recomputed_tensor(self, tensor):
        """Checks the RECOMPUTED_TENSOR_ATTR_NAME on the output tensor.

        To communicate if a reversible layer's outputs have been
        recomputed during the backpropagation of a downstream layer, checks the
        RECOMPUTED_TENSOR_ATTR_NAME on the output tensor. If the recomputed
        tensor exists, uses this recomputed tensor. This is so that you do not
        have to save the original output tensor for backprop.

        Args:
            tensor (Tensor): The tensor to check for a recomputed version.
        Returns:
            Tensor: The recomputed version of the tensor if it exists.
                Otherwise, returns the original tensor.
        """
        assert isinstance(tensor, tf.Tensor)
        if hasattr(tensor, RECOMPUTED_TENSOR_ATTR_NAME):
            return tensor.__getattribute__(RECOMPUTED_TENSOR_ATTR_NAME)
        return tensor

    def _block_recompute_and_gradients(
        self, block, inputs, grad_outputs, variables, **kwargs
    ):
        """Recomputes the given block's outputs.

        Recomputes the given block's outputs and calculates the gradients to
        its inputs from the given output gradients.

        Args:
            block (function): The block to recompute.
            inputs (list of Tensors): The input tensors to process.
            grad_outputs (list of Tensors): The gradient of the outputs of the
                block.
            variables (list of Tensors): The variables that can be used in the
                block's forward pass. Also used to calculate the input
                gradients.
            kwargs (dict): Any keyword arguments that can be passed to the
                block during the recomputation. For example, ``side_inputs``.
        """
        # Stop gradients through the input for recompute
        # NOTE: Currently supports a single input tensor here
        inputs = [tf.stop_gradient(input) for input in inputs]
        # Stop gradients on any kwargs that are input tensors to the block
        side_inputs = []
        for key, value in kwargs.items():
            if isinstance(value, tf.Tensor):
                kwargs[key] = tf.stop_gradient(value)
                side_inputs.append(kwargs[key])
        # Recalculate the output of the block after the gradient is ready
        with tf.control_dependencies(grad_outputs):
            output = block(*inputs, **kwargs)

        # Ensure that the variables with which we're calculating gradients
        # are in the correct control flow context. This defeats a bug that
        # Tensorflow tracks the variables but not the specific tensors read
        # inside particular control flow contexts such as while-loops.
        # Further, we just recomputed a portion of the layer, so where
        # possible, we should use the tensors inside this gradient context,
        # so `tf.gradients` will calculate gradients with respect to them
        # rather than with respect to forward pass variable tensors
        variables = self._update_variables_for_context(grad_outputs, variables)

        # Calculate gradients to inputs, variables, and side_inputs given
        # the output, and grad_outputs
        grads = tf.gradients(
            output, inputs + variables + side_inputs, grad_outputs, name='',
        )

        grad_inputs = grads[: len(inputs)]
        grad_vars = grads[len(inputs) : len(inputs) + len(variables)]
        grad_kwargs = []

        return output, grad_inputs, grad_vars, grad_kwargs

    def _get_clean_grad_scope(self, graph):
        """Cleans the named scope for clean graphs.

        To keep the scopes organized with ``tf.custom_gradient``, this function
        strips off the ``IdentityN_grad`` tail of the scope. This is used below
        to set the scope of the recomputed blocks and their gradient
        calculations, so they are in the same part of the Tensorflow graph and
        easier to view and trace.

        Args:
            graph (tf.Graph): The graph that is currently executing.
        Returns:
            string: The scope name in which to place the recompute and gradient
            calculations.
        """
        name_scope_parts = graph.get_name_scope().split('/')
        # For now, it seems all gradients are calculated in the scope of
        # the IdentityN op added by `tf.custom_gradient`
        assert (
            'IdentityN_' in name_scope_parts[-1]
            and '_grad' in name_scope_parts[-1]
        )
        name_scope_parts[-1] = ''
        rev_grad_scope = '/'.join(name_scope_parts)
        return rev_grad_scope

    def _update_variables_for_context_v2(self, in_tensors, variable_tensors):
        # Get the current graph (context) and check if it is a while-loop
        curr_graph = in_tensors[0].graph
        if isinstance(curr_graph, WhileBodyFuncGraph):
            # This is inside a while-loop, so need to find the versions of
            # variable_tensors that have entered the loop. Get the graph's
            # tensor captures. Captures are pairs of tensors that represent
            # the tensor in the while body and the corresponding tensor in the
            # gradient body. Get the gradient version.
            graph_caps = list(curr_graph._captures.values())
            # Search the graph captures to find appropriate variable_tensors
            # that exist in the gradient body.
            for idx, variable_tens in enumerate(variable_tensors):
                if variable_tens.graph == curr_graph:
                    continue
                op_name_to_find = variable_tens.op.name
                possible_matches = []
                for graph_cap_pair in graph_caps:
                    if op_name_to_find in graph_cap_pair[1].name:
                        possible_matches.append(graph_cap_pair[1])
                if len(possible_matches) == 1:
                    variable_tensors[idx] = possible_matches[0]
                elif len(possible_matches) > 1:
                    raise NotImplementedError(
                        f'Don\'t know how to handle multiple possible matches'
                        ' for variable_tensor in \n'
                        '  var_tens: {variable_tens}\n'
                        '  possible_matches: {possible_matches}'
                    )
                else:
                    if not AbstractRecomputeWrapper.CtrlFlowWarnedOnce:
                        tf.compat.v1.logging.warning(
                            f'A variable tensor ({variable_tens}) was found '
                            'to be outside the currently executing graph, but '
                            'its corresponding tensor inside the graph could '
                            'not be found! The gradients for this variable '
                            'might not be calculated correctly.'
                        )
                        AbstractRecomputeWrapper.CtrlFlowWarnedOnce = True
        elif isinstance(curr_graph, (CondBranchFuncGraph, WhileCondFuncGraph)):
            # These are subgraph types known to exist in TF2.x, but this check
            # will not capture new subgraph types defined in TF. If future
            # types are defined, will need to check for them here.
            graph_type = type(curr_graph)
            raise NotImplementedError(
                f'Don\'t know how to handle graphs of type {graph_type}'
            )

        return variable_tensors

    def _update_variables_for_context(self, in_tensors, variable_tensors):
        """Checks the input tensors to see if they share common control flow
        context.

        Checks the ``in_tensors`` to see if they share a common control flow
        context. If so, then you should update the ``variable_tensors`` to the
        tensors that are read from the ``VariableOps``, and that are within the
        same control flow context as the ``in_tensors``. These tensors are
        required for `tf.gradients` to function correctly.

        Args:
            in_tensors (list of Tensors): The tensors used as input to
                calculate the gradients. These tensors define the control flow
                context in which the gradient calculation must be performed.
            variable_tensors (list of Tensors): Tensor handles for the variable
                ops that must be read for the gradient calculation.
        Returns:
            List of Tensors: The variable tensors updated to the versions
                inside the correct control flow context.
        """
        # Now, use the TF2.x version of this function
        return self._update_variables_for_context_v2(
            in_tensors, variable_tensors
        )

    @staticmethod
    def is_in_while_loop(graph=None):
        """
        Returns ``True`` if the specified, or current if unspecified, graph
        corresponds to a ``while`` loop in the forward, backward or cond
        graph.

        Returns:
            bool: ``True`` if the specified, or current if unspecified, graph
            corresponds to a ``while`` loop in the forward, backward or cond
            graph.
        """
        g = graph or tf.compat.v1.get_default_graph()
        if isinstance(
            g, (WhileBodyFuncGraph, CondBranchFuncGraph, WhileCondFuncGraph,)
        ):
            return True
        # Custom gradients may mean user has not used the default while
        # loop grad class, but if the graph has a _forward_graph attribute,
        # then we can be sure that this is a while loop backward pass graph.
        return hasattr(g, '_forward_graph')

    def _update_variables_for_context_v1(self, in_tensors, variable_tensors):
        # Check if the variables need to be read within a control flow context
        # where the other gradient tensors are created:
        no_control_context = False
        control_flow_context = None
        for in_tensor in in_tensors:
            if no_control_context:
                # Error checking: If the first in_tensor tensor is not within
                # a control flow context, then none of them should be
                assert in_tensor.op._control_flow_context is None
            if control_flow_context is None:
                if in_tensor.op._control_flow_context is not None:
                    control_flow_context = in_tensor.op._control_flow_context
                else:
                    no_control_context = True
            else:
                # Error checking: If one in_tensor tensor is within a control
                # flow context, then they should all be in the same one
                assert (
                    in_tensor.op._control_flow_context == control_flow_context
                )

        if (
            control_flow_context is not None
            and not control_flow_context.IsWhileContext()
        ):
            if control_flow_context.IsCondContext():
                raise NotImplementedError(
                    'Must implement non-while context handling!'
                )
            assert control_flow_context.IsXLAContext()
            control_flow_context = None

        if control_flow_context is not None:
            # Find appropriate variables used inside control flow context
            updated_vars = []
            for variable in variable_tensors:
                found_var_ctx_enter = False
                for var_downstream_op in variable.op.outputs[0].consumers():
                    if (
                        not var_downstream_op.type == 'Enter'
                        and not var_downstream_op.type == 'ReadVariableOp'
                    ):
                        continue
                    if (
                        var_downstream_op._control_flow_context
                        == control_flow_context
                    ):
                        found_var_ctx_enter = True
                        updated_vars.append(var_downstream_op.outputs[0])
                        break
                if not found_var_ctx_enter:
                    if not AbstractRecomputeWrapper.CtrlFlowWarnedOnce:
                        tf.compat.v1.logging.warning(
                            f'A variable ({variable}) was found outside a '
                            'control flow context, but its associated Enter '
                            'or ReadVariableOp inside the control flow '
                            'context ({control_flow_context}) could not be '
                            'found! The gradients for this variable might '
                            'not be calculated correctly.'
                        )
                        AbstractRecomputeWrapper.CtrlFlowWarnedOnce = True
                    updated_vars.append(variable)
        else:
            updated_vars = variable_tensors

        return updated_vars

    @abstractmethod
    def call(self, *args, **kwargs):
        """The call function for the layers that use recomputation during
        backward phase.

        This function is wrapped by the ``__call__`` function of this abstract
        recompute wrapper, and it must be overridden by a child class to
        implement the forward computation of the layer.
        """

    @abstractmethod
    def _gradient(self, grad_ys, rev_ys, **grad_kwargs):
        """The abstract gradient function for layers that use recomputation
        during the backward phase.

        This function is wrapped by the ``__call__`` function of this abstract
        class, and it must be overridden by the child class.

        With this ``_gradient`` definition, the child class can handle tensors
        in the following ways:

            - If the layer is to recompute the activations using the forward
            pass tensors, it should save those tensors as instance variables
            during its overridden ``call`` function and reference them during
            the overridden ``_gradient`` function.
            - For the reversible layers, this wrapper handles the tensors
            recomputed during the backward phase that need to be shuttled
            between reversible layers. Such recomputed tensors are passed to
            the child class using the "reversible Y's" parameter here
            (``rev_ys``). This structure permits composing multiple reversible
            layers together.

        Args:
            grad_ys (tuple of Tensors): The gradients of tensors output by the
                layer's forward pass. The child class must use these
                ``grad_ys`` to calculate the gradients of its inputs and
                variables, similar to the standard gradient calculations).
            rev_ys (tuple of Tensors): The recomputed versions of the output
                tensors (or just the output tensors if no such recomputed
                tensors exist) recovered by this wrapper. The child can use
                these to recompute further tensors for gradients. If so, the
                child must return the recomputed inputs.
            grad_kwargs (dict): Any keyword arguments to be passed to the
                gradient calculation.

        Returns:
            tuples: A tuple of Tensor tuples: grad_xs, rev_xs, and grad_vars.

                - grad_xs: Must be the gradients to pass to the prior layers
                (standard backpropagation).
                - rev_xs: Recomputed tensors to pass to the prior layers that
                might use them for recomputation.
                - grad_vars: Gradients to the layer's variables (standard
                backpropagation).
        """

    def __call__(self, *args, **kwargs):
        """Recompute layer wrapper call.

        This function uses a Tensorflow custom gradient to override and manage
        the input, output, recomputed activation, and gradient tensors for any
        wrapped layer.

        This function has a custom backward pass for layers that do some
        recomputation during the backward phase. The ``_gradient_wrap``
        function can capture tensors either from the forward pass or recomputed
        during the backward pass of other layers. These captured tensors can be
        used during the recomputation and gradient calculations in this current
        layer.

        The ``_gradient_wrap`` can aid with the handling of the recomputed
        tensors for either:

            - Sequential reversible layers or
            - Reversible layers inside the ``while`` loops.
            - For the sequential layers, recomputed tensors are saved and
            communicated between the layers using the pair of functions:

                - ``_set_recomputed_tensor``: Attaches a recomputed tensor to
                the original layer's output tensor, and
                - ``_check_get_recomputed_tensor``: Gets get the recomputed
                tensor if it exists.

            - For the recompute layers in a loop, recomputed tensors are saved\
            and restored following a below (hacky) process:

                - First, add copies of the forward loop's inputs as inputs. As
                a result, Tensorflow will add the gradients for these extra
                tensors during the gradient calculation.
                - Then, add this custom gradient wrapper to carefully strip off
                the extra gradient tensors and override them with recomputed
                tensors. Note that in the first backprop iteration, the
                "recomputed" tensor is the saved forward pass output.

        The ``_gradient_wrap`` calls the ``_gradient`` function of the child
        class, which is abstract in this class, and the child class must define
        how to calculate the recomputed tensors and custom gradients.
        """

        @tf.custom_gradient
        def _forward_wrap(*flat_forward_args):
            # Reconstruct args shape
            args = nest.pack_sequence_as(
                self._args_for_shape, flat_forward_args
            )

            # Compute forward pass. Here, the args are shaped based on how
            # they were passed to `__call__`, so the user sees the same
            # data structure in the child class call. Track variables to see
            # if any variables get defined in the forward pass subgraph.
            with tf.GradientTape() as tape:
                outputs = self.call(*args, **kwargs)
            variables_in_tape = frozenset(
                [v.ref() for v in tape.watched_variables()]
            ) - frozenset(v.ref() for v in nest.flatten(args))

            forward_pass_contains_variables = False
            if len(variables_in_tape) > 0:
                forward_pass_contains_variables = True

            # Store the outputs, so they can be used to reconstruct the shape
            # of the gradients to be passed to the child class
            if not isinstance(outputs, tuple):
                self._outputs_for_shape = (outputs,)
            else:
                self._outputs_for_shape = outputs

            def _gradient_wrap(*grad_ys, **grad_kwargs):
                """

                Args:
                    grad_ys (tuple of Tensors): The gradients of the tensors
                        generated as outputs of the layer's forward pass. Note
                        that if this is a reversible layer inside a ``while``
                        loop, then some of these gradient tensors will just be
                        overridden to be used as slots for recomputed tensors.
                    grad_kwargs (dict): Any keyword args that should be passed
                        to the child's ``_gradient`` function.
                """

                # First, if we happen to be inside a subgraph of the model
                # that requires capturing tensors from the outer graph (e.g.,
                # a while loop), get the tensors representing the layer's
                # variables from inside the subgraph. This helper seeks out
                # and finds those tensors.
                if grad_kwargs.get('variables', False):
                    grad_kwargs[
                        'variables'
                    ] = self._update_variables_for_context(
                        flat_forward_args, grad_kwargs['variables']
                    )

                # Strip off IdentityN from grad scope
                graph = tf.compat.v1.get_default_graph()
                rev_grad_scope = self._get_clean_grad_scope(graph)
                with graph.name_scope(rev_grad_scope):
                    rev_ys = []
                    for out_tens in nest.flatten(outputs):
                        # Check if a downstream reversible layer recomputed
                        # outputs. If so, get those outputs for recomputation
                        rev_ys.append(
                            self._check_get_recomputed_tensor(out_tens)
                        )

                    # Call the child class's gradient function
                    grad_ys = nest.pack_sequence_as(
                        self._outputs_for_shape, grad_ys
                    )
                    rev_ys = nest.pack_sequence_as(
                        self._outputs_for_shape, rev_ys
                    )
                    grad_outputs = self._gradient(
                        grad_ys, rev_ys, **grad_kwargs,
                    )
                    if len(grad_outputs) == 3:
                        grad_xs, rev_xs, grad_vars = grad_outputs
                    elif len(grad_outputs) == 2:
                        grad_xs, rev_xs = grad_outputs
                        grad_vars = []
                    else:
                        raise ValueError(
                            f'AbstractRecomputeWrapper got wrong number of '
                            'outputs ({len(grad_outputs)}) from '
                            'self._gradients in child class of type: '
                            '{type(self)}'
                        )

                    # Attach recomputed inputs to input tensors
                    rev_xs = nest.flatten(rev_xs)
                    for in_x, rev_x in zip(flat_forward_args, rev_xs):
                        self._set_recomputed_tensor(in_x, rev_x)

                    # Flatten grads to be returned
                    grad_xs = nest.flatten(grad_xs)

                    return grad_xs, grad_vars

            # This is messy, because `tf.custom_gradients` is messy... If no
            # variables are instantiated inside the forward pass of the block,
            # then `tf.custom_gradients` will expect the gradient function
            # header NOT accept `grad_kwargs`. The only way to ensure this is
            # to define wrappers for either function header type and detect
            # which one must be used.
            def _gradient_wrap_vars(*grad_ys, **grad_kwargs):
                return _gradient_wrap(*grad_ys, **grad_kwargs)

            def _gradient_wrap_no_vars(*grad_ys):
                grad_xs, _ = _gradient_wrap(*grad_ys)
                return grad_xs

            if forward_pass_contains_variables:
                _gradient_fn = _gradient_wrap_vars
            else:
                _gradient_fn = _gradient_wrap_no_vars

            return outputs, _gradient_fn

        # Custom gradient is very picky about handling arguments and gradients
        # tensors that are in nested tuples or lists. To make the API more
        # user-friendly, handle flattening and reconstructing shape here.
        self._args_for_shape = args
        flat_args = nest.flatten(args)

        return _forward_wrap(*flat_args)
