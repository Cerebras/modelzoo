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

""" CSEstimator Wrapper

Wrapper for allowing users with different environments to run the
CSEstimator in Modelzoo scripts.
"""

import inspect

import tensorflow as tf

from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.tf.estimator.utils import (
    host_call_to_eval_metric_ops,
    validate_host_call,
)

if CSOFT_PACKAGE == CSoftPackage.WHEEL:
    from cerebras_tensorflow.cs_estimator_app import (
        CerebrasAppEstimator as estimator,
    )
elif CSOFT_PACKAGE == CSoftPackage.NONE:
    from tensorflow_estimator.python.estimator.estimator import (
        Estimator as estimator,
    )
else:
    assert False, f"Invalid value for `CSOFT_PACKAGE {CSOFT_PACKAGE}"


class CerebrasEstimator(estimator):
    """ Wrapper class for Modelzoo Estimator. """

    def __init__(
        self,
        model_fn,
        model_dir=None,
        compile_dir=None,
        config=None,
        params=None,
        warm_start_from=None,
    ):
        kwargs = dict()
        self._orig_model_fn = model_fn
        if CSOFT_PACKAGE is not CSoftPackage.NONE:
            kwargs["compile_dir"] = compile_dir
        else:
            model_fn = self._wrapper_model_fn
        self.__class__._assert_members_are_not_overridden = lambda _: None
        super(CerebrasEstimator, self).__init__(
            model_fn=model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from,
            **kwargs,
        )

    def compile(
        self, input_fn, validate_only=False, mode=tf.estimator.ModeKeys.TRAIN
    ):
        if CSOFT_PACKAGE is not CSoftPackage.NONE:
            return super().compile(input_fn, validate_only, mode)

        tf.compat.v1.logging.warning(
            "Running outside the Cerebras Container, so compile will not take"
            " place. Please use Cerebras Container to compile for the Cerebras System."
        )
        return None

    def train(
        self,
        input_fn,
        hooks=None,
        steps=None,
        max_steps=None,
        saving_listeners=None,
        use_cs=True,
        sparsifier=None,
    ):
        kwargs = dict()
        if CSOFT_PACKAGE is not CSoftPackage.NONE:
            kwargs["use_cs"] = use_cs
        if CSOFT_PACKAGE is not CSoftPackage.WHEEL:
            if sparsifier is not None:
                tf.compat.v1.logging.warning(
                    "Running outside Cerebras container, so sparse runs and"
                    " sparsifiers are not supported. Please use Cerebras"
                    " Container + weight streaming. Defaulting "
                    "to no op for sparsifier."
                )

        return super(CerebrasEstimator, self).train(
            input_fn=input_fn,
            hooks=hooks,
            steps=steps,
            max_steps=max_steps,
            saving_listeners=saving_listeners,
            **kwargs,
        )

    def evaluate(
        self,
        input_fn,
        steps=None,
        hooks=None,
        checkpoint_path=None,
        name=None,
        use_cs=False,
    ):
        kwargs = dict()
        if CSOFT_PACKAGE is not CSoftPackage.NONE:
            kwargs["use_cs"] = use_cs

        return super(CerebrasEstimator, self).evaluate(
            input_fn=input_fn,
            steps=steps,
            hooks=hooks,
            checkpoint_path=checkpoint_path,
            name=name,
            **kwargs,
        )

    def predict(
        self,
        input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True,
        num_samples=None,
        use_cs=False,
    ):
        kwargs = dict()
        if CSOFT_PACKAGE is not CSoftPackage.NONE:
            kwargs["use_cs"] = use_cs
            kwargs["num_samples"] = num_samples

        return super(CerebrasEstimator, self).predict(
            input_fn=input_fn,
            predict_keys=predict_keys,
            hooks=hooks,
            checkpoint_path=checkpoint_path,
            yield_single_examples=yield_single_examples,
            **kwargs,
        )

    def _wrapper_model_fn(self, features, labels, mode, params):
        """
        Wrap model_fn to convert host_call in the returned EstimatorSpec into
        eval_metric_ops, be used with TF Estimator.
        """
        # Should match with cerebras estimator
        spec = self._orig_model_fn(
            features=features, labels=labels, mode=mode, params=params
        )

        # Only one must be specified
        host_call = getattr(spec, "host_call", None)
        eval_metric_ops = getattr(spec, "eval_metric_ops", None)

        if host_call and eval_metric_ops:
            raise ValueError(
                "Please specify only one of `host_call` or `eval_metric_ops`, "
                "not both."
            )
        elif host_call:
            # Validate and convert host_call to eval_metric_ops
            host_call = validate_host_call(host_call)
            if host_call:
                eval_metric_ops = host_call_to_eval_metric_ops(host_call)
            else:
                eval_metric_ops = None

        # Create a new EstimatorSpec with host_call turned into eval_metric_ops
        spec_args = inspect.getfullargspec(tf.estimator.EstimatorSpec)
        new_spec_args = {}
        for arg in spec_args.args:
            if arg in ["eval_metric_ops"]:
                new_spec_args[arg] = eval_metric_ops
            elif arg in ["cls"]:
                pass
            else:
                new_spec_args[arg] = getattr(spec, arg)
        new_spec = tf.estimator.EstimatorSpec(**new_spec_args)

        return new_spec
