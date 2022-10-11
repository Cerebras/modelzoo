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

import inspect

import tensorflow as tf

from modelzoo.common.tf.estimator.utils import (
    host_call_to_eval_metric_ops,
    validate_host_call,
)
from modelzoo.common.tf.run_utils import ExecutionMode, get_execution_mode

try:
    # We have two versions of estimator internally depending on whether we are
    # running weight streaming or pipeline. Being able to import tfwse is a
    # good canary for this as it's only possible in the weight streaming
    # environment
    try:
        import tfwse  # noqa
        from cerebras.tf.ws.cs_estimator_ws import (
            CerebrasEstimator as estimator,
        )
    except ImportError:
        from cerebras.tf.cs_estimator import CerebrasEstimator as estimator
except ImportError:
    from tensorflow_estimator.python.estimator.estimator import (
        Estimator as estimator,
    )


class CerebrasEstimator(estimator):
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
        if estimator.__name__ == "CerebrasEstimator":
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
        if estimator.__name__ == "CerebrasEstimator":
            super().compile(input_fn, validate_only, mode)
        else:
            tf.compat.v1.logging.warning(
                "Running outside the Cerebras Container, so compile will not take"
                " place. Please use Cerebras Container to compile for the Cerebras System."
            )

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
        if estimator.__name__ == "CerebrasEstimator":
            kwargs["use_cs"] = use_cs

            if get_execution_mode() == ExecutionMode.WeightStreaming:
                kwargs["sparsifier"] = sparsifier

        if sparsifier is not None and "sparsifier" not in kwargs:
            tf.compat.v1.logging.warning(
                "Running outside weight streaming, so sparse runs and sparsifiers are not"
                "supported. Please use Cerebras Container + weight streaming. Defaulting "
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
        if estimator.__name__ == "CerebrasEstimator":
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
        if estimator.__name__ == "CerebrasEstimator":
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
        spec_args = inspect.getargspec(tf.estimator.EstimatorSpec)
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
