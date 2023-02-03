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

from modelzoo import CSOFT_PACKAGE, CSoftPackage

if CSOFT_PACKAGE == CSoftPackage.SRC:
    from cerebras.tf.cs_estimator_spec import CSEstimatorSpec as EstimatorSpec
elif CSOFT_PACKAGE == CSoftPackage.WHEEL:
    from cerebras_tensorflow.cs_estimator_spec import (
        CSEstimatorSpec as EstimatorSpec,
    )
elif CSOFT_PACKAGE == CSoftPackage.NONE:
    from tensorflow_estimator.python.estimator.model_fn import EstimatorSpec
else:
    assert False, f"Invalid value for `CSOFT_PACKAGE`: {CSOFT_PACKAGE}"


class CSEstimatorSpec(EstimatorSpec):
    def __new__(cls, mode, host_call=None, **kwargs):
        if EstimatorSpec.__name__ == "CSEstimatorSpec":
            kwargs["host_call"] = host_call

        instance = super().__new__(cls, mode, **kwargs)

        if (
            EstimatorSpec.__name__ != "CSEstimatorSpec"
            and host_call is not None
        ):
            setattr(instance, "host_call", host_call)

        return instance
