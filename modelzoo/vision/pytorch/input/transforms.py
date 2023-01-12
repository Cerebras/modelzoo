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

from inspect import getfullargspec, signature

__all__ = [
    "LambdaWithParam",
]


class LambdaWithParam(object):
    def __init__(self, lambd, *args, **kwargs):
        assert callable(lambd), (
            repr(type(lambd).__name__) + " object is not callable"
        )
        self.lambd = lambd
        self.args = args
        self.kwargs = kwargs
        ll_sig = getfullargspec(lambd)
        if not ll_sig.varargs or not ll_sig.varkw:
            raise TypeError(
                "User-defined lambda transform function must have signature: "
                "function(img, positional args, *args, **kwargs). Instead, "
                f"got function{str(signature(lambd))}."
            )

    def __call__(self, img):
        return self.lambd(img, *self.args, **self.kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(args={0}, kwargs={1})'.format(
            self.args, self.kwargs
        )
