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

import torch.nn as nn


class RepeatedLayer(nn.Module):
    r"""RepeatedLayer is a single underlying layer that is applied N times.
        This functionality can be used for cross-layer parameter sharing.

    Args:
        layer: the instance of the layer that should be repeated (required). The
               output of the layer will be treated as the input (first argument
               of the forward function) during each repetition.
        num_repetitions: the number of times the layer should be repeated (required).
        map_args (optional): a function whose arguments are the positional arguments
                            (args) of the forward fn, the keyworded arguments (kwargs)
                            of the forward fn, and the output of the most recent
                            call to the layer that is being repeated. The function
                            should return  the positional and keyworded arguments
                            that should be fed to the layer's forward fn (in order
                            to apply the next repetition). If `map_args` is not
                            provided, the default behavior is to replace 0th
                            positional argument with the output of the last
                            repetition if the output is not None.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = RepeatedLayer(decoder_layer, 4)
        >>> tgt = torch.rand(32, 20, 512)
        >>> memory = torch.rand(32, 10, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, layer, num_repetitions, map_args=None):
        super(RepeatedLayer, self).__init__()
        self.layer = layer
        self.num_repetitions = num_repetitions

        if map_args is None:

            def map_args(args, kwargs, output):
                first_arg = output if output is not None else args[0]
                positional_args = (first_arg,) + args[1:]
                return positional_args, kwargs

        self.map_args = map_args
        # No need to call _reset_parameters during init since `layer` will have
        # already called _reset_parameters during its init.

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, *args, **kwargs):
        output = None

        for i in range(self.num_repetitions):
            positional_args, key_args = self.map_args(args, kwargs, output)
            output = self.layer(*positional_args, **key_args)

        return output
