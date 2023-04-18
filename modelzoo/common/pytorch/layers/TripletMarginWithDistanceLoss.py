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

from modelzoo.common.pytorch.layers.utils import autogen_loss


# A dummy class to wrap nn.TripletMarginWithDistanceLoss loss with Autogen support. To
# enable the Autogen support, add a keyword argument `use_autogen=True`
# when initializing the loss function.
# For example:
# loss = TripletMarginWithDistanceLoss(..., use_autogen=True)
@autogen_loss
class TripletMarginWithDistanceLoss(nn.TripletMarginWithDistanceLoss):
    pass
