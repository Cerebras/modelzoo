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
Config classes of Optimizer Based Configs

"""
import copy
from dataclasses import asdict

# pylint: disable=wildcard-import
from cerebras.modelzoo.config_manager.config_classes.base.base_config import *
from cerebras.modelzoo.config_manager.config_validators import LossScalingFactor
from cerebras.pytorch.optim import (
    configure_optimizer_params,
    configure_scheduler_params,
)


@dataclass
class OptimizerConfig(BaseConfig):
    optimizer_type: str = required
    """
    Optimizer to be used. 
    See supported optimizers - https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-ops/supported-pytorch-optimizers.html)
    """
    weight_decay: float = 0.0
    log_summaries: bool = False
    """
    Flag to log per layer gradient norm in Tensorboard.
    Defaults to False
    """
    loss_scaling_factor: Union[str, float] = config_field(
        default=1.0,
        constraint=LossScalingFactor,
    )
    learning_rate: Optional[Union[float, List[dict]]] = None
    """
    Learning rate scheduler to be used. See [supported LR schedulers]
    (https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-ops/
    supported-pt-learning-rate-schedulers.html).
    optional, defaults to None)
    """
    optim_params: Optional[dict] = None
    """
    A dictionary created internally that holds the optimizer specific params.
    The params that are part of specific optimizers get collapsed into this dictionary
    and validated against optimizer class signature.
    Yaml/config class object can pass these arguements as part of optimizer directly.
    They are all collapsed under optim_params internally
    """
    max_gradient_norm: Optional[float] = None
    """
    Max norm of the gradients for learnable parameters. 
    Used for gradient clipping. Default=None
    """
    adjust_learning_rate: Optional[dict] = None

    # Custom init for optimizer, where we want to capture all fixed optimizer params as members.
    # optim params is a dict that is populated by checking all additional params supplied to us.
    # These are optimizer specific and we use signature of that optimizer to validate these.
    def __init__(self, **kwargs):
        for field_name, field_type in self.__annotations__.items():
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
        self.optim_params = {key: value for key, value in kwargs.items()}
        if self.adjust_learning_rate is None:
            self.adjust_learning_rate = {}
        super().__init__()

    def __post_init__(self):
        # convert to a List[LearningRateConfig] if its only a float value
        float_lr = None
        if isinstance(self.learning_rate, float):
            float_lr = self.learning_rate
            self.learning_rate = [
                {"scheduler": "constant", "learning_rate": self.learning_rate}
            ]

        if isinstance(self.learning_rate, list):
            for lr in self.learning_rate:
                lr_params = copy.deepcopy(lr)
                # Main scheduler isnt expected for signature checks
                lr_params.pop("main_scheduler", None)
                configure_scheduler_params(lr_params)
        elif isinstance(self.learning_rate, dict):
            lr_params = copy.deepcopy(self.learning_rate)
            lr_params.pop("main_scheduler", None)
            configure_scheduler_params(lr_params)
        if float_lr:
            self.learning_rate = float_lr

        optimizer_param_dict = copy.deepcopy(asdict(self))
        # We get a bunch of args in optimizer config but arent used for optim signature
        # As this function is called for signature validation also, these may not be stripped out
        # These arent optimizer specific and we dont need to throw error based on sign for these
        include_list = [
            'learning_rate',
            'optim_params',
        ]
        optimizer_param_dict = {
            k: v for k, v in optimizer_param_dict.items() if k in include_list
        }
        configure_optimizer_params(
            optimizer_type=self.optimizer_type, kwargs=optimizer_param_dict
        )
        super().__post_init__()
