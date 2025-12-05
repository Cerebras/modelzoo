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

import logging
import re
from inspect import getfullargspec
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
from pydantic import Field, field_validator
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.config import BaseConfig, ModelConfig
from cerebras.modelzoo.models.multimodal.multimodal_utils import freeze_modules
from cerebras.modelzoo.models.vision.generic_image_encoders.heads.DinoHead import (
    DinoHeadConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.heads.IJEPAPredictor import (
    IJEPAPredictorConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.losses.DinoDistillationLoss import (
    DinoDistillationLossConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.losses.iBOTPatchLoss import (
    iBOTPatchLossConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.losses.KoLeoLoss import (
    KoLeoLossConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.losses.MaskedSmoothL1Loss import (
    MaskedSmoothL1LossConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.trunks.IJEPAContextEncoder import (
    IJEPAContextEncoderConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.trunks.IJEPATargetEncoder import (
    IJEPATargetEncoderConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.trunks.MultiImageViTModel import (
    MultiImageViTModelConfig,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.utils.ema import (
    EMAWrapper,
    create_momentum_scheduler,
)
from cerebras.modelzoo.trainer import summarize_scalar


class OutputList:
    def __init__(self, input_list):
        assert self.is_all_lists(input_list), "Inconsistent input list"
        self._output_list = input_list

    @property
    def output_list(self):
        return self._output_list

    @output_list.setter
    def output_list(self, val):
        """Sets the value for outermost list."""
        input_list, index = val
        assert isinstance(index, int), "index should be integer value"
        self.output_list[index] = input_list

    def __call__(self, *index):
        out = self.output_list[index[0]]
        for idx in index[1:]:
            out = out[idx]
        return out

    def is_all_lists(self, input_list):
        # TODO: This needs more testing
        if len(input_list) == 0:
            return True
        elif not isinstance(input_list[0], list):
            # if first element not list, all else should not be lists
            return all([not isinstance(x, list) for x in input_list])
        else:
            # if first element is list, all else should be lists
            return all([self.is_all_lists(x) for x in input_list])

    def __repr__(self):
        return f"{self.__class__.__name__}(list={self.output_list}"

    def __iter__(self):
        return iter(self._output_list)


# Note: If not nn.Module, does not show up in state_dict
class SSLComponent(nn.Module):
    def __init__(self, output=None, model=None):
        super().__init__()
        self.output = output
        self.model = model

    def __repr__(self):
        s = f"{self.__class__.__name__}(model={self.model}, \noutput={self.output})"
        return s


class InputKey:
    def __init__(self, batch):
        self._batch = batch

    def __call__(self, key):
        return self._batch[key]

    def __repr__(self):
        s = f"{self.__class__.__name__}(keys={list(self._batch.keys())})"
        return s


class CopyWeightsConfig(BaseConfig):
    source: str = ...
    "string representing source model whose weights should be copied to target"

    target: str = ...
    "string representing target model whose weights should be updated using a copy of weights from source"


class EMAConfig(BaseConfig):
    source: str = ...
    "source model whose weights are used for Exponential moving average"

    target: str = ...
    "target model whose weights are updated using Exponential moving average using"
    "target_weight = decay * target_weight + (1-decay)*source_weight"

    scheduler_name: Literal["linear", "cosine"] = "linear"
    "scheduler that controls the moving average"

    scheduler_params: dict = {}
    "Params to initialize the scheduler"


class MultiForwardModelConfig(BaseConfig):
    forward_args: List[List[Union[str, None]]] = []
    stop_grad: bool = False


class ImageModelTrunk(MultiForwardModelConfig):
    image_model: Annotated[
        Union[
            IJEPAContextEncoderConfig,
            IJEPATargetEncoderConfig,
            MultiImageViTModelConfig,
        ],
        Field(discriminator=ModelConfig.discriminator),
    ]


class HeadModel(MultiForwardModelConfig):
    head_model: Annotated[
        Union[
            DinoHeadConfig,
            IJEPAPredictorConfig,
        ],
        Field(discriminator=ModelConfig.discriminator),
    ]


class Loss(MultiForwardModelConfig):
    loss: Annotated[
        Union[
            DinoDistillationLossConfig,
            KoLeoLossConfig,
            MaskedSmoothL1LossConfig,
            iBOTPatchLossConfig,
        ],
        Field(discriminator=ModelConfig.discriminator),
    ]

    loss_weight: float = 1.0
    "scalar multiplying the loss value computed "


class GenericImageEncodersModelConfig(ModelConfig):
    name: Literal["generic_image_encoders"]
    "Name of the model. Must be set to `generic_image_encoders`."

    image_model_trunks: List[ImageModelTrunk] = ...
    "List of image_model_trunks."

    heads: List[HeadModel] = ...
    "List of heads."

    losses: List[Loss] = ...
    "List of losses."

    copy_init_weights: Optional[List[CopyWeightsConfig]] = None
    "List of params that control weight initialization. Copy weights from source to target model parameters."

    ema: Optional[List[EMAConfig]] = None
    "List of params for Exponential Moving Average of weights."

    freeze: Optional[List[str]] = None
    "List of regex strings used to freeze specific layers."

    @field_validator("copy_init_weights", mode="after")
    def validate_copy_init_weights(cls, copy_init_weights):
        if copy_init_weights is not None:
            logging.warning(
                "copy_init_weights is not None. "
                "Please set this to `None` when initializing from a checkpoint."
            )
        return copy_init_weights

    def validate_forward_args(self, ssl_transform_valid_keys):
        str_pattern = re.compile(r'(\w+)\.(\w+)\([\']*(.*?)[\']*\)')

        # Number of individual models

        _comps = {
            "image_model_trunks": "image_model",
            "heads": "head_model",
            "losses": "loss",
        }
        expected_dict = {
            "num_models": {k: 0 for k in _comps},
            "num_fwd_pass_inputs": {},
            "num_fwd_passes": {},
        }

        for cmp, subcmp in _comps.items():
            _models = getattr(self, cmp)
            if _models is None:
                continue

            expected_dict["num_models"][cmp] = len(_models)
            expected_dict["num_fwd_passes"][cmp] = []

            for idx, model in enumerate(_models):
                expected_dict["num_fwd_passes"][cmp].append(
                    len(model.forward_args)
                )
                model_subcmp = getattr(model, subcmp)
                if (
                    model_subcmp.name
                    not in expected_dict["num_fwd_pass_inputs"]
                ):
                    # get number of inputs in `.forward` fcn
                    num_inp_args = len(
                        getfullargspec(model_subcmp.__model_cls__.forward).args
                    )
                    num_inp_args -= 1  # -1 for avoiding to count `self`

                    expected_dict["num_fwd_pass_inputs"][
                        model_subcmp.name
                    ] = num_inp_args

        # number of inputs in forward args
        for component, subcomponent in _comps.items():
            comp_params = getattr(self, component)
            # List[<ConfigClass>] where <ConfigClass> is
            # as defined in `GenericImageEncodersModelConfig`

            if comp_params is None:
                continue

            for cmp_idx, single_comp_param in enumerate(comp_params):
                fwd_args = single_comp_param.forward_args  # List[List[str]]
                comp_name = getattr(single_comp_param, subcomponent).name

                for fwd_list in fwd_args:
                    # Check if correct number of forward pass arguments being passed
                    if (
                        len(fwd_list)
                        != expected_dict["num_fwd_pass_inputs"][comp_name]
                    ):
                        raise ValueError(
                            f"{comp_name} only accepts "
                            f"{expected_dict['num_fwd_pass_inputs'][comp_name]} "
                            f"(excluding self) in its forward pass. "
                            f"Got {fwd_list} which is of length {len(fwd_list)}"
                        )

                    for val in fwd_list:
                        grp0_valid = [
                            "image_model_trunks",
                            "heads",
                            "losses",
                            "ssl_transform",
                        ]
                        if not isinstance(val, str) or not any(
                            x in val for x in grp0_valid
                        ):
                            # We cannot really check for values which are say None or
                            # some other values which are passed as inputs
                            continue

                        grps = re.search(str_pattern, val).groups()

                        # Check if correct number of patterns
                        if len(grps) != 3:
                            raise ValueError(
                                f"Invalid value {val}. A value in `forward_args` "
                                f"should be pattern`<part1>.<part2>()` where `part1` is one of"
                                f" {'image_model_trunks', 'heads', 'losses', 'ssl_transform'} and "
                                f"`part2` is one of {'output', 'model'}"
                            )

                        # Check if correct value in pattern

                        if grps[0] not in grp0_valid:
                            raise ValueError(
                                f"Invalid value {grps[0]} in {val}. The valid ones are {grp0_valid}"
                            )

                        grp1_valid = ["model", "output"]
                        if grps[1] not in grp1_valid:
                            raise ValueError(
                                f"Invalid value {grps[1]} in {val}. The valid ones are {grp1_valid}"
                            )

                        # ' is also captured in regex, removing it
                        # ALL output structure:
                        #    (a)list of (b)list of (c)lists
                        #    len(a) = len(category) where category one of `image_model_trunks`, `heads`, `losses`
                        #    len(b) = len(forward_args)
                        #    len(c) = num_outputs from model per forward pass
                        # For example:
                        #  output = image_model_trunks.model[i].forward(image_model_trunks.forward_args[j])
                        # "image_model_trunks.output(i,j,k)" = output[k]
                        #
                        #  In case of this yaml: `"image_model_trunks.output(1,0,0)"`
                        #  `"image_model_trunks.output(1,0,0)"`
                        #  output at index 0
                        # of the IJEPATargetEncoder(=0)
                        # with forward_args[0] passed as inputs to forward fcn
                        grp2_val = grps[2].replace("'", "")
                        if "," in grp2_val:
                            grp2_val = grp2_val.split(",")
                            grp2_val = [int(x.strip()) for x in grp2_val]

                            if len(grp2_val) != 3:
                                raise ValueError(
                                    f"Invalid length of indices for {grp2_val}. Valid is `3`."
                                )

                            if (
                                grp2_val[0]
                                >= expected_dict["num_models"][grps[0]]
                            ):
                                raise ValueError(
                                    f"Index at position 0 of {grps[2]} which indexes into "
                                    f"`{grps[0]}` is greater than len "
                                    f"{expected_dict['num_models'][grps[0]]}"
                                )

                            if (
                                grp2_val[1]
                                >= expected_dict["num_fwd_passes"][grps[0]][
                                    grp2_val[0]
                                ]
                            ):
                                raise ValueError(
                                    f"Index at position 1 of {grps[2]} which indexes into "
                                    f"`{grps[0]}` is greater than len "
                                    f"{expected_dict['num_fwd_passes'][grps[0]][grp2_val[0]]}"
                                )

                        else:
                            # str for ssl_transform
                            if grp2_val not in ssl_transform_valid_keys:
                                raise ValueError(
                                    f"Invalid {grp2_val} in {val}. "
                                    f"Valid are {ssl_transform_valid_keys}"
                                )

                            # Additional contraints check
                            if (
                                grps[0] != "ssl_transform"
                                or grps[1] != "output"
                            ):
                                valid_poss = [
                                    f"ssl_transform.output('{out_key}')"
                                    for out_key in ssl_transform_valid_keys
                                ]
                                raise ValueError(
                                    f"Invalid {val}. Valid are {valid_poss}"
                                )


class BaseSSLArch(nn.Module):
    def __init__(self, config: GenericImageEncodersModelConfig):
        """
        ALL output structure: (three level nested list)
        (a)list of (b)list of (c)lists
        len(a) = len(category) where category one of `image_model_trunks`, `heads`, `losses`
        len(b) = len(forward_args)
        len(c) = num_outputs from model per forward pass
        For example:
        output = image_model_trunks.model[i].forward(image_model_trunks.forward_args[j])
        "image_model_trunks.output(i,j,k)" = output[k].
        """
        if isinstance(config, dict):
            config = GenericImageEncodersModelConfig(**config)

        super().__init__()

        self.config = config
        self._component_param_keys = [
            "ssl_transform",
            "image_model_trunks",
            "heads",
            "losses",
        ]
        self.loss_weights = None

        # Create SSLTransform obj with `output` accessible key
        self.ssl_transform = SSLComponent()

        self.image_model_trunks = SSLComponent(
            model=nn.ModuleList(
                # Construct model object using config
                image_model_trunk.image_model()
                for image_model_trunk in config.image_model_trunks
            )
        )
        self.heads = SSLComponent(
            model=nn.ModuleList(
                # Construct model object using config
                head.head_model()
                for head in config.heads
            )
        )
        self.losses = SSLComponent(
            model=nn.ModuleList(
                # Construct model object using config
                loss.loss()
                for loss in config.losses
            )
        )

        # Create losses models with `output` and `model` accessible keys
        if config.losses is not None:
            self.loss_weights = [
                single_loss_params.loss_weight
                for single_loss_params in config.losses
            ]

        # Initialize target model with source model weights
        if config.copy_init_weights is not None and self.training:
            for cp_dict in config.copy_init_weights:
                logging.debug(
                    f"Copying weights {cp_dict.source} to {cp_dict.target}"
                )
                src_model = self._get_model_from_str(cp_dict.source)
                tgt_model = self._get_model_from_str(cp_dict.target)
                tgt_model.load_state_dict(src_model.state_dict())

        # Exponential Moving Average for model weights
        if config.ema is not None:
            self.register_buffer("ema_step", torch.zeros(1, dtype=torch.int32))
        self.ema_instances = self.create_ema()

        # Freeze specified parameters
        if config.freeze is not None:
            freeze_modules(self, config.freeze)

    def _reset_parameters(self, category_name):
        if getattr(self, category_name) is not None:
            # Per model reset params
            for model in getattr(self, category_name):
                model.reset_parameters()

    def reset_parameters(self):
        self._reset_parameters("image_model_trunks")
        self._reset_parameters("heads")
        self._reset_parameters("losses")

    def create_ema(self):
        # Create EMAWrapper objects for all models with EMA updates
        # and return List[EMAWrapper]
        ema_instances = []
        if self.config.ema is not None:
            for _ema in self.config.ema:
                src_model = self._get_model_from_str(_ema.source)
                tgt_model = self._get_model_from_str(_ema.target)
                scheduler = create_momentum_scheduler(
                    _ema.scheduler_name, **_ema.scheduler_params
                )
                ema_instances.append(
                    EMAWrapper(src_model, tgt_model, scheduler)
                )

        return ema_instances

    def _get_model_from_str(self, model_str):
        # Convert notation with () to [] and
        # access models in nn.ModuleList
        # For ex: image_model_trunks.model(0) -> image_model_trunks.model[0]
        model_str = model_str.replace("(", "[").replace(")", "]")
        return eval(f"self.{model_str}")

    def forward_single_component(
        self, models, component_params, output_list_obj
    ):
        """
        Main function that does forward pass per section.
        """

        for model_idx, (model, _model_params) in enumerate(
            zip(models, component_params)
        ):  # num_models iterations

            # per model output for all inputs in forward_args
            output_model = []
            is_no_grad = _model_params.stop_grad

            # forward pass per model all inputs
            for single_fwd_input_list in _model_params.forward_args:
                # list of inputs to be passed per fwd pass

                # build inputs:
                fwd_input = []
                for inp in single_fwd_input_list:
                    # scalar inputs, boolean etc are not part of `self` attrs.
                    # Run eval() only on scalars
                    eval_str = (
                        eval(f"self.{inp}")
                        if (isinstance(inp, str) and ".output" in inp)
                        else inp
                    )
                    fwd_input.append(eval_str)

                # Fwd pass a single input list through the model
                if is_no_grad:
                    with torch.no_grad():
                        output = model(*fwd_input)
                else:
                    output = model(*fwd_input)

                if isinstance(output, tuple):
                    output = list(output)
                else:
                    output = [output]
                output_model.append(output)

            # per model output for all inputs in forward_args
            output_list_obj.output_list = (output_model, model_idx)

    def forward(self, batch):

        # Create Transform output
        self.ssl_transform.output = InputKey(batch)

        # updates with EMA
        # Do not apply EMA at init.
        if self.training:
            for ema_inst in self.ema_instances:
                ema_inst.apply_ema(self.ema_step)

        self.ema_step += 1

        # forward pass using image_model_trunks if present
        if self.image_model_trunks:
            self.image_model_trunks.output = OutputList(
                [[] for _ in range(len(self.image_model_trunks.model))]
            )
            self.forward_single_component(
                self.image_model_trunks.model,
                self.config.image_model_trunks,
                self.image_model_trunks.output,
            )

        # forward pass using heads if present
        if self.heads:
            self.heads.output = OutputList(
                [[] for _ in range(len(self.heads.model))]
            )
            heads_output = self.forward_single_component(  # noqa
                self.heads.model,
                self.config.heads,
                self.heads.output,
            )

        # forward pass from loss if present and aggregate
        total_loss = torch.tensor(0.0, dtype=cstorch.amp.get_half_dtype()).to(
            cstorch.current_torch_device()
        )
        if self.losses:
            self.losses.output = OutputList(
                [[] for _ in range(len(self.losses.model))]
            )
            losses_output = self.forward_single_component(  # noqa
                self.losses.model,
                self.config.losses,
                self.losses.output,
            )

            agg_per_loss_list = [
                self._aggregate_single_loss(loss) * wgt
                for loss, wgt in zip(self.losses.output, self.loss_weights)
            ]
            total_loss = sum(agg_per_loss_list)

            for i, val in enumerate(agg_per_loss_list):
                loss_name = self.config.losses[i].loss.name
                summarize_scalar(f"loss/{loss_name}", val)

        return total_loss

    def _aggregate_single_loss(self, loss_list):
        total_loss = torch.tensor(0.0, dtype=cstorch.amp.get_half_dtype()).to(
            cstorch.current_torch_device()
        )
        for val in loss_list:
            if isinstance(val, list):
                total_loss += self._aggregate_single_loss(val)
            else:
                total_loss += val

        return total_loss

    def __repr__(self):
        s = f"{self.__class__.__name__} \n"
        for k in self._component_param_keys:
            s += f"{k}: {getattr(self, k)}\n"

        return s
