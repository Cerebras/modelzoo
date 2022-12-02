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
Abstract base class for PyTorch models.
"""
from abc import ABC, abstractmethod

import torch

from modelzoo.common.pytorch import amp
from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import modes
from modelzoo.common.pytorch.gradient_clipper import GradientClipper
from modelzoo.common.pytorch.optim import (
    ASGD,
    SGD,
    Adadelta,
    Adafactor,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    Lamb,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    lr_scheduler,
)
from modelzoo.common.pytorch.utils import group_optimizer_params

SUPPORTED_OPTIMIZERS = [
    'Adadelta',
    'Adafactor',
    'Adagrad',
    'Adam',
    'AdamW',
    'Adamax',
    'ASGD',
    'Lamb',
    'NAdam',
    'RAdam',
    'RMSprop',
    'Rprop',
    'SGD',
]


class PyTorchBaseModel(ABC):
    def __init__(
        self, params: dict, model: torch.nn.Module, device: torch.device
    ):
        self.model = model
        if cm.use_cs():
            from modelzoo.common.pytorch import cbtorch

            self.model = cbtorch.module(self.model, device)
        elif device:
            self.model = self.model.to(device)

        self._post_device_transfer()

        self.mode = params["runconfig"]["mode"]
        self.mixed_precision = params["model"]["mixed_precision"]

        # Whether or not to allow multireplica runs
        # default to false for eval runs.
        self.allow_multireplica = params["model"].get(
            "allow_multireplica", True
        ) and self.mode == "train"

        seed = params["runconfig"].get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)

        oparams = params["optimizer"]

        # Learning rate params
        self.lr_scheduler = None
        lr_params = {
            "learning_rate": oparams["learning_rate"],
            "disable_lr_steps_reset": oparams.get(
                "disable_lr_steps_reset", False
            ),
        }
        if not isinstance(lr_params["learning_rate"], (float, str, dict, list)):
            raise ValueError(
                f"Learning rate must be a float, a dict, or a list of dicts. "
                f"Got {type(lr_params['learning_rate'])}"
            )

        self.optimizer = None
        if self.mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
            self.optimizer = self._configure_optimizer(oparams)

            if cm.use_cs():
                from modelzoo.common.pytorch import cbtorch

                self.optimizer = cbtorch.optimizer(self.optimizer)

            self.lr_scheduler = self._configure_lr_scheduler(lr_params)

            if cm.use_cs() and cbtorch.env().weight_streaming_mode:
                self.optimizer.set_main_lr_scheduler(self.lr_scheduler)

        if cm.use_cs():  # init grad scaler for mixed precision
            self.grad_scaler = amp.GradScaler(
                loss_scale=oparams.get("loss_scaling_factor"),
                initial_loss_scale=oparams.get("initial_loss_scale"),
                steps_per_increase=oparams.get("steps_per_increase"),
                min_loss_scale=oparams.get("min_loss_scale"),
                max_loss_scale=oparams.get("max_loss_scale"),
                max_gradient_norm=oparams.get("max_gradient_norm"),
                mixed_precision=self.mixed_precision,
            )

        if self.optimizer:
            # Gradient clipping params
            self.optimizer.gradient_clipper = GradientClipper(
                oparams.get("max_gradient_norm", 0.0),
                oparams.get("max_gradient_value", 0.0),
            )

        # set duplicate params for params and buffers in the model
        self._duplicate_params_map = self._named_members(
            self.model, lambda module: module._parameters.items()
        )
        self._duplicate_params_map.update(
            self._named_members(
                self.model, lambda module: module._buffers.items()
            )
        )

    def train(self):
        """
        Sets the model into training mode, equivalent to .train() called on a torch.nn.Module.
        """
        self.model.train()
        self.mode = modes.TRAIN

    def eval(self):
        """
        Sets the model into eval mode, equivalent to .eval() called on a torch.nn.Module.
        """
        self.model.eval()
        self.mode = modes.EVAL

    @property
    def duplicate_params_map(self):
        """
        Returns a map of param names which hold the same tensors
        key and value are same as the names that appear in state_dict
        """
        return self._duplicate_params_map

    @property
    def supported_cs_modes(self):
        """
        Returns a list of modes that are supported for CS runs.

        By default we support train and eval, however, this property
        is designed to be overriden on a model-by-model basis.
        """
        return (modes.TRAIN, modes.EVAL)

    @property
    def supported_non_cs_modes(self):
        """
        Returns a list of modes that are supported for non-CS (CPU/GPU) runs.

        By default we support train, eval and train_and_eval, however, this
        property is designed to be overriden on a model-by-model basis.
        """
        return (modes.TRAIN, modes.EVAL, modes.TRAIN_AND_EVAL)

    def supports_mode(self, mode) -> bool:
        if cm.use_cs():
            return mode in self.supported_cs_modes
        else:
            return mode in self.supported_non_cs_modes

    def _post_device_transfer(self):
        """
        Callback after model is copied to device, but before optimizers are
        configured.
        """

    def trainable_named_parameters(self):
        return self.model.named_parameters()

    def _configure_optimizer(self, oparams: dict):
        """
        Configure an optimizer based on the params and return it
        """
        optimizer_type = oparams["optimizer_type"].lower()

        learning_rate = oparams["learning_rate"]
        if isinstance(learning_rate, (float, str)):
            learning_rate = float(learning_rate)
        else:  # Indicates learning rate scheduling which sets the LR in the scheduler
            learning_rate = 0.1

        param_optimizer = list(self.trainable_named_parameters())
        param_optimizer_grouped = group_optimizer_params(
            param_optimizer, oparams
        )
        if optimizer_type == "sgd":
            return SGD(
                param_optimizer_grouped,
                lr=learning_rate,
                momentum=oparams["momentum"],
                weight_decay=oparams.get("weight_decay_rate", 0.0),
                nesterov=oparams.get("use_nesterov", False),
            )
        elif optimizer_type == "adam":
            return Adam(
                param_optimizer_grouped,
                lr=learning_rate,
                betas=(oparams.get("beta1", 0.9), oparams.get("beta2", 0.999)),
                eps=oparams.get("eps", 1e-6),
                amsgrad=oparams.get("amsgrad", False),
            )
        elif optimizer_type == "adamw":
            return AdamW(
                param_optimizer_grouped,
                lr=learning_rate,
                betas=(oparams.get("beta1", 0.9), oparams.get("beta2", 0.999)),
                correct_bias=oparams.get("correct_bias", False),
                eps=oparams.get("eps", 1e-6),
                amsgrad=oparams.get("amsgrad", False),
            )
        elif optimizer_type == "adamax":
            return Adamax(
                param_optimizer_grouped,
                lr=learning_rate,
                betas=(oparams.get("beta1", 0.9), oparams.get("beta2", 0.999)),
                eps=oparams.get("eps", 1e-6),
                weight_decay=oparams.get("weight_decay_rate", 0.0),
                maximize=oparams.get("maximize", False),
            )
        elif optimizer_type == "adadelta":
            return Adadelta(
                param_optimizer_grouped,
                lr=learning_rate,
                rho=oparams.get("rho", 0.9),
                eps=oparams.get("eps", 1e-6),
                weight_decay=oparams.get("weight_decay_rate", 0.0),
            )
        elif optimizer_type == "adafactor":
            eps = (oparams.get("eps1", 1e-30), oparams.get("eps2", 1e-3))
            clip_threshold = oparams.get("clip_threshold", 1.0)
            decay_rate = oparams.get("decay_rate", -0.8)
            beta1 = oparams.get("beta1", None)
            scale_parameter = oparams.get("scale_parameter", True)
            relative_step = oparams.get("relative_step", False)
            warmup_init = oparams.get("warmup_init", False)
            return Adafactor(
                param_optimizer_grouped,
                lr=learning_rate,
                eps=eps,
                clip_threshold=clip_threshold,
                decay_rate=decay_rate,
                beta1=beta1,
                scale_parameter=scale_parameter,
                relative_step=relative_step,
                warmup_init=warmup_init,
            )
        elif optimizer_type == "adagrad":
            return Adagrad(
                param_optimizer_grouped,
                lr=learning_rate,
                lr_decay=oparams.get("lr_decay", 0.0),
                weight_decay=oparams.get("weight_decay_rate", 0.0),
                initial_accumulator_value=oparams.get(
                    "initial_accumulator_value", 0.0
                ),
                eps=oparams.get("eps", 1e-6),
                maximize=oparams.get("maximize", False,),
            )
        elif optimizer_type == "asgd":
            return ASGD(
                param_optimizer_grouped,
                lr=learning_rate,
                lambd=oparams.get("lambd", 1e-4),
                alpha=oparams.get("alpha", 0.75),
                t0=oparams.get("t0", 1e-6),
                weight_decay=oparams.get("weight_decay", 0.0),
                maximize=oparams.get("maximize", False),
            )
        elif optimizer_type == "lamb":
            eps = oparams.get("eps", 1e-6)
            betas = (oparams.get("beta1", 0.9), oparams.get("beta2", 0.999))
            adam = oparams.get("adam", False)
            weight_decay = oparams.get("weight_decay_rate", 0.0)
            return Lamb(
                param_optimizer_grouped,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                adam=adam,
            )
        elif optimizer_type == "radam":
            eps = oparams.get("eps", 1e-6)
            betas = (oparams.get("beta1", 0.9), oparams.get("beta2", 0.999))
            weight_decay = oparams.get("weight_decay_rate", 0)
            return RAdam(
                param_optimizer_grouped,
                lr=learning_rate,
                eps=eps,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "nadam":
            eps = oparams.get("eps", 1e-6)
            betas = (oparams.get("beta1", 0.9), oparams.get("beta2", 0.999))
            weight_decay = oparams.get("weight_decay_rate", 0)
            momentum_decay = oparams.get("momentum_decay", 4e-3)
            return NAdam(
                param_optimizer_grouped,
                lr=learning_rate,
                eps=eps,
                betas=betas,
                weight_decay=weight_decay,
                momentum_decay=momentum_decay,
            )
        elif optimizer_type == "rmsprop":
            return RMSprop(
                param_optimizer_grouped,
                lr=learning_rate,
                alpha=oparams.get("alpha", 0.99),
                momentum=oparams.get("momentum", 0),
                centered=oparams.get("centered", False),
                eps=oparams.get("eps", 1e-6),
                weight_decay=oparams.get("weight_decay_rate", 0.0),
            )
        elif optimizer_type == "rprop":
            etas = (oparams.get("eta1", 0.5), oparams.get("eta2", 1.2))
            step_sizes = (
                oparams.get("step_size_min", 1e-6),
                oparams.get("step_size_max", 50.0),
            )
            return Rprop(
                param_optimizer_grouped,
                lr=learning_rate,
                etas=etas,
                step_sizes=step_sizes,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type {optimizer_type}. Supported types:"
                f"{SUPPORTED_OPTIMIZERS}."
            )

    def get_optimizer(self):
        """
        Returns the optimizer associated with this model.
        """
        return self.optimizer

    def _configure_lr_scheduler(self, lr_params):
        """
        Initiates the LR Scheduler associated with this model.
        """
        learning_rate = lr_params["learning_rate"]
        disable_lr_steps_reset = lr_params["disable_lr_steps_reset"]

        def _set_initial_lr(optimizer, lr):
            for group in self.optimizer.param_groups:
                group['lr'] = float(lr)

        def _get_scheduler(optimizer, schedule_params):
            """
            Parses a dict of learning rate scheduler specifications and
            returns a learning rate tensor.

            :param dict schedule_params:
                    A dict with a "scheduler" key (e.g.,
                    schedule_params["scheduler"] = "Exponential") and all
                    params schedulers of that type need.

            :returns: The learning rate tensor.
            """
            scheduler = schedule_params["scheduler"].lower()

            # to handle discrepancy in step parameters
            if "steps" in scheduler:
                scheduler["decay_steps"] = scheduler["steps"]
            elif "decay_steps" in scheduler:
                scheduler["steps"] = scheduler["decay_steps"]

            def check_required_params(required_params):
                missing = list(set(required_params) - set(schedule_params))
                if missing:
                    raise ValueError(
                        f"Missing required parameters {missing} "
                        f"for the {scheduler} learning rate scheduler. "
                        f"Note, the {scheduler} learning rate scheduler "
                        f"requires the following parameters: {required_params}"
                    )

            if scheduler == "constant":
                check_required_params(["learning_rate"])
                return lr_scheduler.Constant(
                    optimizer,
                    val=schedule_params["learning_rate"],
                    decay_steps=schedule_params.get("steps", None),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "exponential":
                check_required_params(
                    ["initial_learning_rate", "decay_steps", "decay_rate"]
                )
                return lr_scheduler.Exponential(
                    optimizer,
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    decay_steps=schedule_params["decay_steps"],
                    decay_rate=schedule_params["decay_rate"],
                    staircase=schedule_params.get("staircase", False),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "piecewiseconstant":
                check_required_params(["values", "boundaries"])
                return lr_scheduler.PiecewiseConstant(
                    optimizer,
                    learning_rates=schedule_params["values"],
                    milestones=schedule_params["boundaries"],
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler in ("polynomial", "linear"):
                check_required_params(
                    ["initial_learning_rate", "end_learning_rate", "steps"]
                )
                power = (
                    1.0
                    if scheduler == "linear"
                    else schedule_params.get("power", 1.0)
                )
                return lr_scheduler.Polynomial(
                    optimizer,
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    end_learning_rate=schedule_params["end_learning_rate"],
                    decay_steps=schedule_params["steps"],
                    power=power,
                    cycle=schedule_params.get("cycle", False),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "inverseexponentialtimedecay":
                check_required_params(
                    [
                        "initial_learning_rate",
                        "step_exponent",
                        "decay_steps",
                        "decay_rate",
                    ]
                )
                return lr_scheduler.InverseExponentialTimeDecay(
                    optimizer,
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    step_exponent=schedule_params["step_exponent"],
                    decay_steps=schedule_params["decay_steps"],
                    decay_rate=schedule_params["decay_rate"],
                    staircase=schedule_params.get("staircase", False),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "cosinedecay":
                check_required_params(
                    [
                        "initial_learning_rate",
                        "end_learning_rate",
                        "decay_steps",
                    ]
                )

                return lr_scheduler.CosineDecay(
                    optimizer,
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    end_learning_rate=schedule_params["end_learning_rate"],
                    decay_steps=schedule_params["decay_steps"],
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "cosineannealing":
                check_required_params(
                    ["learning_rate", "t_max",]
                )

                return lr_scheduler.CosineAnnealing(
                    optimizer,
                    learning_rate=schedule_params["learning_rate"],
                    T_max=schedule_params["t_max"],
                    eta_min=schedule_params.get("eta_min", 0.0),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "step":
                check_required_params(
                    ["learning_rate", "step_size", "gamma",]
                )
                return lr_scheduler.Step(
                    optimizer,
                    learning_rate=schedule_params["learning_rate"],
                    gamma=schedule_params["gamma"],
                    step_size=schedule_params["step_size"],
                    disable_lr_steps_reset=False,
                )
            elif scheduler == "multistep":
                check_required_params(
                    ["learning_rate", "gamma", "milestones",]
                )
                return lr_scheduler.MultiStep(
                    optimizer,
                    learning_rate=schedule_params["learning_rate"],
                    gamma=schedule_params["gamma"],
                    milestones=schedule_params["milestones"],
                    disable_lr_steps_reset=False,
                )
            elif scheduler == "lambda":
                check_required_params(["learning_rate"])
                return lr_scheduler.Lambda(
                    optimizer,
                    learning_rate=schedule_params["learning_rate"],
                    disable_lr_steps_reset=False,
                )
            else:
                raise ValueError(f"Unsupported LR scheduler {scheduler}")

        # handle a constant learning rate
        # scientific notation (e.g. "1e-5") parsed as string in yaml
        if isinstance(learning_rate, (float, str)):
            _set_initial_lr(self.optimizer, learning_rate)

        # handle a single decay schedule
        elif isinstance(learning_rate, dict):
            return _get_scheduler(self.optimizer, learning_rate)

        elif isinstance(learning_rate, list):
            if len(learning_rate) == 1:
                return _get_scheduler(self.optimizer, learning_rate[0])
            else:

                for scheduler in learning_rate[:-1]:
                    assert "steps" in scheduler or "decay_steps" in scheduler, (
                        "Non final learning rate schedulers must have either "
                        "the 'steps' or 'decay_steps' parameter given."
                    )

                schedulers = [
                    _get_scheduler(self.optimizer, scheduler)
                    for scheduler in learning_rate
                ]
                milestones = [
                    scheduler.start_step for scheduler in schedulers[1:]
                ]

                return lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=schedulers,
                    milestones=milestones,
                )
        else:
            raise ValueError(
                f"Unsupported LR scheduler type {type(learning_rate)}"
                f"Supported LR schedulers are ['Constant', 'Exponential',"
                f" 'PiecewiseConstant', 'Polynomial',"
                f" 'InverseExponentialTimeDecay']"
            )

    def get_lr_scheduler(self):
        """
        Returns the LR Scheduler associated with this model.
        """
        return self.lr_scheduler

    def get_state(self):
        """
        Returns the state of the model and optimizer
        """
        state_dict = {
            "model": self.model.state_dict(),
        }

        if self.optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()

        if self.lr_scheduler:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        if self.mixed_precision and cm.use_cs():
            state_dict["amp"] = amp.state_dict()

        return state_dict

    def _named_members(self, model, get_member_fn, prefix='', recurse=True):
        """
        Helper method which returns a map of param_name -> set of duplicate param names
        """
        memo = dict()
        names = dict()
        modules = (
            model.named_modules(prefix=prefix, remove_duplicate=False)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_member_fn(module)
            for k, v in members:
                name = module_prefix + ('.' if module_prefix else '') + k
                if v is None:
                    continue
                elif v in memo:
                    # whenever a duplicate is found
                    # update the existing list of duplicate
                    # names corresponding to the first name
                    duplicates = names.get(memo[v], set([memo[v]]))
                    duplicates.add(name)
                    names[memo[v]] = duplicates
                    # also add a key for new name with
                    # value as the duplicates list
                    names[name] = duplicates
                    continue
                memo[v] = name

        return names

    def set_state(self, state, strict=True):
        """
        Sets the state of the model and optimizer
        """
        is_pretrained_checkpoint = self.params["runconfig"].get(
            "is_pretrained_checkpoint", False
        )
        mode = self.params["runconfig"]["mode"]
        if is_pretrained_checkpoint and mode != modes.EVAL:
            # allow loading weights ignoring the mising and unexpected keys
            # except when doing eval
            strict = False
        self.model.load_state_dict(state["model"], strict=strict)
        if (
            self.optimizer
            and "optimizer" in state
            and not is_pretrained_checkpoint
        ):
            # load optimizer state for resuming training
            self.optimizer.load_state_dict(state["optimizer"])
            if self.lr_scheduler and "lr_scheduler" in state:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        if (
            self.mixed_precision
            and cm.is_wse_device()
            and not is_pretrained_checkpoint
        ):
            amp_state = state.get('amp')
            if amp_state:
                amp.load_state_dict(amp_state)

    @abstractmethod
    def __call__(self, data):
        """
        Given one iteration of a dataloader, returns the loss associated with
        one forward pass of that batch.
        """
        raise NotImplementedError(
            "__call__ must be implemented in a child class!"
        )
