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

"""Contains utility for validating trainer configuration."""

import dataclasses
import inspect
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union, get_args
from warnings import catch_warnings, simplefilter, warn

from pydantic import (
    BeforeValidator,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    WrapSerializer,
    field_validator,
    model_serializer,
    model_validator,
)
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from typing_extensions import Annotated

from cerebras.modelzoo.config import BaseConfig, DataConfig, create_config_class
from cerebras.modelzoo.registry import registry

DEFAULT_CALLBACKS = {
    "CheckLoss": {},
    "LogOptimizerParamGroup": {"keys": "lr"},
    "ComputeNorm": {},
    "ModelEvalMetrics": {},
    "RateProfiler": {},
    "DurationProfiler": {},
    "FlopUtilization": {},
    "SavePerformanceData": {},
    "DumpAvailableTensorNames": {},
    "ModelZooParamsMetadata": {},
    "SamplesStreamedInfo": {},
}
DEFAULT_LOGGERS = {
    # pylint: disable=unnecessary-lambda
    "ProgressLogger": {},
    "TensorBoardLogger": {},
    "TelemetryLogger": {},
}


def construct_trainer_config(model_name: str):
    """
    Construct trainer config class from the given params.

    Args:
        model_name: The model for which to construct a config.

    Returns:
        The trainer config class
    """
    import numpy
    import torch

    import cerebras.pytorch as cstorch
    from cerebras.appliance.utils.classes import retrieve_all_subclasses
    from cerebras.pytorch.optim.optimizer import ParamsT

    discriminator_key = "__name__"
    discriminator = Discriminator(lambda d: d.pop(discriminator_key).lower())

    def unpack(d: dict, name: str, allow_list=False):
        if isinstance(d, (list, tuple)):
            if not allow_list:
                raise TypeError(f"Expected {name} to be a dict. Got: {type(d)}")
            return list(map(partial(unpack, name=name), d))

        if not isinstance(d, dict):
            raise TypeError(f"Expected {name} to be a dict. Got: {type(d)}")
        if len(d) != 1:
            raise ValueError(
                f"Expected {name} to have a single key. Got: {sorted(d)}"
            )

        key = next(iter(d))
        return {discriminator_key: key, **d[key]}

    def pack(value, handler, info, name):
        return {name: handler(value, info)}

    class OptimConfig(BaseConfig):  # pylint: disable=missing-class-docstring
        # TODO: This is a temporary hack to get around the fact
        #       that optimizer params have adjust_learning_rate.
        #       This should be removed once we have a better way
        #       to handle this.
        model_config = dict(extra="allow", include_extra=False)

        @model_validator(mode="after")
        def warn_extra(self):  # pylint: disable=missing-function-docstring
            model_extra = set(self.model_extra) - {"adjust_learning_rate"}
            if model_extra:
                signature = ", ".join(
                    (
                        f"{name}={getattr(self, name)}"
                        if name in self.model_fields_set
                        else f"{name}={'<missing>' if f.is_required() else f.default}"
                    )
                    for name, f in self.model_fields.items()
                )
                warn(
                    f"{self.__orig_class__.__name__} got {len(model_extra)} unexpected "
                    f"and unused parameters: {sorted(model_extra)}.\n"
                    f"Please ensure that you specified the correct parameters:\n"
                    f"{self.__orig_class__.__name__}({signature})\n"
                    f"Passing in unused parameters is deprecated behaviour and "
                    f"support for it will be removed in a future release."
                )
            return self

    optim_annotation = Annotated[
        Union[
            tuple(
                Annotated[
                    create_config_class(
                        cls,
                        custom_type_mapping={
                            ParamsT: Annotated[List[dict], Field(default=[])],
                            # Allow `learning_rate` as an alias of `lr`
                            (float, "lr"): Annotated[
                                float, Field(default=0.1, alias="learning_rate")
                            ],
                            (Optional[float], "lr"): Annotated[
                                Optional[float],
                                Field(default=None, alias="learning_rate"),
                            ],
                            # Allow `lr` as an alias of `learning_rate`
                            (float, "learning_rate"): Annotated[
                                float, Field(default=0.1, alias="lr")
                            ],
                            (Optional[float], "learning_rate"): Annotated[
                                Optional[float], Field(default=None, alias="lr")
                            ],
                        },
                        __base__=OptimConfig,
                    ),
                    Tag(cls.__name__.lower()),
                    WrapSerializer(partial(pack, name=cls.__name__)),
                ]
                for cls in retrieve_all_subclasses(cstorch.optim.Optimizer)
            )
        ],
        discriminator,
        BeforeValidator(partial(unpack, name="optimizer")),
    ]

    # pylint: disable=missing-class-docstring,missing-function-docstring

    class SchedulerBase(BaseConfig):

        def get_orig_class_args(self, **kwargs) -> dict:
            args = super().get_orig_class_args(**kwargs)

            # Schedulers are recursively defined and thus need to be
            # recursively constructed
            def recurse(val):
                # Recursively construct the scheduler
                if isinstance(val, SchedulerBase):
                    return val(**kwargs)
                if isinstance(val, (list, tuple, set)):
                    return type(val)(map(recurse, val))
                if isinstance(val, dict):
                    return {k: recurse(v) for k, v in val.items()}

                return val

            return recurse(args)

        @model_validator(mode="before")
        @classmethod
        def compute_milestones(cls, data):
            if issubclass(
                cls.__orig_class__, cstorch.optim.scheduler.SequentialScheduler
            ) and not issubclass(
                cls.__orig_class__,
                cstorch.optim.scheduler.PiecewiseConstantScheduler,
            ):
                milestones = numpy.array(
                    [
                        unpack(scheduler, "scheduler")["total_iters"]
                        for scheduler in data["schedulers"][:-1]
                    ]
                )
                milestones = milestones.cumsum().tolist()
                if "milestones" not in data:
                    data["milestones"] = milestones
                elif data["milestones"] != milestones:
                    raise ValueError(
                        f"Expected {cls.__orig_class__} to have milestones {milestones}, "
                        f"but got {data['milestones']}"
                    )

            if (
                issubclass(
                    cls.__orig_class__,
                    (
                        cstorch.optim.scheduler.SequentialScheduler,
                        cstorch.optim.scheduler.ChainedScheduler,
                    ),
                )
                and "param_group_tags" in data
            ):
                param_group_tags = data["param_group_tags"]

                for scheduler in data["schedulers"]:
                    scheduler_name = list(scheduler.keys())[0]
                    scheduler = scheduler[scheduler_name]

                    if "param_group_tags" in scheduler:
                        raise ValueError(
                            f"Parameter `param_group_tags` found in a nested scheduler "
                            f"{scheduler_name}. "
                            f"A {cls.__orig_class__.__name__} expects all "
                            f"nested schedulers to have no `param_group_tags`."
                        )

                    scheduler["param_group_tags"] = param_group_tags

            return data

    LRScheduler_annotation = Annotated[
        Union[
            tuple(
                Annotated[
                    create_config_class(
                        cls,
                        custom_type_mapping={
                            torch.optim.Optimizer: Annotated[
                                Optional[torch.optim.Optimizer],
                                Field(default=None, exclude=True),
                            ],
                            cstorch.optim.lr_scheduler.LRScheduler: "LRScheduler_annotation",
                        },
                        __base__=SchedulerBase,
                    ),
                    Tag(cls.__name__.lower()),
                    WrapSerializer(partial(pack, name=cls.__name__)),
                ]
                for cls in retrieve_all_subclasses(
                    cstorch.optim.lr_scheduler.LRScheduler
                )
            )
        ],
        discriminator,
        BeforeValidator(partial(unpack, name="LR scheduler")),
    ]
    WeightDecayScheduler_annotation = Annotated[
        Union[
            tuple(
                Annotated[
                    create_config_class(
                        cls,
                        custom_type_mapping={
                            torch.optim.Optimizer: Annotated[
                                Optional[torch.optim.Optimizer],
                                Field(default=None, exclude=True),
                            ],
                            cstorch.optim.weight_decay_scheduler.WeightDecayScheduler: (
                                "WeightDecayScheduler_annotation"
                            ),
                        },
                        __base__=SchedulerBase,
                    ),
                    Tag(cls.__name__.lower()),
                    WrapSerializer(partial(pack, name=cls.__name__)),
                ]
                for cls in retrieve_all_subclasses(
                    cstorch.optim.weight_decay_scheduler.WeightDecayScheduler
                )
            )
        ],
        discriminator,
        BeforeValidator(partial(unpack, name="weight decay scheduler")),
    ]

    scheduler_annotation = Annotated[
        List[
            Annotated[
                Union[
                    get_args(get_args(LRScheduler_annotation)[0])
                    + get_args(get_args(WeightDecayScheduler_annotation)[0])
                ],
                discriminator,
            ],
        ],
        BeforeValidator(partial(unpack, name="scheduler", allow_list=True)),
    ]

    def unpack_sparsity(s):
        if isinstance(s, float):
            return unpack_sparsity([{"sparsity": s}])
        elif isinstance(s, dict):
            return unpack_sparsity([s])
        elif isinstance(s, (list, tuple)):
            for config in s:
                if "schedule" in config:
                    if "sparsity" in config:
                        raise ValueError(
                            "Cannot specify both 'sparsity' and 'schedule' in the same config"
                        )
                    config["sparsity"] = config.pop("schedule")

            return s

        raise TypeError(
            f"Expected sparsity to be a float, dict, or list of dicts. "
            f"Got: {type(s)}"
        )

    sparsity_annotation = Annotated[
        List[
            Annotated[
                Union[
                    tuple(
                        # TODO: Need to improve sparsity algorithm type hints
                        Annotated[
                            create_config_class(cls), Tag(cls.__name__.lower())
                        ]
                        for cls in retrieve_all_subclasses(
                            cstorch.sparse.SparsityAlgorithm
                        )
                        if not inspect.isabstract(cls)
                        and not isinstance(cls, cstorch.sparse.Group)
                    )
                ],
                Discriminator(
                    lambda sparsity: sparsity.get("algorithm", "static").lower()
                ),
            ]
        ],
        BeforeValidator(unpack_sparsity),
    ]

    import cerebras.modelzoo.trainer.extensions  # noqa # pylint: disable=unused-import
    from cerebras.modelzoo.trainer.callbacks import (
        AutoRestart,
        Callback,
        Checkpoint,
        CoreCallback,
        Logging,
        MixedPrecision,
        ModelZooParamsMetadata,
        TrainingLoop,
    )
    from cerebras.modelzoo.trainer.loggers import Logger

    def inject_defaults(data, defaults: dict):
        defaults = {k.lower(): v for k, v in defaults.items()}

        params = []
        for p in data:
            name = next(iter(p))
            defaults.pop(name.lower(), None)
            # Only include the param if it is not None
            if p[name] is not None:
                params.append(p)

        for name, kwargs in defaults.items():
            params.append({name: kwargs})

        return params

    def inject_metadata_params(cls, params, info):
        context = info.context or {}
        return context.get("metadata_params", params)

    callback_validators = {
        ModelZooParamsMetadata: {
            "inject_metadata_params": field_validator("params")(
                inject_metadata_params
            )
        }
    }

    callback_annotation = Annotated[
        List[
            Annotated[
                Union[
                    tuple(
                        Annotated[
                            create_config_class(
                                cls,
                                __validators__=callback_validators.get(cls, {}),
                            ),
                            Tag(cls.__name__.lower()),
                            WrapSerializer(partial(pack, name=cls.__name__)),
                        ]
                        for cls in retrieve_all_subclasses(Callback)
                        if not issubclass(cls, CoreCallback)
                    )
                ],
                discriminator,
                BeforeValidator(partial(unpack, name="callback")),
            ]
        ],
        BeforeValidator(
            partial(
                inject_defaults,
                defaults=DEFAULT_CALLBACKS,
            )
        ),
    ]

    logger_annotation = Annotated[
        List[
            Annotated[
                Union[
                    tuple(
                        Annotated[
                            create_config_class(cls),
                            Tag(cls.__name__.lower()),
                            WrapSerializer(partial(pack, name=cls.__name__)),
                        ]
                        for cls in retrieve_all_subclasses(Logger)
                    )
                ],
                discriminator,
                BeforeValidator(partial(unpack, name="logger")),
            ]
        ],
        BeforeValidator(
            partial(
                inject_defaults,
                defaults=DEFAULT_LOGGERS,
            )
        ),
    ]

    model_cls = registry.get_model_class(model_name)
    model_annotation = create_config_class(model_cls)

    data_configs = {
        name: create_config_class(data_processor)
        for name, data_processor in registry.get_data_processors_map(
            model_name
        ).items()
    }

    if len(data_configs) == 1:
        dataprocessor_annotation = next(iter(data_configs.values()))
    else:
        data_config_names = set(data_configs.keys())

        def data_discriminator(data):
            if DataConfig.discriminator not in data:
                names = "\n- ".join(sorted(data_config_names))
                raise ValueError(
                    f"Data Processor config must have a `{DataConfig.discriminator}` key "
                    f"with one of the following values:\n- {names}"
                )
            if data[DataConfig.discriminator] not in data_config_names:
                names = "\n- ".join(sorted(data_config_names))
                raise ValueError(
                    f"Expected {DataConfig.discriminator} to be one of the "
                    f"following values:\n- {names}\n"
                    f"Got: {data[DataConfig.discriminator]}"
                )
            return data[DataConfig.discriminator]

        dataprocessor_annotation = Annotated[
            Union[
                tuple(
                    Annotated[data_config, Tag(name)]
                    for name, data_config in data_configs.items()
                )
            ],
            Discriminator(data_discriminator),
        ]

    class TrainerInitConfig(BaseConfig):
        model_config = dict(protected_namespaces=())

        # Instantiate the model first so that it may provide any required
        # context.
        model: model_annotation = ...

        device: Optional[str] = None
        backend: dict = {}
        model_dir: str = "./model_dir"
        optimizer: Optional[optim_annotation] = None
        schedulers: Optional[scheduler_annotation] = None
        precision: Optional[create_config_class(MixedPrecision)] = None
        sparsity: Optional[sparsity_annotation] = None
        loop: Optional[create_config_class(TrainingLoop)] = None
        checkpoint: Optional[create_config_class(Checkpoint)] = None
        logging: Optional[create_config_class(Logging)] = None
        callbacks: callback_annotation = []
        loggers: logger_annotation = []
        seed: Optional[int] = None
        autorestart: Optional[create_config_class(AutoRestart)] = None

    class TrainerFitConfig(BaseConfig):
        train_dataloader: dataprocessor_annotation = ...
        val_dataloader: Optional[List[dataprocessor_annotation]] = None
        ckpt_path: Union[str, None, type(Ellipsis)] = Field(
            default_factory=lambda: ...
        )

        @model_validator(mode="before")
        @classmethod
        def unpack_val_dataloaders(cls, data):
            key = "val_dataloader"
            if val_dataloaders := data.get(key):
                if isinstance(val_dataloaders, dict):
                    val_dataloaders = [val_dataloaders]

                data[key] = val_dataloaders

            return data

        @model_serializer(mode="wrap")
        def serialize_model(self, handler):
            serialized = handler(self)
            if self.ckpt_path is Ellipsis:
                serialized.pop("ckpt_path", None)
            return serialized

    class TrainerValidateConfig(BaseConfig):
        val_dataloader: dataprocessor_annotation = ...
        ckpt_path: Union[str, None, type(Ellipsis)] = Field(
            default_factory=lambda: ...
        )

        @model_serializer(mode="wrap")
        def serialize_model(self, handler):
            serialized = handler(self)
            if self.ckpt_path is Ellipsis:
                serialized.pop("ckpt_path", None)
            return serialized

    class TrainerValidateAllConfig(BaseConfig):
        val_dataloaders: Optional[List[dataprocessor_annotation]] = None
        ckpt_paths: Union[str, List[str], None, type(Ellipsis)] = Field(
            default_factory=lambda: ...
        )

        @model_validator(mode="before")
        @classmethod
        def unpack_val_dataloaders(cls, data):
            key = "val_dataloaders"
            if val_dataloaders := data.get(key):
                if isinstance(val_dataloaders, dict):
                    val_dataloaders = [val_dataloaders]

                data[key] = val_dataloaders

            return data

        @model_serializer(mode="wrap")
        def serialize_model(self, handler):
            serialized = handler(self)
            if self.ckpt_paths is Ellipsis:
                serialized.pop("ckpt_paths", None)
            return serialized

    with catch_warnings():
        # This filter is needed as the Trainer config has a field called `validate`
        simplefilter("ignore", category=UserWarning)

        class TrainerConfig(BaseConfig):
            init: TrainerInitConfig = ...
            fit: Optional[TrainerFitConfig] = None
            validate: Optional[TrainerValidateConfig] = None
            validate_all: Optional[TrainerValidateAllConfig] = None

            @field_validator("init")
            @classmethod
            def add_model_config_to_context(cls, init, info):
                assert info.context is not None
                info.context["model"] = init.model.get_orig_class_args()
                return init

            @model_serializer(mode="wrap")
            def replace_tuple_with_list(self, handler):
                serialized = handler(self)

                values, spec = tree_flatten(serialized)

                # Replace all tuples with lists so that it can be safe loaded
                def replace_tuple_with_list(spec):
                    if isinstance(spec, TreeSpec):
                        if spec.num_children == 0:
                            return spec
                        children_specs = [
                            replace_tuple_with_list(child)
                            for child in spec.children_specs
                        ]
                        spec_type = spec.type
                        if spec.type is tuple:
                            spec_type = list
                        return dataclasses.replace(
                            spec,
                            type=spec_type,
                            children_specs=children_specs,
                        )

                    return spec

                return tree_unflatten(values, replace_tuple_with_list(spec))

        TrainerConfig.model_rebuild()

        return TrainerConfig


def unpack_trainer(t):
    """Unpack multi trainer configuration."""
    if isinstance(t, dict):
        if isinstance(t.get("trainer"), (tuple, list)):
            t = t.get("trainer")
        else:
            t = [t]

    if isinstance(t, (tuple, list)):

        def check(d):
            # trainer
            if not isinstance(d, dict):
                raise TypeError(
                    f"Expected trainer to be a dict. Got: {type(d)}"
                )
            if "trainer" not in d:
                raise KeyError(
                    "Trainer configuration not found in params. "
                    "Please ensure that the params contain a 'trainer' key."
                )

            # trainer.init
            if not isinstance(d["trainer"], dict):
                raise TypeError(
                    f"Expected trainer configuration to be a dict. "
                    f"Got: {type(d['trainer'])}"
                )
            if "init" not in d["trainer"]:
                raise KeyError("Trainer configuration must have an 'init' key.")

            # trainer.init.model
            if not isinstance(d["trainer"]["init"], dict):
                raise TypeError(
                    f"Expected trainer init configuration to be a dict. "
                    f"Got: {type(d['trainer']['init'])}"
                )
            if "model" not in d["trainer"]["init"]:
                raise KeyError(
                    "Trainer init configuration must have a 'model' key."
                )

            # trainer.init.model.name
            model_dict = d["trainer"]["init"]["model"]
            if not isinstance(model_dict, dict):
                raise TypeError(
                    f"Expected trainer init model configuration to be a dict. "
                    f"Got: {type(model_dict)}"
                )
            # Prior to 2.4.0, we accepted "model_name". Starting from 2.4.0,
            # we accept "name". This block handles backwards compatibility for now.
            if "model_name" in model_dict:
                warn(
                    "The \"model_name\" key is now deprecated and will be removed "
                    "in future releases. Please use \"name\" instead.",
                    category=FutureWarning,
                )

                if (
                    "name" in model_dict
                    and model_dict["model_name"] != model_dict["name"]
                ):
                    raise ValueError(
                        f"Got conflicting model names:"
                        f"\n\tmodel_name: {model_dict['model_name']}"
                        f"\n\tname: {model_dict['name']}"
                        f"\nPlease refrain from using \"model_name\" and only "
                        f"use \"name\" instead."
                    )

                model_dict["name"] = model_dict.pop("model_name")
            elif "name" not in d["trainer"]["init"]["model"]:
                raise KeyError("Model configuration must have a 'name' key.")

            return d["trainer"]

        return tuple(map(check, t))

    raise TypeError(
        f"Expected trainer to be a dict, or list of dicts. Got: {type(t)}"
    )


def construct_multi_phase_trainer_config(model_names):
    """Construct multi-phase trainer config."""

    MultiPhaseTrainer = Annotated[
        Tuple[tuple(map(construct_trainer_config, model_names))],
        BeforeValidator(unpack_trainer),
    ]

    return TypeAdapter(MultiPhaseTrainer)


def extract_model_names(params: dict):
    """Extract model names from the given params."""

    return tuple(
        trainer["init"]["model"]["name"] for trainer in unpack_trainer(params)
    )


def validate_trainer_params(params: dict):
    """
    Validate trainer params.

    Args:
        params: Trainer params dictionary.
        model_name: The model name to use if not present in the params dictionary.

    Returns:
        A list of trainer configs
    """
    # Params have already been validated
    if isinstance(params, BaseConfig):
        return params

    if "trainer" not in params:
        raise KeyError(
            "Trainer configuration not found in params. "
            "Please ensure that the params contain a 'trainer' key."
        )

    metadata_params = deepcopy(params)
    try:
        return construct_multi_phase_trainer_config(
            extract_model_names(params)
        ).validate_python(
            params,
            context={"metadata_params": metadata_params},
        )
    except:
        from pprint import pformat

        warn(
            f"Failed to validate params:\n"
            f"{pformat(metadata_params, sort_dicts=False, width=1)}\n"
        )
        raise
