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

'''
This is registry for the cerebras modelzoo
'''
import importlib
import os
import pathlib
from pathlib import Path

import cerebras.modelzoo as modelzoo


class Registry:
    mapping = {
        "model": {},
        "datasetprocessor": {},
        "lr_scheduler": {},
        "loss": {},
        "dataset": {},
        "paths": {},
        "config": {},
    }
    _modules_imported = False

    @classmethod
    def _import_modules_for_registry(
        cls, directory_path: str, import_files_regex: str
    ):
        """Importing all classes from the files mentioned in the directory path and
        in import_files. If no files are specified, all python files from that
        directory will be imported."""

        modelzoo_path = os.path.dirname(os.path.realpath(modelzoo.__file__))
        for file in Path(directory_path).rglob(import_files_regex):
            filename = pathlib.Path(file).name
            module_path = "cerebras.modelzoo.{}".format(
                os.path.relpath(file, modelzoo_path).replace(os.path.sep, '.')[
                    :-3
                ]
            )
            # Import the module dynamically
            try:
                importlib.import_module(module_path, package=__name__)
            except Exception as ex:
                raise Exception("Registry Import Failure: {}".format(ex))

    @classmethod
    def _import_modules(cls):
        if cls._modules_imported:
            return

        for path in cls.mapping["paths"]["model_path"]:
            cls._import_modules_for_registry(
                path,
                import_files_regex="**/model.py",
            )

        for path in cls.mapping["paths"]["loss_path"]:
            cls._import_modules_for_registry(path, import_files_regex="**/*.py")

        for path in cls.mapping["paths"]["datasetprocessor_path"]:
            cls._import_modules_for_registry(
                path,
                import_files_regex="**/*Processor*.py",
            )

        for path in cls.mapping["paths"]["model_path"]:
            cls._import_modules_for_registry(
                path,
                import_files_regex="**/config.py",
            )
        cls._modules_imported = True

    @classmethod
    def register_model(cls, model_name, datasetprocessor=[], dataset=[]):
        """
        This method is added to register models
        """

        def wrap(model_cls):
            if not isinstance(model_name, list):
                names = [model_name]
            else:
                names = model_name

            for name in names:
                if name in cls.mapping["model"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            name, cls.mapping["model"][name]
                        )
                    )
                cls.mapping["model"][name] = dict()
                cls.mapping["model"][name]["class"] = model_cls
                cls.mapping["model"][name]["run"] = cls.register_run_path(name)
                cls.mapping["model"][name][
                    "datasetprocessor"
                ] = datasetprocessor
                cls.mapping["model"][name]["dataset"] = dataset
            return model_cls

        return wrap

    @classmethod
    def register_datasetprocessor(cls, name):
        """
        This method is added to register datasetprocessor
        """

        def wrap(datasetprocessor_cls):
            if name in cls.mapping["datasetprocessor"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["datasetprocessor"][name]
                    )
                )
            cls.mapping["datasetprocessor"][name] = datasetprocessor_cls
            return datasetprocessor_cls

        return wrap

    @classmethod
    def register_loss(cls, name):
        """
        This method is added to register loss
        """

        def wrap(loss_cls):
            if name in cls.mapping["loss"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["loss"][name]
                    )
                )
            cls.mapping["loss"][name] = loss_cls
            return loss_cls

        return wrap

    @classmethod
    def register_lr_scheduler(cls, name):
        """
        This method is added to register lr_schedular
        """

        def wrap(lr_scheduler_cls):
            if name in cls.mapping["lr_scheduler"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["lr_scheduler"][name]
                    )
                )
            cls.mapping["lr_scheduler"][name] = lr_scheduler_cls
            return lr_scheduler_cls

        return wrap

    @classmethod
    def register_dataset(cls, name):
        """
        This method is added to register dataset
        """

        def wrap(dataset_cls):
            if name in cls.mapping["dataset"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["dataset"][name]
                    )
                )
            cls.mapping["dataset"][name] = dataset_cls
            return dataset_cls

        return wrap

    @classmethod
    def register_paths(cls, kind, path):
        """
        This method is register paths useful for the user
        """
        if kind in cls.mapping["paths"]:
            cls.mapping["paths"][kind].append(path)
        else:
            cls.mapping["paths"].setdefault(kind, [path])

    @classmethod
    def register_config(cls, name):
        """
        This method is added to register config classes
        """

        def wrap(model_cls):
            cls.mapping["config"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def get_path(cls, kind, name):
        if kind in cls.mapping["paths"]:
            for path in cls.mapping["paths"][kind]:
                if os.path.isdir(os.path.join(path, name)):
                    return os.path.join(path, name)
            return None
        else:
            raise ValueError("{} not initialised in registry".format(kind))

    @classmethod
    def register_run_path(cls, name):
        """
        Look for run path for the model
        """
        return cls.get_path("model_path", name)

    @classmethod
    def unregister(cls, region, name):
        """
        This method is added to unregister
            region can be ['model', 'loss', 'lr_scheduler',
                            'datasetprocessor', 'dataset']
        """
        if cls.mapping.get('region') is None:
            raise KeyError("Undefined {}".format(region))
        return cls.mapping[region].pop(name, None)

    @classmethod
    def list_models(cls):
        cls._import_modules()
        return sorted(cls.mapping["model"].keys())

    @classmethod
    def list_loss(cls):
        cls._import_modules()
        return sorted(cls.mapping["loss"].keys())

    @classmethod
    def list_datasetprocessor(cls, model_name=None):
        cls._import_modules()
        if model_name is None:
            return sorted(cls.mapping["datasetprocessor"].keys())
        if model_name in cls.mapping["model"]:
            for dl in cls.mapping["model"][model_name]["datasetprocessor"]:
                if not (dl in cls.mapping["datasetprocessor"]):
                    raise ValueError(
                        "{} datasetprocessor is not registered".format(dl)
                    )
            return cls.mapping["model"][model_name]["datasetprocessor"]
        else:
            raise ValueError("{} model is not registered".format(model_name))

    @classmethod
    def list_lr_scheduler(cls):
        cls._import_modules()
        return sorted(cls.mapping["lr_scheduler"].keys())

    @classmethod
    def list_dataset(cls, model_name=None):
        cls._import_modules()
        if model_name is None:
            return sorted(cls.mapping["dataset"].keys())
        if model_name in cls.mapping["model"]:
            for ds in cls.mapping["model"][model_name]["dataset"]:
                if not (ds in cls.mapping["datset"]):
                    raise ValueError("{} dataset is not registered".format(ds))
            return cls.mapping["model"][model_name]["dataset"]
        else:
            raise ValueError("{} model is not registered".format(model_name))

    @classmethod
    def get_model_class(cls, name):
        cls._import_modules()
        if name in cls.mapping["model"]:
            return cls.mapping["model"][name]["class"]
        return ValueError("{} model is not registered".format(name))

    @classmethod
    def get_config_class(cls, name):
        cls._import_modules()
        if name in cls.mapping["config"]:
            return cls.mapping["config"][name]
        return None

    @classmethod
    def get_loss_class(cls, name):
        cls._import_modules()
        return cls.mapping["loss"].get(name, None)

    @classmethod
    def get_run_path(cls, name):
        if name in cls.mapping["model"]:
            return cls.mapping["model"][name]["run"]
        else:
            return cls.get_path("model_path", name)


registry = Registry()
