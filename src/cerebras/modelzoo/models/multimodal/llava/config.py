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

import copy
import math
from dataclasses import dataclass
from typing import List, Literal, Optional

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.model_config import (
    ModelConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.optimizer_config import (
    OptimizerConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.run_config import (
    RunConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.sparsity_config import (
    SparsityConfig,
)
from cerebras.modelzoo.models.nlp.gpt2.config import (
    GPT2Config,
    GPT2LMHeadModelConfig,
)
from cerebras.modelzoo.models.nlp.t5.config import (
    T5Config,
    T5ForConditionalGenerationModelConfig,
)
from cerebras.modelzoo.models.vision.vision_transformer.config import (
    VisionTransformerConfig,
    ViTModelConfig,
)

_submodel_mapping = {
    "ViTModel": ViTModelConfig,
    "LlamaModel": GPT2LMHeadModelConfig,
    "T5ForConditionalGeneration": T5ForConditionalGenerationModelConfig,
}

_set_defaults_mapping = {
    "ViTModel": VisionTransformerConfig.set_config_defaults,
    "T5ForConditionalGeneration": T5Config.set_config_defaults,
    "LlamaModel": GPT2Config.set_config_defaults,
}


@dataclass
class LlavaModelConfig(ModelConfig):
    freeze: Optional[List[str]] = None
    image_feature_select_layer_idx: Optional[int] = -1
    image_start_idx: int = required
    image_feature_select_mode: Literal["patch", "cls_patch"] = "patch"
    loss_scaling: str = "num_tokens"
    loss_weight: float = 1.0
    image_model: BaseConfig = required
    "The underlying image model being used"
    text_model: BaseConfig = required
    "The underlying text model being used"
    projector: Optional[dict] = None
    fp16_type: Literal["bfloat16", "float16", "cbfloat16"] = "bfloat16"
    "Type of 16bit precision used"

    def __post_init__(self):
        for component in ["image_model", "text_model"]:
            submodel_handle = self._get_submodel_handle(
                getattr(self, component)["name"]
            )
            setattr(
                self, component, submodel_handle(**getattr(self, component))
            )

        super().__post_init__()

        if self.projector:
            for comp in self.projector.values():
                if comp["name"] == "FeedForwardNetwork":
                    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
                    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
                    # https://github.com/pytorch/pytorch/issues/57109
                    comp["kernel_initializer"] = {
                        "name": "kaiming_uniform",
                        "a": math.sqrt(5),
                    }
                    # Note: Using Kaiming_uniform directly on bias tensor
                    # results in PyTorch error:`ValueError: Fan in and fan out
                    # can not be computed for tensor with fewer than 2 dimensions`
                    # While this mismatches the src code, since we load from
                    # HF -> CS converted checkpoint, this is initialized in the
                    # checkpoint correctly
                    comp["bias_initializer"] = {
                        "name": "zeros",
                    }
        elif self.image_model.hidden_size != self.text_model.hidden_size:
            raise ValueError(
                f"The model should have a projector when the image model "
                f"and text model do not have the same `hidden_size`."
            )

        # Last encoder layer as default
        if self.image_feature_select_layer_idx:
            # Convert negative index and positive index representing layer_id of encoder to positive index. All indices are zero-based.
            self.image_feature_select_layer_idx = (
                self.image_feature_select_layer_idx
                % self.image_model.num_hidden_layers
            )

    def _get_submodel_handle(self, name):
        if name not in _submodel_mapping:
            raise KeyError(f"submodel '{name}' not mapped to any config.")
        return _submodel_mapping[name]


@dataclass
class LlavaDataConfig(DataConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        self.params.setdefault("use_worker_cache", False)
        super().__post_init__()
        self.params["data_processor"] = self.data_processor


@registry.register_config("llava")
@dataclass
class LlavaConfig(BaseConfig):
    train_input: Optional[LlavaDataConfig] = None
    "Input params class for train mode"

    eval_input: Optional[LlavaDataConfig] = None
    "Input params class for eval mode"

    model: LlavaModelConfig = required
    "Model level params class. Supported params differ for each model."

    optimizer: OptimizerConfig = required
    "Optimizer specific prameters captured in this class."

    runconfig: RunConfig = required
    "Params class to define params for controlling runs."

    sparsity: Optional[SparsityConfig] = None
    "Params class for sparsity related cofigurations"

    def __post_init__(self):
        super().__post_init__()
        self.set_config_defaults(
            self.model,
            self.train_input.params if self.train_input else None,
            [self.eval_input.params if self.eval_input else None],
        )

    @staticmethod
    def set_config_defaults(mparams, tparams, eparams_list):
        for model_key, input_key in [
            ("mixed_precision", "mixed_precision"),
            ("fp16_type", "fp16_type"),
            ("image_model.patch_size", "patch_size"),
            ("image_model.prepend_cls_token", "prepend_cls_token"),
            ("text_model.vocab_size", "params.vocab_size"),
        ]:
            for section in [tparams, *eparams_list]:
                if section is None:
                    continue

                if "." not in model_key:
                    section[input_key] = getattr(mparams, model_key)
                else:
                    # Split `image_model.patch_size`
                    # into ['image_model', 'patch_size'] and traverse
                    _key = model_key.split(".")
                    val = getattr(mparams, _key[0])
                    for _k in _key[1:]:
                        val = getattr(val, _k)
                    section[_key[-1]] = val

        for section in [tparams, *eparams_list]:
            if section is None:
                continue

            section.setdefault("use_worker_cache", False)
            section.setdefault("noaugment", False)

            # TODO: be consistent with naming `image_size` vs `image_data_size`
            section["image_data_size"] = [
                mparams.image_model.num_channels,
                *mparams.image_model.image_size,
            ]

            if len(section["image_data_size"]) == 3:
                section["num_patches"] = (
                    section["image_data_size"][-1]
                    // mparams.image_model.patch_size[0]
                ) * (
                    section["image_data_size"][-2]
                    // mparams.image_model.patch_size[1]
                )
            else:
                section["num_patches"] = section["image_data_size"][0]
                # when len(section["image_data_size"]) == 2, we assume image has already been
                # embedded and the ViT model is not included in the LLaVA model

        # model_keys refers to the name of keys which have models
        # and associated projectors if needed
        # So if model_keys = ["image_model", "image_model_1"],
        # we have params["model"]["image_model"], params["model"]["image_model_1"]
        # and projector params["model"]["projector"]["image_model"] and
        # projector params["model"]["projector"]["image_model_1"]

        mpparams = mparams.projector
        mixed_precision = mparams.mixed_precision
        fp16_type = mparams.fp16_type

        for component in ("image_model", "text_model"):
            for tgt_params, is_projector in [
                (mparams, False),
                (mpparams, True),
            ]:
                if not tgt_params:
                    continue

                if is_projector:
                    tgt_params_component = tgt_params.get(component, None)
                    getter = lambda k: tgt_params_component[k]

                    def setter(o, k, v):
                        o[k] = v

                else:
                    tgt_params_component = getattr(tgt_params, component, None)
                    getter = lambda k: getattr(tgt_params_component, k)
                    setter = setattr

                if tgt_params_component:
                    model = tgt_params_component
                    if hasattr(model, "mixed_precision"):
                        model.mixed_precision = mixed_precision
                    if hasattr(model, "fp16_type"):
                        model.fp16_type = fp16_type
                    if not is_projector:
                        _set_defaults_mapping[getter("name")](
                            model,
                            copy.deepcopy(tparams),
                            copy.deepcopy(eparams_list),
                        )
                    setter(tgt_params, component, model)
