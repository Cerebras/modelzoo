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
from cerebras.modelzoo.config_manager.config_classes.base.optimizer_config import (
    OptimizerConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.run_config import (
    RunConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.sparsity_config import (
    SparsityConfig,
)
from cerebras.modelzoo.models.nlp.gpt2.config import GPT2LMHeadModelConfig
from cerebras.modelzoo.models.nlp.t5.config import (
    T5ForConditionalGenerationModelConfig,
)
from cerebras.modelzoo.models.vision.vision_transformer.config import (
    ViTModelConfig,
)

_submodel_mapping = {
    "ViTModel": ViTModelConfig,
    "LlamaModel": GPT2LMHeadModelConfig,
    "T5ForConditionalGeneration": T5ForConditionalGenerationModelConfig,
}


def _get_submodel_handle(name):
    if name not in _submodel_mapping:
        raise KeyError(f"submodel '{name}' not mapped to any config.")
    return _submodel_mapping[name]


@dataclass
class MultimodalImageModel(BaseConfig):
    image_model: List[dict] = required
    "Image encoder model when only one image encoder is needed. Note either image_model OR image_model_list must be specified"


@dataclass
class MultimodalImageModelList(BaseConfig):
    image_feature_select_mode: str = "patch"
    image_models: List[dict] = None
    "List of image MultimodalImageModels"
    global_image_projection: Optional[dict] = None
    "Projector model that is applied to the concat of the output of all the image models.  Options include MLP, Q-Former, Pooling, Identity"

    @staticmethod
    def is_image_model(field_dict):
        if (field_dict is not None) and ("image_model" in field_dict):
            return True
        else:
            return False

    def __post_init__(self):
        super().__post_init__()
        for image_plus_proj in self.image_models:
            assert (len(image_plus_proj) == 1) and (
                "image_model" in image_plus_proj
            )
            field_dict = image_plus_proj['image_model'][0]
            submodel_handle = _get_submodel_handle(field_dict["name"])
            image_plus_proj['image_model'][0] = submodel_handle(**field_dict)


@dataclass
class MultimodalDecoderModelConfig(BaseConfig):
    freeze: Optional[List[str]] = None
    """
    Allows us to freeze parts of the model through just yaml file manipulation. 
    Filter to select which parameters are frozen. 
    Note that regex patterns should be specified as single quotes in the yaml for escape codes
    """

    image_model_list: MultimodalImageModelList = required
    "Allows the model to instantiate a list of image models that run same image through multiple image encoders"

    text_model: dict = required
    "Decoder-only LLM model that processes all the modalities together through the backbone and produces output"

    output_list: Optional[list] = None
    "List of intermediate values that should be returned. Options include: image, image_encoder_out, projector_out. Model always returns output of VLLMModel"
    loss_scaling: Literal["batch_size", "num_tokens"] = "num_tokens"
    """The scaling type used to calculate the loss. Accepts - `batch_size`, `num_tokens`.
    See [more](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html).
    **Note:** It is recommended to set this to `num_tokens` for convenience."""

    loss_weight: float = 0.0
    """The weight for the loss scaling when `loss_scaling = 'batch_size'`, generally set to
    '1/max_sequence_length`.
    """
    mixed_precision: bool = True
    fp16_type: str = "bfloat16"

    def __post_init__(self):
        super().__post_init__()
        image_model_kwargs = {}
        image_model_kwargs["image_model"] = []
        for component in [
            "text_model",
        ]:
            submodel_handle = _get_submodel_handle(
                getattr(self, component)["name"]
            )
            setattr(
                self, component, submodel_handle(**getattr(self, component))
            )


@dataclass
class MultimodalSimpleDataConfig(DataConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        self.params.setdefault("shuffle", False)
        self.params.setdefault("shuffle_seed", 274508134)
        self.params.setdefault("noaugment", False)
        self.params.setdefault("drop_last", True)
        self.params.setdefault("num_workers", 0)
        self.params.setdefault("prefetch_factor", 10)
        self.params.setdefault("persistent_workers", True)
        self.params.setdefault("use_worker_cache", False)
        self.params.setdefault("max_num_img", 1)
        if self.params["noaugment"]:
            self.params["transforms"] = None

        super().__post_init__()


@registry.register_config("multimodal_simple")
@dataclass
class MultimodalSimpleDecoderConfig(BaseConfig):
    train_input: Optional[MultimodalSimpleDataConfig] = None
    "Dataloader config for the train mode"

    eval_input: Optional[MultimodalSimpleDataConfig] = None
    "Dataloader config for the eval mode"

    model: MultimodalDecoderModelConfig = required
    "Model Arch config"

    sparsity: Optional[SparsityConfig] = None
    optimizer: OptimizerConfig = required
    runconfig: RunConfig = required

    def __post_init__(self):
        super().__post_init__()
        self.set_config_defaults(
            self.model,
            self.train_input.params if self.train_input else None,
            [self.eval_input.params if self.eval_input else None],
        )

    @staticmethod
    def set_config_defaults(mparams, tparams, eparams_list):
        # Copy params from model section to
        # train_input and eval_input section
        for section in [tparams, *eparams_list]:
            if section is None:
                continue

            section["mixed_precision"] = mparams.mixed_precision
            section["fp16_type"] = mparams.fp16_type
            section["vocab_size"] = mparams.text_model.vocab_size

            # extract image info from each image encoder.
            image_sizes = []
            patch_sizes = []
            num_c = []
            prepend_cls = []
            image_feature_select_mode = "patch"
            image_feature_select_mode = (
                mparams.image_model_list.image_feature_select_mode
            )
            for im_param in mparams.image_model_list.image_models:
                if "image_model" in im_param:
                    im = im_param["image_model"][0]
                    if hasattr(im, "image_size"):
                        image_sizes.append(im.image_size)
                    else:
                        raise ValueError(
                            f"The image size should be provided for all image encoders."
                        )

                    if hasattr(im, "num_channels"):
                        num_c.append(im.num_channels)
                    else:
                        raise ValueError(
                            f"The num_channels should be provided for all image encoders."
                        )

                    if hasattr(im, "patch_size"):
                        patch_sizes.append(im.patch_size)
                    else:
                        raise ValueError(
                            f"The patch_size should be provided for all image encoders."
                        )

                    if hasattr(im, "prepend_cls_token"):
                        prepend_cls.append(im.prepend_cls_token)
                    else:
                        raise ValueError(
                            f"The prepend_cls_token should be provided for all image encoders."
                        )

            if not all(i == image_sizes[0] for i in image_sizes):
                raise ValueError(
                    f"The image sizes for all image encoders should be the same."
                )
            if not all(i == num_c[0] for i in num_c):
                raise ValueError(
                    f"The num_channels for all image encoders should be the same."
                )
            if (
                len(image_sizes) != len(patch_sizes)
                or len(image_sizes) != len(num_c)
                or len(image_sizes) != len(prepend_cls)
            ):
                raise ValueError(
                    f"Each image encoder should have both 'image_size', 'num_channels', 'prepend_cls_token' and 'patch_size'."
                )

            section["image_data_size"] = [
                num_c[0],
                *image_sizes[0],
            ]

            # Compute num_patches for each image encoder and sum them up.
            num_img_encoder = len(image_sizes)
            num_patches = []
            for idx in range(num_img_encoder):
                if len(section["image_data_size"]) == 3:
                    num_p = (
                        section["image_data_size"][-1] // patch_sizes[idx][0]
                    ) * (section["image_data_size"][-2] // patch_sizes[idx][1])
                else:
                    num_p = section["image_data_size"][0]

                if prepend_cls[idx]:
                    num_p += 1

                if image_feature_select_mode == "patch":
                    num_p -= 1

                num_patches.append(num_p)

            section["num_patches"] = sum(num_patches)
