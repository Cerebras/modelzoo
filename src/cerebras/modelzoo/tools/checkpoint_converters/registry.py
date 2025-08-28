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

# Separate file for converter list so it can be lazily imported
from typing import Dict, List

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter,
)
from cerebras.modelzoo.tools.checkpoint_converters.bert import (
    Converter_Bert_CS17_CS18,
    Converter_Bert_CS18_CS20,
    Converter_Bert_CS20_CS21,
    Converter_BertPretrainModel_CS16_CS17,
    Converter_BertPretrainModel_CS16_CS18,
    Converter_BertPretrainModel_HF_CS21,
    Converter_BertPretrainModel_HF_CS23,
)
from cerebras.modelzoo.tools.checkpoint_converters.bert_finetune import (
    Converter_BertFinetuneModel_CS16_CS17,
    Converter_BertFinetuneModel_CS16_CS18,
    Converter_BertForQuestionAnswering_HF_CS21,
    Converter_BertForSequenceClassification_HF_CS21,
    Converter_BertForTokenClassification_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.falcon import (
    Converter_Falcon_CS20_CS21,
    Converter_Falcon_Headless_HF_CS21,
    Converter_Falcon_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.gpt2_hf_cs import (
    Converter_GPT2LMHeadModel_CS18_CS20,
    Converter_GPT2LMHeadModel_CS20_CS21,
    Converter_GPT2LMHeadModel_CS22_CS23,
    Converter_GPT2LMHeadModel_HF_CS21,
    Converter_GPT2Model_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.gpt_neox_hf_cs import (
    Converter_GPT_Neox_Headless_HF_CS21,
    Converter_GPT_Neox_LMHeadModel_CS18_CS20,
    Converter_GPT_Neox_LMHeadModel_CS20_CS21,
    Converter_GPT_Neox_LMHeadModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.gptj_hf_cs import (
    Converter_GPTJ_Headless_HF_CS20,
    Converter_GPTJ_Headless_HF_CS23,
    Converter_GPTJ_LMHeadModel_CS18_CS20,
    Converter_GPTJ_LMHeadModel_CS20_CS21,
    Converter_GPTJ_LMHeadModel_HF_CS20,
    Converter_GPTJ_LMHeadModel_HF_CS23,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    Converter_LlamaForCausalLM_CS19_CS20,
    Converter_LlamaForCausalLM_CS20_CS21,
    Converter_LlamaForCausalLM_HF_CS21,
    Converter_LlamaModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.llava import (
    Converter_LLaVA_HF_CS22,
)
from cerebras.modelzoo.tools.checkpoint_converters.mm_simple import (
    Converter_MMSimple_LLaVA_HF_CS23,
    Converter_MMSimple_LLaVA_HF_CS24,
)
from cerebras.modelzoo.tools.checkpoint_converters.roberta import (
    Converter_RobertaPretrainModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.salesforce_codegen_hf_cs import (
    Converter_Codegen_Headless_HF_CS20,
    Converter_Codegen_LMHeadModel_CS18_CS20,
    Converter_Codegen_LMHeadModel_CS20_CS21,
    Converter_Codegen_LMHeadModel_HF_CS20,
)
from cerebras.modelzoo.tools.checkpoint_converters.t5 import (
    Converter_T5_CS16_CS17,
    Converter_T5_CS16_CS18,
    Converter_T5_CS17_CS18,
    Converter_T5_CS18_CS20,
    Converter_T5_CS20_CS21,
    Converter_T5_HF_CS21,
    Converter_T5_HF_CS23,
)

from cerebras.modelzoo.tools.checkpoint_converters.btlm_hf_cs import (  # noqa
    Converter_BTLMLMHeadModel_CS20_CS21,
    Converter_BTLMLMHeadModel_HF_CS21,
    Converter_BTLMModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.jais_hf_cs import (  # noqa
    Converter_JAISLMHeadModel_CS20_CS21,
    Converter_JAISLMHeadModel_HF_CS21,
    Converter_JAISModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.bloom_hf_cs import (  # noqa
    Converter_BloomLMHeadModel_CS19_CS20,
    Converter_BloomLMHeadModel_CS20_CS21,
    Converter_BloomLMHeadModel_HF_CS21,
    Converter_BloomModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.clip_vit import (  # noqa
    Converter_CLIPViT_Projection_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.dpo import (  # noqa
    Converter_DPO_HF_CS21,
    Converter_NON_DPO_TO_DPO_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.dpr import (  # noqa
    Converter_DPRModel_HF_CS23,
    Converter_DPRModel_CS22_CS23,
)

from cerebras.modelzoo.tools.checkpoint_converters.vit import (  # noqa
    Converter_ViT_Headless_HF_CS21,
    Converter_ViT_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.esm2 import (  # noqa
    Converter_Esm2PretrainModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.gemma2 import (  # noqa
    Converter_Gemma2ForCausalLM_HF_CS23,
)

from cerebras.modelzoo.tools.checkpoint_converters.dragon_plus import (  # noqa
    Converter_DragonModel_HF_CS23,
)


from cerebras.modelzoo.tools.checkpoint_converters.starcoder import (  # noqa
    Converter_StarcoderForCausalLM_HF_CS21,
    Converter_StarcoderModel_HF_CS21,
    Converter_StarcoderLMHeadModel_CS20_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.santacoder import (  # noqa
    Converter_SantacoderLMHeadModel_HF_CS21,
    Converter_SantacoderModel_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.mistral import (  # noqa
    Converter_MistralModel_HF_CS21,
    Converter_MistralForCausalLM_HF_CS21,
)

from cerebras.modelzoo.tools.checkpoint_converters.mixtral import (  # noqa
    Converter_MixtralModel_HF_CS23,
    Converter_MixtralForCausalLM_HF_CS23,
)

from cerebras.modelzoo.tools.checkpoint_converters.gpt_backbone import (  # noqa
    Converter_GPT2LMHeadModel_GPTBackboneLMHeadModel_CS24,
)

from cerebras.modelzoo.tools.checkpoint_converters.internal.qwen2 import (  # noqa
    Converter_Qwen2Model_HF_CS25,
    Converter_Qwen2ForCausalLM_HF_CS25,
)

converters: Dict[str, List[BaseCheckpointConverter]] = {
    "bert": [
        Converter_BertPretrainModel_HF_CS21,
        Converter_BertPretrainModel_HF_CS23,
        Converter_BertPretrainModel_CS16_CS17,
        Converter_BertPretrainModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS18_CS20,
        Converter_Bert_CS20_CS21,
    ],
    "bert-sequence-classifier": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
        Converter_BertForSequenceClassification_HF_CS21,
    ],
    "bert-token-classifier": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
        Converter_BertForTokenClassification_HF_CS21,
    ],
    "bert-summarization": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
    ],
    "bert-qa": [
        Converter_BertFinetuneModel_CS16_CS17,
        Converter_BertFinetuneModel_CS16_CS18,
        Converter_Bert_CS17_CS18,
        Converter_Bert_CS20_CS21,
        Converter_BertForQuestionAnswering_HF_CS21,
    ],
    "bloom": [
        Converter_BloomLMHeadModel_CS19_CS20,
        Converter_BloomLMHeadModel_CS20_CS21,
        Converter_BloomLMHeadModel_HF_CS21,
    ],
    "bloom-headless": [
        Converter_BloomModel_HF_CS21,
    ],
    "btlm": [
        Converter_BTLMLMHeadModel_CS20_CS21,
        Converter_BTLMLMHeadModel_HF_CS21,
    ],
    "btlm-headless": [Converter_BTLMModel_HF_CS21],
    "clip-vit": [Converter_CLIPViT_Projection_HF_CS21],
    "codegen": [
        Converter_Codegen_LMHeadModel_CS18_CS20,
        Converter_Codegen_LMHeadModel_CS20_CS21,
        Converter_Codegen_LMHeadModel_HF_CS20,
    ],
    "codegen-headless": [
        Converter_Codegen_Headless_HF_CS20,
    ],
    "esm-2": [Converter_Esm2PretrainModel_HF_CS21],
    "dpo": [
        Converter_DPO_HF_CS21,
        Converter_NON_DPO_TO_DPO_CS21,
    ],
    "dragon": [Converter_DragonModel_HF_CS23],
    "falcon": [
        Converter_Falcon_CS20_CS21,
        Converter_Falcon_HF_CS21,
    ],
    "falcon-headless": [
        Converter_Falcon_Headless_HF_CS21,
    ],
    "gemma2": [Converter_Gemma2ForCausalLM_HF_CS23],
    "gpt2": [
        Converter_GPT2LMHeadModel_CS18_CS20,
        Converter_GPT2LMHeadModel_CS20_CS21,
        Converter_GPT2LMHeadModel_CS22_CS23,
        Converter_GPT2LMHeadModel_HF_CS21,
    ],
    "gpt2-headless": [
        Converter_GPT2Model_HF_CS21,
    ],
    "gptj": [
        Converter_GPTJ_LMHeadModel_CS18_CS20,
        Converter_GPTJ_LMHeadModel_CS20_CS21,
        Converter_GPTJ_LMHeadModel_HF_CS20,
        Converter_GPTJ_LMHeadModel_HF_CS23,
    ],
    "gptj-headless": [
        Converter_GPTJ_Headless_HF_CS20,
        Converter_GPTJ_Headless_HF_CS23,
    ],
    "gpt-neox": [
        Converter_GPT_Neox_LMHeadModel_CS18_CS20,
        Converter_GPT_Neox_LMHeadModel_CS20_CS21,
        Converter_GPT_Neox_LMHeadModel_HF_CS21,
    ],
    "gpt-neox-headless": [
        Converter_GPT_Neox_Headless_HF_CS21,
    ],
    "jais": [
        Converter_JAISLMHeadModel_CS20_CS21,
        Converter_JAISLMHeadModel_HF_CS21,
    ],
    "llama": [
        Converter_LlamaForCausalLM_CS19_CS20,
        Converter_LlamaForCausalLM_CS20_CS21,
        Converter_LlamaForCausalLM_HF_CS21,
    ],
    "llama-headless": [
        Converter_LlamaModel_HF_CS21,
    ],
    "llava": [Converter_LLaVA_HF_CS22],
    "mmsimple-llava": [
        Converter_MMSimple_LLaVA_HF_CS23,
        Converter_MMSimple_LLaVA_HF_CS24,
    ],
    "mistral": [Converter_MistralForCausalLM_HF_CS21],
    "mistral-headless": [Converter_MistralModel_HF_CS21],
    "mixtral": [Converter_MixtralForCausalLM_HF_CS23],
    "mixtral-headless": [Converter_MixtralModel_HF_CS23],
    "roberta": [
        Converter_RobertaPretrainModel_HF_CS21,
        Converter_Bert_CS20_CS21,
    ],
    "santacoder": [Converter_SantacoderLMHeadModel_HF_CS21],
    "santacoder-headless": [Converter_SantacoderModel_HF_CS21],
    "starcoder": [
        Converter_StarcoderForCausalLM_HF_CS21,
        Converter_StarcoderLMHeadModel_CS20_CS21,
    ],
    "starcoder-headless": [
        Converter_StarcoderModel_HF_CS21,
    ],
    "t5": [
        Converter_T5_CS16_CS17,
        Converter_T5_CS16_CS18,
        Converter_T5_CS17_CS18,
        Converter_T5_CS18_CS20,
        Converter_T5_CS20_CS21,
        Converter_T5_HF_CS21,
        Converter_T5_HF_CS23,
    ],
    "qwen2": [Converter_Qwen2ForCausalLM_HF_CS25],
}

# Add some model aliases
converters["ul2"] = converters["t5"]
converters["flan-ul2"] = converters["t5"]
converters["transformer"] = converters["t5"]
converters["llamaV2"] = converters["llama"]
converters["llamaV2-headless"] = converters["llama-headless"]
converters["code-llama"] = converters["llama"]
converters["code-llama-headless"] = converters["llama-headless"]
converters["octocoder"] = converters["starcoder"]
converters["octocoder-headless"] = converters["starcoder-headless"]
converters["wizardcoder"] = converters["starcoder"]
converters["wizardcoder-headless"] = converters["starcoder-headless"]
converters["sqlcoder"] = converters["starcoder"]
converters["sqlcoder-headless"] = converters["starcoder-headless"]
converters["wizardlm"] = converters["llama"]
converters["wizardlm-headless"] = converters["llama-headless"]


# A mapping between model names in the converter to the model names in the
# ModelZoo registry
_model_aliases = {
    "bert-sequence-classifier": "bert/classifier",
    "bert-token-classifier": "bert/token_classifier",
    "bert-summarization": "bert/extractive_summarization",
    "bloom-headless": "bloom",
    "clip-vit": "vision_transformer",
    "codegen": "gptj",
    "codegen-headless": "gptj",
    "code-llama": "llama",
    "code-llama-headless": "llama",
    "dino-v2": "vision_transformer",
    "dino-v2-headless": "vision_transformer",
    "esm-2": "esm2",
    "falcon-headless": "falcon",
    "fid": "fido",
    "gpt2-headless": "gpt2",
    "gptj-headless": "gptj",
    "gpt-neox-headless": "gpt-neox",
    "opt-headless": "opt",
    "octocoder": "starcoder",
    "octocoder-headless": "starcoder",
    "llama-headless": "llama",
    "mistral-headless": "mistral",
    "mmsimple-llava": "multimodal_simple",
    "decoder-only-with-moe": "moe",
    "roberta": "bert",
    "swin": "swin/classifier",
    "swin-headless": "swin/classifier",
    "swin-v2": "swin/classifier",
    "swin-v2-headless": "swin/classifier",
    "swin-mim": "swin/mim",
    "swin-v2-mim": "swin/mim",
    "vit": "vision_transformer",
    "vit-headless": "vision_transformer",
    "vit-mae": "vit_mae",
    "wizardcoder": "gpt2",
    "wizardcoder-headless": "gpt2",
    "wizardlm": "llama",
    "wizardlm-headless": "llama",
    "xlm-headless": "xlm",
}


def get_cs_model_name(model, raise_error=True):
    from cerebras.modelzoo.registry import registry

    model = _model_aliases.get(model, model)
    if model in registry.get_model_names():
        return model

    if raise_error:
        raise ValueError(f"Model {model} not found in the ModelZoo registry")
    return None


def get_cs_model_wrapper_class(model):
    from cerebras.modelzoo.registry import registry

    return registry.get_model_class(get_cs_model_name(model))
