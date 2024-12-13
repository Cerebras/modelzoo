
# Multimodal Model
- [Multimodal Model](#multimodal-llava)
  - [Model overview](#model-overview)
  - [Structure of the code](#structure-of-the-code)
  - [Model Training Approach](#structure-of-the-code)
  - [Steps to train a model](#structure-of-the-code)
  - [Structure of the code](#structure-of-the-code)
  - [Configuration files included for this model](#config)
  - [Implementation notes](#impl)
  - [Citations](#citation)


## Model Overview

This directory contains our multimodal library, which can be used to instantiate many of the current state-of-the-art models such as LLaVA, CogVLM, and MM1 among others. Our implementation supports multiple images interleaved with text as input, and can generate text as output. The building blocks for this implementation are as follows:
- **Vision Encoder**: Process images through one or more image encoders to produce embeddings.
- **Image Embedding Projector**: Projects the embeddings from vision encoder into a shared latent space with LLM using MLPs. 
- **Language Model**: Accepts the vision and language embeddings as input and produces text as output.

## Structure of the code

-   `configs/`: YAML configuration files.
-   `modeling_mmsimple.py`: Defines the core multimodal model.
-   `model.py`: The entry point to the model.
-   `run.py`: Training script. Performs training and validation.

## Model Training Approach

A common approach to build high-quality multimodal models with limited data is to initialize the vision encoder and language models from pretrained checkpoints (for instance CLIP-VIT-L-336/14 for vision and LLaMA/Mistral/Zephyr models for language). While there are many possible recipes for training the model, the high-level goals are as follows:
- **Pre-training for Feature Alignment**: This involves training the randomly-initialized projector weights to align the image features with that of the LLM embeddings. Optionally, this could also involve training all the blocks -- vision encoder, llm and projector together for further alignment of modalities.

- **Instruction Fine-tuning**: In this stage, the model is trained to handle multimodal question-answering and dialogue.

## Steps to train a model

The high-level steps for training this model are consistent with other models such as LLMs
- Dataset preparation: Download datasets of interest and process them using our data pre-processing scripts to generate H5 files
- Checkpoint preparation: Download pretrained checkpoints for vision and language models to prepare the initial checkpoint
- Training: Train the model using `run.py`
- Export to HF: Convert checkpoint to HF checkpoint format
- Evaluation: Use standard multimodal benchmarks such lmms-eval or LLaVA source-repo

### Step 1: Dataset Prep
Please follow instructions for data preprocessing in our documentation.

### Step 2: Checkpoint Prep


Checkpoint converter script for converting vision encoder and LLM model checkpoints to CS format require the following directory structure:
```
/path/to/pretrained/checkpoints
├── image_model
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
└── text_model
    ├── config.json
    ├── config_lmsys.json
    ├── pytorch_model-00001-of-00002.bin
    ├── pytorch_model-00002-of-00002.bin
    ├── pytorch_model.bin.index.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── tokenizer.model
```

Below, we describe how to setup CLIP + LLAMA3 8B as a LLaVA model. However, one can follow same approach to setup other vision and LLM models as well as multimodal architecture variants.

a.  [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) checkpoints, config.json and preprocessor_config.json should be downloaded to a subdirectory `image_model`. 

b. [LLAMA3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) checkpoints and tokenizer files should be downloaded to a subdirectory `text_model`

c. Rename `config.json` to `config_lmsys.json`

    mv /path/to/pretrained/checkpoints/text_model/config.json /path/to/pretrained/checkpoints/text_model/config_lmsys.json


d. Download [LLaVA-8B config.json](https://huggingface.co/lmms-lab/llama3-llava-next-8b/blob/main/config.json) from HuggingFace

We do steps (c) and (d) above since we need additional information about LLaVA model such as `mm_projector_type` etc to build appropriate CS config yaml and checkpoint

### Step 4: Convert checkpoints to CS Model Zoo format using checkpoint converter

* Checkpoint conversion script: [modelzoo/tools/convert_checkpoint.py](../../../tools/convert_checkpoint.py)
* LLaVA Model checkpoint converter: [[modelzoo/tools/checkpoint_converters/mm_simple.py](../../../tools/checkpoint_converters/llava.py)]
* Command:
  ```
  python modelzoo/tools/convert_checkpoint.py convert \
  --model mm_simple \
  --src-fmt hf \
  --tgt-fmt cs-2.3 \
  --output-dir /path/to/converted_checkpoint \
  --config /path/to/pretrained/checkpoints \
  /path/to/pretrained/checkpoints
  ```

More information about checkpoint converters can be obtained by
`python modelzoo/tools/convert_checkpoint.py list`

### Step 3: Training the model on CS system using `run.py`

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](configs/) directory. Before starting a training run, make sure that the YAML configs used have the following set correctly:

-   The `train_input.data_dir` parameter points to the correct dataset
-   The `train_input.img_data_dir` parameter points to the correct parent directory containing all images needed by the dataset.
-   The `train_input.image_size` parameter corresponds to the image-size of the dataset.
-   Also change sizes in `train_input.transforms` appropriately if `train_input.image_size` is updated.

-   The `image_model.image_size` points to the image size passed to each ViTModel
-   The `image_model.patch_size` parameter to use different patch sizes within each ViTModel
-  `model.freeze` contains the regex patterns to freeze appropriate layers in the model
-  `image_model.image_layer_idx` parameter to specify the image_model encoder layer from which features are extracted for the input image.

**YAML config files**: Details on the configs for this model can be found in [Configs included for this model](#configs-included-for-this-model)

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

-   `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in [Configs included for this model](#configs-included-for-this-model).
-   `/path/to/model_dir` is a path to the directory where we would like to store the logs and other artifacts of the run.
-   `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode.

#### To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our [quickstart in the Developer Docs](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).

#### To run train and eval on GPU/CPU

If running on a CPU or GPU, activate the environment from [Python GPU Environment setup](../../../../../../PYTHON-SETUP.md), and simply run:

```
python run.py {CPU,GPU} \
--mode train \
--params /path/to/yaml \
--model_dir /path/to/model_dir
```

Based on your training recipe, you might choose to train the multimodal in multiple phases. Please specify the params file and model directory for each run -- this is consistent with any multi-stage training appraoch on Cerebras.


### Step 4: Convert checkpoint to source code repository format to run eval

We perform evaluations on multimodal benchmarks using [LLaVA source code](https://github.com/haotian-liu/LLaVA) repository. For this, we need to convert the checkpoints generated using the training run from Phase-2 to LLaVA source code repository format. This can be done using the command:

```
python modelzoo/tools/convert_checkpoint.py convert \
--model mm_simple \
--src-fmt cs-2.3 \
--tgt-fmt hf \
--output-dir /path/to/hf_converted_checkpoint \
--config /path/to/cs_config.yaml \
/path/to/checkpoint.mdl
```

The above command generates two folders `image_model` and `text_model` under `output-dir` as shown below:
  ```
  /path/to/converted_checkpoint
  ├── image_model
  │   ├── config.json
  │   ├── preprocessor_config.json
  │   ├── pytorch_model-00001-of-00001.bin
  │   └── pytorch_model.bin.index.json
  └── text_model
      ├── config.json
      ├── pytorch_model-00001-of-00004.bin
      ├── pytorch_model-00002-of-00004.bin
      ├── pytorch_model-00003-of-00004.bin
      ├── pytorch_model-00004-of-00004.bin
      ├── pytorch_model.bin.index.json
  ```
* Folder `image_model` consists of weights for `vision_tower` in source repository
* Folder `text_model` consists of weights to be loaded for the Language model and projectors
  
* The LLaVA source code repository expects tokenizer files to be present along with the language model weights ([code pointer](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/builder.py#L116)). For this, **please copy the tokenizer files into `text_model` folder**.
  
* Also, please make sure `text_model.mm_vision_tower` points to the `image_model` path to ensure the weights from `image_model` folder are loaded into the source code `vision_tower`. This path is automatically added during checkpoint conversion.

* **Rename folder `text_model` to `text_model_llava`. This is since the source code repository expects the path to include `llava` keyword in order to correctly load the checkpoints. (code pointers: [builder.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/builder.py#L48), [mm_utils.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py#L207))**
  
* So, after the relevant tokenizer files are copied, the `output-dir` should look like below:
  ```
  /path/to/converted_checkpoint
  ├── image_model
  │   ├── config.json
  │   ├── preprocessor_config.json
  │   ├── pytorch_model-00001-of-00001.bin
  │   └── pytorch_model.bin.index.json
  └── text_model_llava
      ├── config.json
      ├── pytorch_model-00001-of-00004.bin
      ├── pytorch_model-00002-of-00004.bin
      ├── pytorch_model-00003-of-00004.bin
      ├── pytorch_model-00004-of-00004.bin
      ├── pytorch_model.bin.index.json
      ├── special_tokens_map.json
      ├── tokenizer_config.json
      └── tokenizer.model
  ```

### Step 5: Set up source code repository for benchmark evaluation and run evaluation benchmarks

* Setup LLaVA source code repository for multimodal benchmark evaluation by following instructions mentioned in [Evaluation Docs](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). 
*  Instructions for creating conda environment and setting up the repository are mentioned in [Installation Section](https://github.com/haotian-liu/LLaVA/blob/main/README.md#install)
*  Scripts to run various benchmarks are provided [here](https://github.com/haotian-liu/LLaVA/tree/main/scripts/v1_5/eval)
*  **Pass `text_model_llava` folder path to `--model-path` in [eval scripts in LLaVA source code repository](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/llavabench.sh#L4)**

## Configuration files included for this model

We provide the following config files for LLaVA located under the [configs](configs) directory.

| Config File | Dataset |  Notes |
| ------------- | ------------- | ------------- | 
|[params_mm_llava_llama2_7b_phase1.yaml](./configs/params_mm_llava_llama2_7b_phase1.yaml) | [LLaVA Visual Instruct Pretrain LCS-558K Dataset](#llava-visual-instruct-pretrain-lcs-558k-dataset) | LLaVA-7B Phase-1 with CLIP ViT image encoder, Vicuna-7B text encoder and `mlp2x-gelu` Feedforward network for Projector. Freeze `image_model` and `text_model` during training  |
|[params_mm_llava_llama2_7b_phase2.yaml](./configs/params_mm_llava_llama2_7b_phase2.yaml) | [LLaVA Visual Instruct 150K Dataset](#llava-visual-instruct-150k-dataset) | LLaVA-7B Phase-2 with CLIP ViT image encoder, Vicuna-7B text encoder and `mlp2x-gelu` Feedforward network for Projector. Freeze `image_model` during training|


## Implementation notes

The following modifications and assumptions are made in this implementation:

1. This implementation brings in support for multiple images per sample, interleaving of image and text, as well as placement of images at arbitrary position within the sample. However, currently we do not have checkpoint converter for these new features since there is no one HF model that supports these multimodal features.
2. This implementation assumes that the H5 files for the dataset are created with the release 2.3 data preprocessor, and is not backward compatible with the H5 datasets produced with the previous release (2.2).
3. **We currently expect all the images under a single parent folder and the relative path of images from different datasets are written under `image_key` in the H5 files generated.**
   For example:
   ```
    train_input.img_data_dir
    ├── coco
    ├── gqa
    ├── ocr_vqa
    ├── textvqa
    └── vg
   ```

## Citations

[1] [LLaVA-v1: Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485.pdf)

[2] [LLaVA-v1.5: Improved Baselines with Visual Instruction Tuning](https://arxiv.org/pdf/2310.03744.pdf)

[3] [LLaVA source code repository](https://github.com/haotian-liu/LLaVA/tree/main/)

[4] [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)

[5] [CogVLM: Visual Expert for Pretrained Language Models](https://arxiv.org/abs/2311.03079)

