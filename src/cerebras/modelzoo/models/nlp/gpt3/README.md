# GPT-3 Language Models

This directory contains the PyTorch ML reference for GPT-2 and GPT-3 models.

- [GPT-3 Language Models](#gpt-3-language-models)
  - [Overview of the model](#overview-of-the-model)
  - [Structure of the code](#structure-of-the-code)
  - [Prepare the data](#prepare-the-data)
      - [GPT-3 DataProcessor output](#gpt-3-dataprocessor-output)
  - [GPT-3 input function](#gpt-3-input-function)
      - [GPT-3 features dictionary](#gpt-3-features-dictionary)
- [How to run](#how-to-run)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
  - [Configs included for this model](#configs-included-for-this-model)
  - [Maximal Update Parameterization](#maximal-update-parameterization)
      - [&mu;Transfer Methodology](#μtransfer-methodology)
      - [&mu;P configuration for a GPT-3 run](#μp-configuration-for-a-gpt-3-run)
      - [Convert a config to a &mu;P config](#convert-a-config-to-a-μP-config)
  - [Appendix](#appendix)

## Overview of the model

[GPT-3](https://arxiv.org/abs/2005.14165) is a very similar architecture to [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) except that every other self-attention layer in GPT-3 uses locally banded sparse attention in which tokens only attend to each other if they are nearby in the sequence
(see section 2.1 of the [GPT-3 paper](https://arxiv.org/abs/2005.14165) for more details). Figure below describes a high level model architecture of GPT3 model.

![GPT3 Architecture Diagram](./images/architecture_diagram.png)

The larger versions of GPT-3 range from 1.3B to 175B parameters.

**NOTE:** In our current implementation, we use the code from [GPT2 implementation](../gpt2/) which does not have banded sparse attention implemented. We plan to add this support in the future releases.

## Structure of the code

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Prepare the data

You need to download raw PILE data following [these instructions](../../../data_preparation/nlp/pile/) and create preprocessed dataset files using [`create_hdf5_dataset.py`](../../../data_preparation/nlp/hdf5_preprocessing/).

#### GPT-3 DataProcessor output
  The `GptHDF5DataProcessor` class in [`GptHDF5DataProcessor.py`](../../../data/nlp/gpt/GptHDF5DataProcessor.py) creates `example_dict` iterative from the `self.features_list` which is returned on the call iteratively. 
 
## GPT-3 input function

If you want to use your own data loader with this example code, then this section describes the input data format expected by `Gpt2Model` class defined in [model.py](../gpt2/model.py). The `Gpt2Model` supports GPT-2 and GPT3 model architecture.

When you create your own custom GPT input function, you must ensure that your GPT input function produces a features dictionary as described in this section.

#### GPT-3 features dictionary

The features dictionary has the following key/values:

- `input_ids`: Input token IDs, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `attention_mask`: Mask for padded positions. Has values `0` on the padded positions and `1` elsewhere.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `labels`: Labels for language modeling pre-training task, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`

# How to run

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](configs/) directory. Before starting a pre-training run, make sure that in the YAML config file you are using:

-   The `train_input.data_dir` parameter points to the correct dataset, and
-   The `train_input.max_sequence_length` parameter corresponds to the sequence length of the dataset.
-   The `model.max_position_embeddings` parameter corresponds to the maximum dimension of position embeddings.

**YAML config files**: Details on the configs for this model can be found in [Configs included for this model](#configs-included-for-this-model)

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

-   `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in [Configs included for this model](#configs-included-for-this-model).
-   `/path/to/model_dir` is a path to the directory where you would like to store the logs and other artifacts of the run.
-   `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode.

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our [quickstart in the Developer Docs](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../../../PYTHON-SETUP.md), and simply run:

```
python run.py {CPU,GPU} --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```
## Configs included for this model

For convenience, we provide different configurations of common model setups designed to give examples of models of different sizes.

- [params_gpt3_xl.yaml](./configs/params_gpt3_xl.yaml): A 1.3B parameter model designed to match the configuration of the GPT-3 XL model.
- [params_gpt3_2p7b.yaml](./configs/params_gpt3_2p7b.yaml): A 2.7B parameter GPT-2 model designed to match the configuration of the GPT-3 6.7B model.
- [params_gpt3_6p7b.yaml](./configs/params_gpt3_6p7b.yaml): A 6.7B parameter GPT-2 model designed to match the configuration of the GPT-3 6.7B model.
- [params_gpt3_13b.yaml](./configs/params_gpt3_13b.yaml): A 13B parameter GPT-2 model designed to match the configuration of the GPT-3 13B model. Available as an early limited access.
- [params_gpt3_20b.yaml](./configs/params_gpt3_20b.yaml): A 20B parameter GPT-2 model designed to match the configuration of the GPT-NeoX. Available as an early limited access.

Additionally, the configs under [Cerebras_GPT](./configs/Cerebras_GPT/) are the configurations necessary to reproduce the results in our [Cerebras-GPT Blog](https://www.cerebras.net/cerebras-gpt).

> **NOTE:** The 1.3b(xl), 2.7b, 6.7b and 13b configs above show an example of setting micro batch size explicitly in the `train_input` section of the config. Without this setting, the best micro batch size search will be performed automatically during compilation which could take long time for larger models.
> **NOTE**: In absence of banded sparse attention feature, the GPT3 small, medium and large models are equivalent to the corresponding GPT2 variants available in [gpt2 configs](../gpt2/configs/) directory.

## Maximal Update Parameterization
[&mu;P (Maximal Update Parameterization)](https://arxiv.org/abs/2203.03466) benefits in two ways: i) Stable training dynamics at large scale by controlling the initialization, activations magnitude and layer-wise adaptive learning rates independent of model width, ii) it allows for zero-shot hyperparameter transfer from a smaller model to larger models. Essentially, muP facilitates width invariance to the model’s hyperparameters. 

Standard Parameterization (SP) (default hyperparameters and initialization scheme used for GPT-X models) doesn’t account for inter-layer interactions which could result in unstable training dynamics as they hit the limit of numerical representations. Therefore, hyperparameters are not transferable as the width of the model scales. 

### &mu;Transfer Methodology
&mu;Transfer is a hyperparameter transfer paradigm which makes zero-shot transfer of near optimal hyperparameters possible from a small version of model to a large model via &mu;P (Maximal Update Parameterization). The "small model" is called the *proxy-model* for which the hyperparameters are tuned and the "large model" is referred to as the *target-model*. 

Following table shows the &mu;Transferable hyperparameters:
| &mu;Transferable  | Not &mu;Transferable | Not &mu;Transferred Across
| ------------- | ------------- | ------------- |
| optimization related, init, parameter multipliers, etc | regularization  |  width, depth*, batch size*, training time*, seq length*|

Following notation is followed in this document:
|Variable||
|-------|-------|
|W|Weights tensor|
|b|Bias weights tensor|
|$d_{model,0}$|Proxy model's width|
|$d_{model}$|Model width|
|$\tilde{d}_{model}$|Width multiplier (d<sub>model</sub> / d<sub>model,0</sub>) |
|$d_{head}$|Attention head size|
|$\eta_{base}$|Base learning rate|
|$\sigma_{base}$|Base initialization standard deviation|
|embed|Combined token and position embedding function|
|$m_{emb}$| Embedding output multiplier|
|X|Layer input activation tensor|
|Y|Layer output activation tensor|


There are three &mu;Transferable hyperparameters which are tuned for the *proxy-model*, $\sigma_{base}$, $\eta_{base}$ and $m_{emb}$.

The table below depicts the &mu;Transfer of initialization variance, learning rate and output scaling:
| Tensors           | Initializer            |          Learning Rate |            Output |
| -------------    | ------------- | ------------- | ---------------------|
| Embeddings |  $N_{trunc}(0, \sigma_{base}^{2})$ | $\eta_{base}$ | $m_{emb} . embed(X)$|
| LN | $W^{LN}$ ~ 1, $b^{LN}$ ~ 0 | $\eta_{base}$ | - |
| Bias | b~0 | $\eta_{base}$ | - |
|Attention Logits| - | - | ($Q^TK/d_{head}$)V | 
| QKV Weights |  $N_{trunc}$(0, $\sigma_{base}^{2}/\tilde{d}_{model}$) | $\eta_{base}/\tilde{d}_{model}$ | - |
| Attention Output Weights |  $N_{trunc}(0, \sigma_{base}^{2}/(2.\tilde d_{model}.n_{layers}))$ | $\eta_{base}/\tilde{d}_{model}$ | - |
| FFN1 Weights  |  $N_{trunc}(0, \sigma_{base}^{2}/\tilde{d}_{model})$ | $\eta_{base}/\tilde{d}_{model}$ | - |
| FFN2 Weights |  $N_{trunc}(0, \sigma_{base}^{2}/(2.\tilde d_{model}.n_{layers}))$ | $\eta_{base}/\tilde{d}_{model}$ | - |
|Output Logits| - | - | $W_{unemb}X/\tilde{d}_{model}$ |


### &mu;P configuration for a GPT-3 run
GPT-3 model supports &mu;Transfer of near optimal hyperparameters to the *target-model* which are tuned for the *proxy-model*. $d_{model,0}=256$ can be used as the width of the *proxy-model* in the hyperparameter search. Once you have the optimal set of hyperparameters ($\sigma_{base}$, $\eta_{base}$ and $m_{emb}$), use the table below to deduce the configuration settings for the *target-model*.

| Parameter |       Usage           |Data Type      |  Value   |
| -------------    | ------------- | ------------- | ------|
|model.scale_qk_dot_by_d| ($Q^TK/d_{head}$)V | Bool | True |
|model.output_logits_scale| $Y_{logits} = W_{unemb}X/\tilde{d}_{model}$| Float| $1/\tilde{d}_{model}$|
|model.embeddings_scale| $Y_{embd} = m_{emb} . embed(X)$ | Float| $m_{emb}$ (tunable) |
|optimizer.adjust_learning_rate| $\eta_{base}/\tilde{d}_{model}$  | Dict | 'decoder_kernel': $1/\tilde{d}_{model}$ |
|optimizer.embedding_initializer| $W_{emb}$ ~ $N_{trunc}$(0, $\sigma_{base}^{2}$) | Dict |'std': $\sigma_{base}^2$|
|optimizer.initializer| $W_{emb}$ ~ $N_{trunc}$(0, $\sigma_{base}^{2}$) | Dict | 'std': $\sigma_{base}^{2}$ |
|optimizer.initializer| $W_{qkv}$ ~ $N_{trunc}$(0, $\sigma_{base}^{2}/\tilde{d}_{model}$) | Dict | 'std': $\sigma_{base}^{2}/\tilde{d}_{model}$ |
|optimizer.output_initializer| $W_{o},W_{FFN2}$ ~ $N_{trunc}(0, \sigma_{base}^{2}/(2.\tilde d_{model}.n_{layers}))$ | Dict | 'std': $\sigma_{base}^{2}/(2.\tilde d_{model}.n_{layers})$ |

Example configuration with $d_{model}$ = 1088 and $d_{model,0}$ = 256 ( $\tilde d_{model}$= 1088/256 = 4.25), $\sigma_{base}$=0.08, $\eta_{base}=6e-3$ and $m_{emb}=10$. These base hyperparameter values are taken from [Cerebras-GPT](https://arxiv.org/abs/2304.03208):
```yaml
model:  
  embedding_initializer:
      mean: 0.0
      name: truncated_normal
      std: 0.08 #base_initialization_std = 0.08
      a: -0.16
      b: 0.16
  initializer:
      mean: 0.0
      name: truncated_normal
      std: 0.0388 # base_initialization_std/sqrt(4.25)
      a: -0.0776
      b: 0.0776
  output_layer_initializer:
      mean: 0.0
      name: truncated_normal
      std: 0.007333 # base_initialization_std/sqrt(4.25)/sqrt(2*num_layers)
      a: -0.014668
      b: 0.014668
  output_logits_scale: 0.23529411764705882 # 1/4.25
  embeddings_scale: 10
  scale_qk_dot_by_d: True 
  
optimizer:
  adjust_learning_rate:
      decoder_kernel: 0.23529411764705882 # 1/4.25
```

### Convert a config to a &mu;P config

To convert your GPT-3 configuration file to [&mu;P](https://arxiv.org/abs/2203.03466) compatible configuration, use the [convert_config_to_mup.py](../../../tools/convert_config_to_mup.py) script. Ideally you need to run a hyperparameter sweep on a *proxy-model* to determine optimal values for the tunable parameters such as base learning rate, base standard deviation for the initializer and embeddings multiplier. In the Cerebras-GPT work, we used `sequential` learning rate schedule with `linear` warmup and `linear` or `cosine` decay following the convention in GPT-2/3 models. The conversion script only supports this `sequential` schedule, and to use a different schedule, you'll need to manually set it in the generated &mu;P configuration. 

#### Usage

The configuration conversion script requires an input config file in `yaml` format. There are five other optional arguments:
- `--input_yaml` or `-i`: Input Standard Parameterized (SP) configuration file.

All the remaining parameters are *optional*. They need to be determined by a hyperparameter sweep but if not provided, default to Cerebras-GPT recommendations.
- `--base_layer_width` or `-d_base`: base/proxy-model's width, defaults to 256.
- `--base_lr` or `-lr_base`: base/proxy-model's lr determined by hyperparameter sweep, defaults to 6e-3. Currently, we support config generation for sequential `Linear` and `CosineDecay` learning rate schedules. First lr scheduler should perform `linear` warm-up and second scheduler should perform `linear` or `cosine` decay.
- `--base_init_std` or `-std_base`: base/proxy-model's initial standard deviation,  defaults to 0.08
- `--m_embed` or `-m_base`: base/proxy-model's embeddings multiplier, defaults to 10.0
- `output_yaml` or `-o`: Output `yaml` file to save the &mu;P config.

#### Example of the script usage
Example usage of the script with input configuration file provided and rest of the optional arguments will assume default values:
```bash
python convert_config_to_mup.py --input_yaml </path/to/config>/params_gpt3_2p7b.yaml
muP config saved to </path/to/config>/params_gpt3_2p7b_mup.yaml
```


## Appendix

**Reference**: Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

**Reference**: Brown, T.B. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).

**Reference**: Yang, G. et al. (2022). [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466).

**Reference**: Dey, N. et al. (2023). [Cerebras-GPT: Open Compute-Optimal Language Models
Trained on the Cerebras Wafer-Scale Cluster](https://arxiv.org/abs/2304.03208).


