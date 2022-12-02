# Release Notes

The following are the release notes for the Model Zoo repository.

## Version 1.6.1

- First Weight Streaming model support for PyTorch GPT2 XL model as an early access.
- Improvements on Pipeline Legacy flow using Kubernetes and appliance mode.

## Version 1.6.0

### New features and enhancements

#### TensorFlow

- Increased support for GPT3 style models:
  - GPT3 style model with 6.7B parameters.
  - Early limited access for GPT3 style 13B parameter model.
  - Early limited access for GPT3 style 20B parameter model, inspired from GPT-NeoX architecture.

- Support for Appliance Mode run scripts to run models on Cerebras Wafer-Scale Cluster in Weight Streaming.

#### PyTorch

- PyTorch Layer API support for following layers:
  - AttentionLayer
  - EmbeddingLayer
  - FeedForwardNetwork
  - RelativePositionEmbeddingLayer
  - TransformerDecoderLayer
  - TransformerDecoder
  - TransformerEncoderLayer
  - TransformerEncoder

- Transformer style demo model using the Layer API.
- Migrated GPT2 model implementation to use PyTorch Layer API from the HuggingFace based implementation.
  - HuggingFace based implementation for these models is deprecated.

- Support for PyTorch Optimizers:
  - Adafactor, Adam (including AdamW), Lamb, RMSprop, SGD.
  - Experimental: RAdam, Rprop, ASGD, NAdam, Adadelta, Adagrad, Adamax.

#### Usability

- Support for Pipeline models in Kubernetes(k8s) workflow for running on Cerebras Wafer-Scale Clusters.


## Version 1.5.0

Documentation enhancement for all the supported models.

Documentation enhancement for dataloader in PyTorch and Tensorflow framework.

Dropped support for:

- LSTM models
- GNN models
- CTR model
- Audio model
- RevBERT

### Known issues

#### Checkpoint save issue in GPT2 large model

The GPT2 [Large](transformers/tf/gpt2/configs/params_gpt2_large.yaml) config or any model with checkpoint size larger than 9024 MB runs into the following error when running on CS2:

```bash
2022-06-27 03:12:27,424 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local:            COUT030: Default Fatal message.
2022-06-27 03:12:27,426 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local:            COUT030: CEREBRAS_THROW_UNLESS in file: /spare/jenkins/workspace/release/cbcore-rel-1.4.0/src/workflow_common/compile_context.cc line: 417 function: uint64_t cerebras::workflow::CompileContext::hmc_checkpoint_size_mb() const
2022-06-27 03:12:27,428 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local: This condition is false: skip_ckpt_size_check || hmc_checkpoint_size_mb <= 9024
2022-06-27 03:12:27,429 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local: Calculated checkpoint size is 11808MB, but it should not exceed 9024MB
2022-06-27 03:12:27,431 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local:            COUT030: Please contact Cerebras Systems Support.
```

A workaround for this is setting one environment variable, and run the Tensorflow model as:

```bash
export SKIP_CKPT_SIZE_CHECK=1; csrun_wse python run.py --mode train \
    --params configs/<name-of-the-params-file.yaml>
```

## Version 1.4.0

### New features and enhancements

#### Multireplica

The Pytorch [BERT](transformers/pytorch/bert), [Transformer and T5](transformers/pytorch/t5) models can now be run in multireplica mode using the `--multireplica` flag, depending on the size of the model.

Multireplica is not supported for [GPT2](transformers/pytorch/gpt2/). In case a user runs GPT2 with `--multireplica` option to run.py, user will be running with config that has not been qualified.

### Known issues

#### Checkpoint save issue in GPT2 large model

The GPT2 [Large](transformers/tf/gpt2/configs/params_gpt2_large.yaml) config or any model with checkpoint size larger than 9024 MB runs into the following error when running on CS2:

```bash
2022-06-27 03:12:27,424 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local:            COUT030: Default Fatal message.
2022-06-27 03:12:27,426 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local:            COUT030: CEREBRAS_THROW_UNLESS in file: /spare/jenkins/workspace/release/cbcore-rel-1.4.0/src/workflow_common/compile_context.cc line: 417 function: uint64_t cerebras::workflow::CompileContext::hmc_checkpoint_size_mb() const
2022-06-27 03:12:27,428 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local: This condition is false: skip_ckpt_size_check || hmc_checkpoint_size_mb <= 9024
2022-06-27 03:12:27,429 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local: Calculated checkpoint size is 11808MB, but it should not exceed 9024MB
2022-06-27 03:12:27,431 INFO     [hosts.py:413] lab@sc-r9ra3-s1.cerebrassc.local:            COUT030: Please contact Cerebras Systems Support.
```

A workaround for this is setting one environment variable, and run the Tensorflow model as:

```bash
export SKIP_CKPT_SIZE_CHECK=1 csrun_wse python run.py --mode train \
    --params configs/<name-of-the-params-file.yaml>
```

NOTE: Use `python-pt` for PyTorch models.

## Version 1.3.0

### New features and enhancements

#### PyTorch BERT Model with VTS

[Configs](transformers/pytorch/bert/configs) have been added for running the PyTorch BERT model with Variable Tensor Shape (VTS).

#### PyTorch BERT Classifier Model

The PyTorch BERT [Classifier](transformers/pytorch/bert/fine_tuning/classifier) fine-tuning model has been added, and is supported on the Cerebras System in `train` and `eval` modes.

#### PyTorch BERT Extractive Summarization Model

The PyTorch BERT [Extractive Summarization](transformers/pytorch/bert/fine_tuning/extractive_summarization) fine-tuning model has been added, and is supported on the Cerebras System in `train` and `eval` modes.

#### PyTorch BERT Question Answering (SQuAD) Model

The PyTorch BERT [Question Answering](transformers/pytorch/bert/fine_tuning/qa) fine-tuning model has been added, and is supported on the Cerebras System in `train` and `eval` modes. This model can be used to for the SQuAD dataset, among others.

#### PyTorch T5 and Transformer Models with VTS

[Configs](transformers/pytorch/t5/configs) have been added for running the PyTorch T5 and Transformer models with Variable Tensor Shape (VTS).

#### TensorFlow GPT-J Model

The TensorFlow [GPT-J](transformers/tf/gptj) model has been added, and can be run in weight-streaming mode. In addition, the [abstractive summarization](transformers/tf/gptj/fine_tuning/abstractive_summarization) fine-tuning model for GPT-J is available.

#### Multireplica

The TensorFlow [BERT](transformers/tf/bert) and [Transformer](transformers/tf/transformer) models can now be run in multireplica mode using the `--multireplica` flag.

### Known issues

#### PubMed BERT with multireplica

The PubMed BERT Base MSL128 [config](transformers/tf/bert/configs/params_pubmedbert_base_msl128.yaml) hits a compile failure when using the `--multireplica` flag.

## Version 1.2.0

### New features and enhancements

#### PyTorch T5 Model

The PyTorch [T5](transformers/pytorch/t5) has been added and is supported on the Cerebras System in `train` and `eval` modes.

#### PyTorch Transformer Model

The [PyTorch Transformer-Attention is All You Need](transformers/pytorch/t5) model is now fully supported on the Cerebras System in both `train` and `eval` modes. It shares the `transformers/pytorch/t5` folder with the T5 model, since both models rely on the same underlying HuggingFace implementation.

#### PyTorch GPT-2 Model

The PyTorch [GPT-2](transformers/pytorch/gpt2) has been added and is supported on the Cerebras System in `train` and `eval` modes.

#### PyTorch BERT Token Classifier (NER) Model

The PyTorch BERT [token classifier](transformers/pytorch/bert/fine_tuning/token_classifier) fine-tuning model for named entity recognition (NER) has been added, and is supported on the Cerebras System in `train` and `eval` modes. 

#### TF T5 and Transformer models

The TF T5 and Transformer models are out of beta and will converge.

#### TF Linformer Model

The TF [Linformer](transformers/tf/linformer) model has been added and is supported on the Cerebras System in `train` and `eval` modes.

## Version 1.1.0

### New features and enhancements

#### PyTorch

- The PyTorch support is enhanced. Key changes include but not limited to:

  - Support for `eval` mode is added. Now both `train` and `eval` modes are supported.
  - Simplified `cbtorch-session`.
  - Enhanced the flexibility in specifying the `cerebras.framework.torch.initialize()`.
  - Use of `cbfloat16` data format is now supported. See [CB16 half-precision](https://docs.cerebras.net/en/latest/performance-tuning/cs-1-data-formats.html#cb16-half-precision).
  - Made mixed precision interface more intuitive, via `GradScaler`. See [PyTorch Dynamic Loss Scaling](https://docs.cerebras.net/en/latest/performance-tuning/dynamic-loss-scaling.html).
  - Fixed several bugs in the areas of numerics, convergence and performance.

- Supported PyTorch ops

A preliminary list of supported PyTorch ops is released. See [Supported PyTorch Ops](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-ops.html).

#### Multi-replica data parallel training

A new feature called **multi-replica data parallel training** is released. Currently this feature is available only for TensorFlow models. When you use this feature, the Cerebras compiler uses several copies (replicas) of the same model to run data parallel training. See [Multiple Models](https://docs.cerebras.net/en/latest/tensorflow-docs/multiple-models/index.html) for detailed documentation.

For a list of TensorFlow models supporting the multi-replica data parallel training, see [Supported multi-replica models](tensorflow-docs/multiple-models/multi-replica-data-parallel-training.html#supported-models). This feature is not yet supported for PyTorch models.

#### PyTorch BERT Model

The [PyTorch BERT pretraining model](transformers/pytorch/bert) is no longer in BETA. In addition:

- Support for `eval` mode is added.
- RoBERTa (Next Sentence Prediction (NSP) only) configurations are supported. See [roberta_base.yaml and roberta_large.yaml](transformers/pytorch/bert/configs/).
- Longer Maximum Sequence Length (MSL) configurations are supported, at least up to MSL 4096.

#### PyTorch FC MNIST Model

The [PyTorch FC MNIST model](fc_mnist/pytorch) is no longer in Beta. The model can now be trained and evaluated on the Cerebras System.

#### PyTorch Transformer Model (BETA)

The [PyTorch Transformer-Attention is All You Need](transformers/pytorch/transformer) model is added as a Beta feature. This model can be compiled using `run.py` with the `--compile_only` flag, as well as run on CPU or GPU using `run_cpu_gpu.py`. To train this model on the Cerebras System at your own risk, comment out the following lines from `run.py`:

```python
    if not runconfig_params["compile_only"]:
        raise ValueError(
            "Running the Transformer model on the Cerebras System is in beta."
            "Convergence is not guaranteed. Remove this exception to proceed."
        )
```

### Known issues

#### T5 and Transformer (Attention is All You Need)

- The TensorFlow versions of the [T5](transformers/tf/t5) and [Transformer](transformers/tf/transformer) models are not guaranteed to converge. These models can still be compiled to the Cerebras system. However, to train these models on the Cerebras System at your own risk, comment out the following lines from `run.py` corresponding to the model:

```python
    if not runconfig_params["compile_only"]:
        raise ValueError(
            "Running the Transformer model on the Cerebras System is in beta."
            "Convergence is not guaranteed. Remove this exception to proceed."
        )
```

#### PyTorch

- For PyTorch, when you are targeting GPU, the following warning will be displayed. This can be safely ignored. This issue does not exist when you target Cerebras system for your acceleration.

    ```text

    UserWarning: Detected call of ``lr_scheduler.step()`` before
    ``optimizer.step()``. In PyTorch 1.1.0 and later, you should
    call them in the opposite order: ``optimizer.step()`` before
    ``lr_scheduler.step()``.  Failure to do this will result in
    PyTorch skipping the first value of the learning rate schedule.
    ```

- For PyTorch models only, to run the training on the Cerebras system, the `cs_ip` flag must include both the IP address and the port number of the CS system. Only the IP address, for example: `--cs_ip 192.168.1.1`, will not be sufficient. You must also include the port number, for example: `--cs_ip 192.168.1.1:9000`.

### Multi-replica data parallel training

- Dynamic loss scaling is not yet supported with [Multi-replica Data Parallel Training](https://docs.cerebras.net/en/latest/tensorflow-docs/multiple-models/multi-replica-data-parallel-training.html#multi-replica-data-parallel-training).

- Eval on Cerebras system is not yet supported for multi-replica data parallel trained models. You can run eval on CPU or GPU for these models.

## Version 1.0.0

### New features and enhancements

#### PyTorch (BETA)

Support is added, in beta phase only, for the PyTorch framework. The models and quickstart provided in this repo are strictly intended as advanced information only.

- A [PyTorch version of FC-MNIST](fc_mnist/pytorch) is added as a part of PyTorch (BETA) support. This version only supports compiling on a CPU node with the `train` mode. To train this model on the Cerebras System at your own risk, edit the `run.py` file and comment out the entire `raise ValueError()` function, as shown below:

    ```python
    elif runconfig_params["mode"] == TRAIN:
            # raise ValueError(
            #    "Training PyTorch models on the Cerebras System is in beta "
            #    "and is only validated with the default config provided in the "
            #    "Model Zoo. Remove this exception and use the provided config to"
            #    "proceed."
            #)
            runner.train(train_loader)
    ```

- The [PyTorch versions of BERT Base and BERT Large](transformers/pytorch/bert) are added as a part of PyTorch (BETA) support. These versions only support compiling on a CPU node with the `train` mode. To train these models on the Cerebras System at your own risk, edit the `run.py` file and comment out the entire `raise ValueError()` function, as shown below:

    ```python
    elif runconfig_params["mode"] == TRAIN:
            #raise ValueError(
            #"Training PyTorch models on the Cerebras System is in beta "
            #"and is only validated with the default configs provided in the "
            #"Model Zoo. Remove this exception and use one of the provided "
            #"configs to proceed."
            #)
            runner.train(train_loader)
    ```

#### RevBERT

A new model, the [RevBERT](transformers/tf/rev_bert), is introduced. The [RevBERT](transformers/tf/rev_bert) is a Cerebras-specific BERT model that improves the BERT performance on Cerebras accelerator. Using the RevBERT model you can run up to 20x larger batch sizes and 2.7x larger models on the Cerebras System. This version of RevBERT is only supported with TensorFlow and only supports the `train` mode.

#### Transformer (Attention Is All You Need)

Support is added in the `train` mode for Variable Sequence Length (VSL) on the CS system.

#### T5 model

- Support is enhanced from loss-only eval to full eval metrics.
- Support is added in the `train` mode for Variable Sequence Length (VSL) on the CS system.

#### GPT-2

Support is added in the `train` mode for Variable Sequence Length (VSL) on the CS system.

**NOTE**: See [Model Support Matrix](https://docs.cerebras.net/en/latest/modelzoo-intro/model-support-matrix.html) for a full list of supported models.

## Version 0.9.0

### New features and enhancements

#### Transformer (Attention Is All You Need)

Support is added for the [Transformer (Attention Is All You Need)](https://arxiv.org/abs/1706.03762), with the following capabilities:

- On CS system: Training, and Eval (loss only).
- On GPU: Train, Eval (`eval` and `eval_all`).

#### T5 model

Support is added for the following [T5](https://arxiv.org/abs/1910.10683) family of models:

- Small model:
  - d<sub>model</sub> = 512
  - d<sub>ff</sub> = 2,048.
  - 8-headed attention.
  - 6 layers each in the encoder and decoder.
  - About 60 million parameters.

- Model:
  - Base, BERT Base-sized encoder and decoder.
  - About ~ 220 million parameters.

- Model: Large, BERT Large-sized encoder and decoder.
  - d<sub>model</sub> = 1,024.
  - d<sub>ff</sub>  = 4,096.
  - d<sub>kv</sub>  = 64.
  - 16-headed attention.
  - 24 layers each in the encoder and decoder.
  - Around 770 million parameters.

- Dataset supported: [Colossal Clean Crawled Corpus (C4) dataset](https://github.com/allenai/allennlp/discussions/5056).

- On CS system: Pre-training, Eval (loss only).
- On GPU: Train, Eval (`eval` and `eval_all`).

#### Variable Sequence Length

The variable sequence length (VSL) performance of BERT-style encoder-decoder models is enhanced. Previously, a sequence of less than pre-defined maximum sequence length is padded up to the maximum sequence length. The compute and memory are also spent on processing these tokens used for padding, resulting in a significant loss of performance.

With this enhancement, by taking advantage of the sparsity the tokens used for padding are not processed, thereby enhancing the performance of the variable length sequences.

#### VSL-enhanced models

The performance-optimized variable sequence length is now available for the following models on the CS system:

- BERT Pre-training (training only).
- RNN Language Model (LM) (training only).
- RNN Sentiment (training only).

#### Enhanced BERT- and GPT-style models

Performance is enhanced for long sequences (MSL up to 8K for smaller models) for BERT- and GPT-style models. This is accomplished by making use of sparse attention to reduce memory requirements.

**See also**: For a full list of supported models, see [Model Support Matrix](https://docs.cerebras.net/en/latest/cs-1-tf-user-guide/model-support-matrix.html).

### Known issues

- When you use [AdamW Optimizer](https://github.com/Cerebras/modelzoo/blob/master/common/optimizers/AdamWOptimizer.py) from Model Zoo and if both the following conditions are true:

  - The parameter `weight_decay` is set to a non-zero value, and
  - The parameter `loss_scaling_factor` is not set to "dynamic".

then the execution will stop with the following error message:

```bash
"When using the AdamW optimizer with weight decay, set the loss_scaling_factor to dynamic."
```

- For the models T5 and Transformer (Attention Is All You Need), the performance in samples-per-sec is optimal when the source `max_seq_len` and the target `max_seq_len` are equal.

- When running evaluation with a BERT model, if the `max_predictions_per_seq` parameter is set to an odd value and if the following conditions are true:

  - The tensor is multi-dimensional (>1D).
  - The inner dimension is an odd value.
  - The datatype is < 4 bytes, i.e., FP16 or INT16 or UINT16.

    then this leads to a compile failure in 0.9.0 and execution failure in 0.8.0.

    **Workaround**: Set the  `max_predictions_per_seq` parameter to an even value.

## Release 0.8.0

### New models and enhancements

#### Inference support

- Inference is now supported for the following models:

  - [Graph Convolutional Network](https://github.com/Cerebras/modelzoo/tree/master/graphs/tf).
  - [Graph Attention Network](https://github.com/Cerebras/modelzoo/tree/master/graphs/tf#graph-attention-network-gat).

Also see [Model Support Matrix](https://docs.cerebras.net/en/latest/cs-1-tf-user-guide/model-support-matrix.html).

#### CS_AUTOTUNE

- The `CS_AUTOTUNE` (Cerebras AUTOTUNE) is similar to the TensorFlow <a href="https://www.tensorflow.org/guide/data_performance" class="external-link">tf.data.AUTOTUNE</a>. These two AUTOTUNE constants have the same value. Wherever the standard `tf.data.AUTOTUNE` or `tf.data.experimental.AUTOTUNE` is used, the `CS_AUTOTUNE` will be used automatically in its place. When targeting the Cerebras System wafer scale engine, using `CS_AUTOTUNE` will result in a better specification of parameters such as:

- `num_parallel_calls`
- `cycle_length`
- `num_parallel_reads`

#### Summary ops

- Use summary ops to write the summary data as you would in TensorFlow, by using the `tf.summary` module in your model function. To enable the summaries, you must wrap your Estimator call with the following:

```python
from cerebras.tf.summary import cs1_enable_summaries
with cs1_enable_summaries():
   # estimator code
```

See an example [here](https://github.com/Cerebras/modelzoo-internal/blob/5bbeeb517a99a3df0609ca2cf72b0c1513efb632/fc_mnist/tf/run.py#L243). Also see <a href="https://docs.cerebras.net/en/latest/cs-1-tf-user-guide/tuning-tf-for-cs-1/using-tensorboard.html#using-summary-ops" class="external-link">Writing summary data</a>.

### Known issues

#### TensorFlow Datasets

- [FC MNIST](fc_mnist/tf), [RNN Sentiment](rnn_encoder/sentiment/tf), and [BERT Classifier](transformers/bert/tf/fine_tuning/classifier) models rely on `tensorflow-datasets` (`tfds`) to load datasets.
- Automated download of `tfds` datasets will not work when running the training on a CPU. For running the training on a CPU, you have to first download the required `tfds` dataset before you start the training. To download the required `tfds` dataset, run with `--validate_only` or `--compile_only` flag. After the `tfds` dataset is downloaded you can run the training on a CPU.
- Training on the Cerebras System and offline compilation for the Cerebras System is not affected by this issue.

## Release 0.7.1

### New models and enhancements

#### Support for BERT evaluation and prediction

- Evaluation and prediction are now supported for BERT networks. While executing the `run.py`, you can run evaluation or prediction with your network as follows:

  - **Evaluation**: The following two modes are supported with the ``--mode`` option:

    - ``eval``
    - ``eval_all`` (runs locally on either CPU or GPU, not on the Cerebras System)

    - **Prediction**: Use ``--mode predict`` to use the prediction feature.

    See the following for additional documentation:

    - <a href="https://docs.cerebras.net/en/latest/cs-1-tf-user-guide/running-a-model/anatomy-of-run-py.html#run-py-example-template" class="external-link">Run.py example template</a> for a description of the ``mode`` parameter using an example BERT ``run.py`` script.

    - <a href="https://docs.cerebras.net/en/latest/cs-1-tf-user-guide/running-a-model/train-eval-predict.html" class="external-link">Train, Eval and Predict</a> for usage examples, and

    - <a href="https://docs.cerebras.net/en/latest/getting-started/cs-1-execution-flow.html#cs-1-inference-exec-flow" class="external-link">The the Cerebras System flow for prediction</a> for a sequence diagram on how the Cerebras System execution flow works for prediction.

## Release 0.7.0

These release notes describe the changes from the Version 0.6.3 to make the
repository compatible with the
<a href="https://docs.cerebras.net/en/latest/release-notes/rel-notes-cumulative.html#version-0-7-0" class="external-link">Version 0.7.0</a>
of Cerebras Graph Compiler (CGC) software.

### New models and enhancements

- The following new models are released:
  - <a href="https://github.com/Cerebras/modelzoo/tree/master/transformers/gpt2/tf" class="external-link">GPT-2</a>,
        an example of auto-regressive transformer model. With the
        addition of GPT-2 into a family of the Model Zoo models, we have
        also refactored the BERT code. Now, both the GPT and BERT models
        are located in the
        <a href="https://github.com/Cerebras/modelzoo/tree/master/transformers" class="external-link">transformers</a>
        folder and share some common code.

  - <a href="https://github.com/Cerebras/modelzoo/tree/master/graphs/tf#general-architecture" class="external-link">Graph Attention Network (GAT)</a>.

- <a href="https://github.com/Cerebras/modelzoo/blob/master/graphs/tf/run.py#L230" class="external-link">Eval</a>
    and
    <a href="https://github.com/Cerebras/modelzoo/blob/master/graphs/tf/run.py#L233" class="external-link">inference</a>
    on the Cerebras System for
    <a href="https://github.com/Cerebras/modelzoo/tree/master/graphs/tf" class="external-link">GCN</a>.

- <a href="https://github.com/Cerebras/modelzoo/blob/master/modelzoo/common/tf/model_utils/cs_loss_process.py#L25" class="external-link">Support</a>
    for `cbfloat` datatype for BERT and UNet. See
    <a href="https://docs.cerebras.net/en/latest/general-guides/cs-1-data-formats.html#cb16-half-precision" class="external-link">here</a>
    for the documentation for `cbfloat`.

### Breaking changes

- The `use_cs` parameter in the
    <a href="https://github.com/Cerebras/modelzoo/blob/master/modelzoo/common/tf/estimator/cs_estimator.py#L26" class="external-link">CerebrasEstimator API</a>
    is removed and will result in compiler error if used in this API.
    The target hardware will now be automatically determined from a
    combination of the runtime configuration parameter `cs_ip` and
    the` use_cs` parameter setting in the method definitions for
    `train`, `evaluate` and `predict`. See
    <a href="https://docs.cerebras.net/en/latest/cs-1-tf-user-guide/adapting-to-cs-1/cerebrasestimator-interface.html" class="external-link">here</a>
    for the documentation.

- The format of the YAML config files for all the Model Zoo models is
    slightly changed as follows:

  - All training-related parameters have been moved to the `runconfig` section.

  - The `max_steps` parameter is added as a default parameter to control the duration of training.

### Known issues

#### TensorFlow GLUE dataset

- Due to broken download links, the TensorFlow Glue datasets must be
    manually downloaded. The standard Cerebras wrapper for TFDS Glue
    datasets,
    <a href="https://github.com/Cerebras/modelzoo/blob/master/transformers/bert/tf/fine_tuning/classifier/input/TfdsTextDataProcessor.py" class="external-link">TfdsTextDataProcessor</a>,
    will not attempt to download the Glue datasets SST-2 and MNLI.

  - **Workaround**: Use the script
        <a href="https://github.com/Cerebras/modelzoo/blob/master/transformers/bert/tf/fine_tuning/classifier/input/download_glue_data.py" class="external-link">download_glue_data.py</a>
        to download the Glue datasets SST-2 and MNLI. See this
        <a href="https://github.com/Cerebras/modelzoo/blob/master/transformers/bert/tf/fine_tuning/classifier/input/README.md%20" class="external-link">README</a>.

  - **Important**: If you are running your code outside of the
        `cbcore` then make sure that you install
        `tensorflow-datasets==1.0.2` before proceeding.

#### CTR dataset

- At the time of this release, the dataset used in CTR model
    (<a href="http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/" class="external-link">Criteo dataset for the Kaggle Display Advertising Challenge</a>)
    is no longer available for download from the Criteo Labs web site.
