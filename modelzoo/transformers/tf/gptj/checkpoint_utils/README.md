# Checkpoint Utilities

The `convert_hf_checkpoint_to_cerebras.py` file is used to map the HuggingFace GPT-J 6B weights to a Tensorflow checkpoint which can be loaded for either continuous pre-training or fine-tuning with the Cerebras implementation of GPT-J.

## Installation

The following pre-requisites are needed to enable a clean run of the script. We recommend installing an [Anaconda](https://www.anaconda.com/distribution/#download-section) environment and running the following commands to get started:

```bash
conda create --name <env> python==3.7 pip
conda activate <env>
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**NOTE:** In this setup we install PyTorch and Tensorflow in the same environment. Generally it is not recommended to keep both the same so use this env for checkpoint conversion only. Moreover, we provide CPU versions of both packages as we do not need GPU for this task.

## Running the conversion

To convert the checkpoints, activate the created conda environment and run the following command:

```bash
python convert_hf_checkpoint_to_cerebras.py --input_dir </path/to/hf/checkpoint> --ouput_dir </path/to/store/tf/checkpoint> --share_embeddings
```

There are two required arguments for this file:

- The `--input_dir` specifies the directory where the HuggingFace checkpoint is stored. If this folder does not exist, it is created during runtime. If the folder exists but does not have a `pytorch_model.bin` file in it, it means that the checkpoint is not available locally, and it will be downloaded from HuggingFace Hub.
- The `--output_dir` specifies the directory where the converted checkpoints will be stored. If this folder does not exist, it is created during runtime.

There are two optional arguments:

- The `--share_embeddings` argument, if supplied replicates the original JAX model code by sharing embedding weights with the classifier. Without this, the script creates a checkpoint without shared embeddings, the standard implementation that is widely popular.
- The `--debug` argument, which is set to False by default. This helps debug the GPT-J model created from HuggingFace by printing the number of parameters, to verify that you have the right configuration passed in.

__User-Note__: The pre-provided script will download the checkpoint from HuggingFace if it does not exist in the provided `input_dir`. Since the model is very large, it has a checkpoint around 23GB in size. A single checkpoint conversion will take about 110GB RAM and about 10-15mins to run end to end. Please use a system or server with sufficient compute and memory (storage).

## Verifying the conversion

To verify the conversion, activate the created conda environment and run the following command:

```bash
python verify_checkpoint_conversion.py --input_dir </path/to/hf/checkpoint> --ouput_dir </path/to/store/tf/checkpoint> --share_embeddings
```

All the arguments passed to this script are the same as the script above. The key difference here is that both the `input_dir`, the `output_dir`, and the corresponding HuggingFace and Cerebras checkpoints should be available during runtime, else the script throws an error and exits.