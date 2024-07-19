# ESM-2: Evolutionary Scale Modeling for Protein Language

# Overview of the model

Evolutionary Scale Modeling ([ESM-2](https://www.science.org/doi/abs/10.1126/science.ade2574)) is a transformer protein language models from the Meta Fundamental AI Research Protein Team (FAIR). This directory contains implementations of the ESM-2 model.

# Structure of the code
* `configs/`: YAML configuration files.
* `model.py`: Provides a common wrapper for all models under class `Esm2ForPreTrainingModel`, which interfaces with model-specific code. In this repo the model-specific code, i.e., model architecture is in `esm2_pretrain_models.py::Esm2PretrainModel`. This wrapper provides a common interface for handling the function call of the model with its specific data format. It also provides a common interface to use the same format of configuration files from `configs/` to construct various models.
* `data.py`: The entry point to the data input pipeline code.
* `run.py`: Training script. Performs training and validation.
* `utils.py`: Miscellaneous helper functions.

# Dataset Preparation

Follow these steps to prepare the dataset in HDF5 format for training ESM-2 model

* Download the Uniref 50 dataset using this command  
  `curl -O https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz `
* Split the dataset into training and validation splits in the ratio 90:10
* Follow the steps to preprocess the esm2 dataset in our data preprocessing documentation. Refer to example data config file for esm2 dataset [`esm2_preprocessing_pre_training.yaml`] (../../../data_preparation/nlp/hdf5_preprocessing/configs/esm2_preprocessing_pre_training.yaml). The command for creating the ESM2 dataset is given below - 
`python preprocess_data.py --config /path/to/data_config`

#### ESM2 DataProcessor output

The `BertHDF5DataProcessor` class in [`BertHDF5DataProcessor.py`](../../../data/nlp/bert/BertHDF5DataProcessor.py) is used to process the preprocessed esm2 data and feed it to the model. 

# How to run

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](./configs/) directory. Before starting a pre-training run, make sure that in the YAML config file you are using:

-   The `train_input.data_dir` and/or `eval_input.data_dir` parameter points to the correct dataset.

**YAML config files**: Details on the configs for this model can be found in the [Configs included for this model](#configs-included-for-this-model) section.

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

-   `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in the [configs included for this model](#configs-included-for-this-model) section.
-   `/path/to/model_dir` is a path to the directory where you would like to store the logs and other artifacts of the run.
-   `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode, or change to `--mode train_and_eval` to run in train_and_eval mode.

## To compile/validate, run train and eval on Cerebras System
Please follow the instructions on our [quickstart in the Developer Docs](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).

## To run train and eval on GPU/CPU
If running on a CPU or GPU, activate the environment for GPU given in our documentation, and simply run:

```
python run.py {CPU,GPU} --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```

## Configs included for this model
For convenience, we provide the configurations used to train ESM-2.
* [params_esm2_t12_35M_UR50D.yaml](./configs/params_esm2_t12_35M_UR50D.yaml): A 35M parameter ESM-2 model.
* [params_esm2_t30_150M_UR50D.yaml](./configs/params_esm2_t30_150M_UR50D.yaml): A 150M parameter ESM-2 model.
* [params_esm2_t33_650M_UR50D.yaml](./configs/params_esm2_t33_650M_UR50D.yaml): A 650M parameter ESM-2 model.
* [params_esm2_t33_650M_UR50D_vsl.yaml](./configs/params_esm2_t33_650M_UR50D_vsl.yaml): A 650M parameter ESM-2 model with Variable Sequence Length (VSL). VSL allows short sequences to be packed into max sequence length, and makes training more efficient by reducing paddings.
* [params_esm2_t36_3B_UR50D.yaml](./configs/params_esm2_t36_3B_UR50D.yaml): A 3B parameter ESM-2 model.
* [params_esm2_t48_15B_UR50D.yaml](./configs/params_esm2_t48_15B_UR50D.yaml): A 15B parameter ESM-2 model.
