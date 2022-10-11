# UNet Model (Experimental)

**NOTE:** This is an experimental model and support for this model may be dropped in the future releases.

UNet [1] is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg, Germany. The network is a fully convolutional network. This model currently supports two different datasets: DAGM2007, Severstal Steel Defect Detection.

### How to Run on GPU:

To run the UNet model, use the following command:
```
python run.py --params /path/to/yaml --model_dir /path/to/modeldir --mode train
```

### How to run on the Cerebras System

To run on the Cerebras System, you need to specify the Cerebras System IP address and to execute `run.py` script within Cerebras environment, i.e. within Singularity container with Cerebras client software. The Cerebras System IP address can be typically set either in `params.yaml` or as a CLI argument `--cs_ip x.x.x.x`.

Within Cerebras environment usually the following modes are supported for `run.py` :

* `validate_only`: with this mode, we'll do a quick validation that the model code is compatible with the Cerebras System. Compilation process will go up to kernel matching.  
* `compile_only`: will run end-to-end compilation and generate compiled executable.
* `train`: will compile and train on the Cerebras System.

To train on the Cerebras System, run the same script inside the Cerebras environment and pass a Cerebras System IP address:

```bash
python run.py --mode train --cs_ip x.x.x.x
```

### Structure of the code in this folder

* `configs/` - YAML configuration files
* `data.py` - Entry-point to the data input pipeline code, defines `train_input_fn` and `eval_input_fn`
* `input/` - Input pipeline implementation
* `model.py` - Entry-point to the model, defines `model_fn`
* `UNetModel.py` - Model implementation; bulk of model is defined in here. It inherits from our central BaseModel located in `common/BaseModel.py`. The model also uses Cerebras-defined layers that are located in `common/layers/tf`.
* `run.py` - Train script. Performs training on the Cerebras System/GPU.
* `utils.py` - Misc scripts, including `get_params` to parse the params dictionary from the YAML files. Also the color codes to visualize different segments of the image are defined in this file.

### Datasets: DAGM 2007/Severstal Steel Defect Detection

The UNet model is intended to run on these datasets:

1. [DAGM 2007 competition dataset](https://www.kaggle.com/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)

2. [Severstal Steel Defect Detection dataset](https://www.kaggle.com/c/severstal-steel-defect-detection/overview)

#### DAGM 2007 competition dataset

This is a synthetic dataset for defect detection on textured surfaces (Binary classification, defective vs. not defective). It was originally created for a competition at the 2007 symposium of the DAGM (Deutsche Arbeitsgemeinschaft für Mustererkennung e.V., the German chapter of the International Association for Pattern Recognition). The dataset has 10  different textures types. The texture class is selectable though `configs/params_dagm.yaml` file.

The image below shows the original input image on the left, mask image in the middle and reconstructed mask image at the output of the network on the right for this dataset.

![Sample Image](./images/sample_dagm.png)

Download the dataset from Kaggle website and extract. Update `dataset_path` field in `configs/params_dagm.yaml` to point to the dataset folder. Additionally, you can create a TFRecords dataset from the raw dataset, to speedup the dataloader. To do so, use `inputs/write_dagm_tfrecords.py` script:

```bash
python inputs/write_dagm_tfrecords.py --params configs/params_dagm.yaml --output_directory /path/to/store/dagm/tfrecords/dataset
```
Then, update the `dataset_path` field in `configs/params_dagm_tfrecords.yaml` and use this config file to train UNet.

On this dataset, the UNet model is expected to achieve mIOU of 0.87 and Dice Similarity Coefficient of 0.93 at convergence, with default parameters in YAML file.

### Severstal Steel Defect Detection dataset

This dataset includes real images of steel surface. Each image can have no defects or one or more defects of these types: `[1, 2, 3, 4]`. This model solves a binary classification task, to classify pixels of each image as non-defective/defective of pre-selected class (selected in YAML config file).

The image below shows the original input image on the left, mask image in the middle and reconstructed mask image at the output of the network on the right for this dataset.

![Sample Image](./images/sample_severstal.png)

Download the dataset from Kaggle website and extract. Update `dataset_path` field in `configs/params_severstal.yaml` to point to the dataset folder. Additionally, you can create a TFRecords dataset from the raw dataset, to speedup the dataloader. To do so, use `inputs/write_severstal_tfrecords_binary.py` script:

```bash
python inputs/write_severstal_tfrecords_binary.py --params configs/params_severstal.yaml --output_directory /path/to/store/severstal/tfrecords/dataset
```
Then, update the `dataset_path` field in `configs/params_severstal_tfrecords.yaml` and use this config file to train UNet.

On this dataset, the UNet model is expected to achieve mIOU of 0.71 and Dice Similarity Coefficient of 0.81 at convergence, with default parameters in YAML file.

### YAML File Layout

The bulk of our parametrization can be done through the YAML config files. Each file is nested and has the following general categories of parameters to set:

* `train_input` are the params relevant for the input pipeline. For more information, look at the code in `data.py`.
* `model` params related to model such as filter sizes, skip connections etc.
* `optimizer` Optimizer params such as optimizer type (Adam by default), loss-scale, etc.
* `training` params are relevant for the train script.

### Citations

For further reading, this design has been drawn from a variety of different papers, including:

1. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf), Ronneberger et. al 2015
