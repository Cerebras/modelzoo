# List of topics

- [Overview of the FC-MNIST model](#overview-of-the-fc-mnist-model)
- [Sequence of the steps to perform](#sequence-of-the-steps-to-perform)
- [Key features from CSoft platform used in this reference implementation](#key-features-from-csoft-platform-used-in-this-reference-implementation)
  - [Multi-Replica data parallel training](#multi-replica-data-parallel-training)
- [Structure of the code](#structure-of-the-code)
- [Dataset and input pipeline](#dataset-and-input-pipeline)
- [How to run:](#how-to-run)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
- [Configurations included for this model](#configurations-included-for-this-model)
- [References](#references)

# Overview of the FC-MNIST model

A simple multi-layer perceptron model composed of fully-connected layers
for performing handwriting recognition on the MNIST dataset.
The model is a `3`-layer multi-layer perceptron. The first layer has hidden
size `500`, the second `300`, and the third layer has `num_classes` number of
hidden units (which here is `10`). It then trains on a categorical cross entropy
loss. This structure is based on the survey of different structures on the
MNIST website [ref #2](#references).

# Sequence of the steps to perform
See the following diagram:

![Diagram](../images/torch_fcmnist.png)

# Key features from CSoft platform used in this reference implementation
FC MNIST model configs are supported in the [Layer Pipelined mode](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html#layer-pipelined-mode).

## Multi-Replica data parallel training
When training on the Cerebras System, the `--multireplica` flag can be used to perform data-parallel training
across multiple copies of the model at the same time. For more details about this feature, please refer
to [Multi-Replica Data Parallel Training](https://docs.cerebras.net/en/private/general/multi-replica-data-parallel-training.html) documentation page.

# Structure of the code
* `data.py`: Simple data input pipeline loading the [TorchVision MNIST dataset](https://pytorch.org/vision/stable/datasets.html).
* `model.py`: Model implementation. 
* `configs/params.yaml`: Example of a YAML configurations file.
* `run.py`: Train script, performs training and validation.
* `utils.py`: Miscellaneous helper functions.

# Dataset and input pipeline

The MNIST dataset comes from `torchvision.datasets`. The train dataset
has a size of `60,000` and the eval dataset `10,000` images.
More information can be found on the
[PyTorch website](https://pytorch.org/vision/0.8/datasets.html#mnist).
Each sample in the dataset is a black and white image of size `28x28`, where
each pixel is an integer from `0 to 255` inclusive.

The first time that the input function is run, it will take some time
to download the entire dataset.
The dataset is to downloaded to the `data_dir` provided in [`configs/params.yaml`](./configs/params.yaml).

The input pipeline does minimal processing on this dataset. The dataset returns one batch at a time, of the form:
```
inputs = (
    features = Tensor(size=(batch_size, 28*28), dtype=torch.floatX,
    labels = Tensor(size=(batch_size,), dtype=torch.int32,
)
```
Where here, `torch.floatX = torch.float32` if we are running in full precision and float16 if we are running in mixed precision mode. You can simply set the mixed precision mode by passing `model.mixed_precision False` as part of the arguments.

# How to run:

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our Developer Docs at:
https://docs.cerebras.net/en/latest/getting-started/cs-pytorch-qs.html

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../PYTHON-SETUP.md), and simply run:

```bash
python run.py --mode train --params configs/params.yaml
```

If run outside of the Cerebras environment with `--mode train`, it will skip validation and compilation steps and proceed straight to the training on your allocated hardware.


# Configurations included for this model
In the [configs](./configs/) directory we have config files to train FC-MNIST model.
    * [params.yaml](./configs/params.yaml) with `depth=10`, `hidden_size=50`, and `SGD` optimizer.

# References

1. [Original MLP MNIST paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
2. [MNIST website with wide survey of different parameters](
    http://yann.lecun.com/exdb/mnist/)
