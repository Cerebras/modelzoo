# List of topics
- [List of topics](#list-of-topics)
- [Overview of the demo](#overview-of-the-demo)
- [Dummy dataset](#dummy-dataset)
- [Steps to port the model](#steps-to-port-the-model)
- [Convert model with layers API](#convert-model-with-layers-api)
- [Running the model](#running-the-model)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)

# Overview of the demo

We will demo how to convert your PyTorch model implemented with modules like torch and torch.nn to a model that is ready to compile and run on CS Systems. We utilize our new layers API
 for transformers to make the conversion. Our layers API is created to minic the PyTorch implementation of transformer components. It currently contains modules that resemble torch.nn modules with customized implementation for Cerebras Architecture.

For more information of layers API please visit our [developer docs page](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-ops/index.html).

This demo includes but not limited to the following torch.nn modules:

1. [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
2. [torch.nn.TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
3. [torch.nn.TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
4. [torch.nn.TransformerDecoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html)
5. [torch.nn.TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html)
6. [torch.nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

The original python version of the model is pulled and modified from a [PyTorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) on how to build transformer models with their API.

# Dummy dataset

For simplicity of the demo, we first create a dummy dataset with a small vocabulary set of 26, each representing an English letter. The task of the model will be simple, to produce the English letters in the right order. Dataset (`AlphabetDataset`) definition and train/test input dataloader are located in file [data.py](./data.py). We randomly create sequences of a certain length to be used to train the model on autoregressive generation. For example, an example pair of input and target (labels) is `input: [a, b, c, d] -> target: [b, c, d, e]`.

# Steps to port the model

The model code from the PyTorch tutorial is put under file [pytorch_transformer.py](./pytorch_transformer.py). From this, only a few minor modifications need to be made as listed below:
1. Set `batch_first` argument to be `True`. Currently we only support tensors with `batch_size` as the first dimension.
2. Modified `PositionalEncoding` class to match `batch_first` by changing
```
self.register_buffer('pe', pe)
```
to
```
self.register_buffer('pe', pe.transpose(1, 0))
```
and
```
x = x + self.pe[:x.size(0)]
```
to
```
x = x + self.pe[:x.size(1)]
```
3. Modify function `generate_square_subsequent_mask` such that it creates the tensor on the device with `torch.float16` as data type and uses `-1.0e4` as the negative constant instead of `-inf`.

Now we set up the files `run.py`, `model.py` and `params.yaml` as described in the [workflow documentation](https://docs.cerebras.net/en/latest/pytorch-docs/adapting-pytorch-to-cs.html#porting-pytorch-to-cs).
Under `model.py`, we make modifications to the loss function and mask generator:
1. Instead of using the default `torch.nn.CrossEntropyLoss`, we use our customized `GPTLMHeadModelLoss` from [model_utils](./../../../common/pytorch/model_utils/GPTLMHeadModelLoss.py).
2. In `__call__` function of `model.py`, we calculate the loss by calling `self.loss_fn` defined by `GPTLMHeadModelLoss`.

We could then verify that the model setup is correct by running on a cpu with command 
```
python run.py --mode train --params configs/params.yaml
```

# Convert model with layers API

To support large models on our CS system, we developed [layers API](../../../common/pytorch/layers/) as a replacement of the PyTorch APIs. 
We currently support:
1. [MultiheadAttention](../../../common/pytorch/layers/AttentionLayer.py)
2. [TransformerEncoderLayer](../../../common/pytorch/layers/TransformerEncoderLayer.py)
3. [TransformerEncoder](../../../common/pytorch/layers/TransformerEncoder.py)
4. [TransformerDecoderLayer](../../../common/pytorch/layers/TransformerDecoderLayer.py)
5. [TransformerDecoder](../../../common/pytorch/layers/TransformerDecoder.py)
6. [Transformer](../../../common/pytorch/layers/Transformer.py)
7. [optimizers](../../../common/pytorch/optim)

We also provide helper functions to [create masks](../transformer_utils.py) or [initializers](../../../common/pytorch/model_utils/).

To convert this PyTorch model to a model that is ready to compile and train on the CS system, we can harness the `TransformerEncoderLayer`, `TransformerEncoder` and the `EmbeddingLayer` APIs:

1. Change the `import` paths as shown below:

From
```
from torch.nn import (
    Embedding,
    TransformerEncoder,
    TransformerEncoderLayer,
)
``` 
to
```
from modelzoo.common.pytorch.layers import (
    EmbeddingLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
```

2. Replace Embedding layer and PositionalEncoding with the `EmbeddingLayer` imported from layers API:

In PyTorch, we need an `Embedding` module and a `PositionalEncoding` module for the input embeddings. With the layer API, it's done by a single EmbeddingLayer. It supports word embedding, different kinds of position embeddings, segment embedding and various initializers.

By specifying `position_embedding_type="fixed"` we tell the embedding layer to activate word embeddings with fixed (sinusoidal) position embeddings.

From
```
self.encoder = Embedding(ntoken, d_model)
self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
```
to
```
self.encoder = EmbeddingLayer(
    vocab_size=ntoken,
    embedding_size=d_model,
    position_embedding_type="fixed",
    max_position_embeddings=max_len,
)
```

3. Change the `forward()` function:

Remove 
```
src = self.pos_encoder(src)
```
from the `forward()` function. 

A completed version of the modification is put under [cb_transformer.py](./cb_transformer.py), which will be used to replace [pytorch_transformer.py](./pytorch_transformer.py).

4. Modify `model.py` to import from `cb_transformer.py`
```
from modelzoo.transformers.pytorch.layers_api_demo.cb_transformer import (
    TransformerModel,
)
```

# Running the model 

After we have finished converting the model, it is ready to be compiled or run on a CS system.

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our Developer Docs at:
https://docs.cerebras.net/en/latest/getting-started/pytorch/index.html

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../PYTHON-SETUP.md), and simply run:
