# Linformer model

- [Linformer model](#linformer-model)
  - [Overview of the model](#overview-of-the-model)
  - [Sequence of steps to perform](#sequence-of-steps-to-perform)
  - [Key features from CSoft platform used in this reference implementation](#key-features-from-csoft-platform-used-in-this-reference-implementation)
  - [Structure of the code](#structure-of-the-code)
  - [Dataset generation](#dataset-generation)
    - [Prerequisites](#prerequisites)
    - [Download and preprocess](#download-and-preprocess)
    - [TFRecords generation](#tfrecords-generation)
  - [Input function pipeline](#input-function-pipeline)
  - [How to run](#how-to-run)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
  - [Configuration files included for this model](#configuration-files-included-for-this-model)
  - [References](#references)

## Overview of the model

This directory contains an implementation for pre-training Linformer model as described by Wang et al. in [[1]](https://arxiv.org/pdf/2006.04768.pdf). 

Linformer's main idea is to address $O(n^2)$ attention complexity, where $n$ is the length of the input sequence [[3]](https://arxiv.org/pdf/1810.04805.pdf). In this regard, two projection matrices $E$ and $F$ of shapes $k \times n$ are introduced where $k$ is the projected dimension. These matrices pre-multiply the Keys and Values matrices passed as inputs to multi-head self attention layer. The following figures gives more insight [[1]](https://arxiv.org/pdf/2006.04768.pdf):

<p align = "center">
<img src = ./images/LinformerAttentionBlock.png>
</p>
<p align = "center">
Fig.1 - Linformer multi-head linear self attention.
</p>

<p align = "center">
<img src = ./images/TransformerSelfAttention.png>
</p>
<p align = "center">
Fig.2 - Transformer multi-head self attention.
</p>

<p align = "center">
<img src = ./images/LinformerAttention.png>
</p>
<p align = "center">
Fig.3 - Linformer self attention
</p>

where $n$ is sequence length, $d_{m}$ is the embedding
dimension, $d_k$, $d_v$ are the hidden dimensions of the projection subspaces, $k$ is the projected dimension.


There are three variations of Linformer architecture based on the number of projection matrices [[1]](https://arxiv.org/pdf/2006.04768.pdf):

1. **Head-wise sharing**: for each layer, we share two projection matrices $E$ and $F$ such that
    $E_{i}$ = $E$ and $F_{i}$ = $F$ across all heads $i$.

2. **Key-value sharing**: in addition to head-wise sharing, we share Key and Value projection matrices.
For each layer, we create a single projection matrix $E$ such that
$E_{i}$ = $F_{i}$ = $E$ for each Key-Value projection matrix across all head $i$.

3. **Layer-wise sharing**: we use a single projection matrix $E$ across all layers, for all heads, and
for both Keys and Values matrices.

## Sequence of steps to perform
The following block diagram shows a high-level view of the sequence of steps you will perform in this example.
<p align = "center">
<img src = ./images/FlowDiagram.png>
</p>
<p align = "center">
Fig.4 - Flow Chart of steps to pretrain Linformer model
</p>

## Key features from CSoft platform used in this reference implementation

* Linformer supports Variable Sequence Length (VSL) configurations. At a high-level, this means that we can take advantage of Cerebras hardware's differences from GPU's to perform operations on different sized sequences in parallel, without processing padding tokens. This reduces the amount of time spent on computations that are never used in the end. To use VSL, simply add `use_vsl: True` to the `model` section of the configuration YAML file. For more details, see [[2]](https://www.cerebras.net/software/increasing-model-throughput-with-variable-tensor-shape-computations/).


## Structure of the code

* [configs](./configs/): YAML configuration files. The parameter config files are for different maximum sequence lengths and supported Linformer variations as elaborated in [Model Overview](#model-overview).
* [layers](./layers/): Implementations of Linformer-specific layers. 
* [LinformerModel.py](./LinformerModel.py): Model implementation. A bulk of the model is defined in this script. It inherits from [TFBaseModel](../../../common/tf/TFBaseModel.py). The model also uses Cerebras-defined layers that are located in [common/tf/layers](../../../common/tf/layers/).
* [data.py](./data.py): The entry point to the data input pipeline code. Defines `input_fn`.
* [model.py](./model.py): The entry point to the model. Defines `model_fn`.
* [run.py](./run.py): Training script. Performs training and validation.
* [utils.py](./utils.py): Miscellaneous scripts, including `get_params` to parse the `params` dictionary from the YAML files.

## Dataset generation

### Prerequisites

If you do not have [spaCy](https://spacy.io/), the natural language processing (NLP) library, then install it with the following commands:

```bash
pip install spacy
python -m spacy download en

```

### Download and preprocess
The Wikicorpus dataset is used for pre-training Linformer model. We use [mlcommons codebase](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) to download and preprocess the Wikicorpus dataset. The dataset can be preprocessed by following steps below:

1. Download and preprocess raw data using instructions mentioned [here](https://github.com/mlcommons/training/blob/master/language_model/tensorflow/bert/dataset.md). At the end of this step, there are `500` files with names `part-00xxx-of-00500` and `eval.txt` generated. These files contain one sentence of an article in one line and different articles separated by blank line.

2. After the dataset is preprocessed, create a file called `metadata.txt` which contains full paths to files from `part-00000-of-00500` to `part-00499-of-00500` in order generated from step 1. This file would be passed to the input arg `metadata_files` in [create_tfrecords_mlm_only.py](../bert/input/scripts/create_tfrecords_mlm_only.py)` to generate TFRecords for Masked Language Modeling task.

Sample `metadata.txt`:
```
/path_to_preprocessedfile/part-00001-of-00500.txt
/path_to_preprocessedfile/part-00002-of-00500.txt
/path_to_preprocessedfile/part-00003-of-00500.txt
...
...
/path_to_preprocessedfile/part-00498-of-00500.txt
/path_to_preprocessedfile/part-00499-of-00500.txt

```

### TFRecords generation

Use [create_tfrecords_mlm_only.py](../bert/input/scripts/create_tfrecords_mlm_only.py) to generate TFRecords for pre-training. Note that Linformer model uses maximum sequence length (MSL) of `512` for its pre-training runs unlike BERT [[4]](https://arxiv.org/pdf/1810.04805.pdf). Refer to [create_tfrecords_mlm_only.py](../bert/input/scripts/create_tfrecords_mlm_only.py) and [bert/input/scripts/README.md](../bert/input/scripts/README.md) for more details.

[create_tfrecords_mlm_only.py](../bert/input/scripts/create_tfrecords_mlm_only.py) should be used to create data without the next sentence prediction (NSP) labels. Note that [create_tfrecords_mlm_only.py](../bert/input/scripts/create_tfrecords_mlm_only.py) generates TFRecords to be used with [BertMlmOnlyTfRecordsDynamicMaskProcessor.py](../bert/input/BertMlmOnlyTfRecordsDynamicMaskProcessor.py) only when `--disable_masking` is passed as input to [create_tfrecords_mlm_only.py](../bert/input/scripts/create_tfrecords_mlm_only.py) or with [BertMlmOnlyTfRecordsStaticMaskProcessor.py](../bert/input/BertMlmOnlyTfRecordsStaticMaskProcessor.py).

<details>
    <summary><strong><em>Usage: create_tfrecords_mlm_only.py</em></strong></summary>

        Usage: create_tfrecords_mlm_only.py [-h] --metadata_files METADATA_FILES
                                                [METADATA_FILES ...]
                                                [--multiple_docs_in_single_file]
                                                [--multiple_docs_separator MULTIPLE_DOCS_SEPARATOR]
                                                [--single_sentence_per_line]
                                                [--allow_cross_document_examples]
                                                [--document_separator_token DOCUMENT_SEPARATOR_TOKEN]
                                                [--overlap_size OVERLAP_SIZE]
                                                [--buffer_size BUFFER_SIZE]
                                                [--input_files_prefix INPUT_FILES_PREFIX]
                                                --vocab_file VOCAB_FILE [--do_lower_case]
                                                [--max_seq_length MAX_SEQ_LENGTH]
                                                [--dupe_factor DUPE_FACTOR]
                                                [--short_seq_prob SHORT_SEQ_PROB]
                                                [--min_short_seq_length MIN_SHORT_SEQ_LENGTH]
                                                [--disable_masking]
                                                [--masked_lm_prob MASKED_LM_PROB]
                                                [--max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ]
                                                [--spacy_model SPACY_MODEL]
                                                [--mask_whole_word]
                                                [--output_dir OUTPUT_DIR]
                                                [--num_output_files NUM_OUTPUT_FILES]
                                                [--name NAME] [--seed SEED]

        Required arguments:
        --metadata_files METADATA_FILES [METADATA_FILES ...]
                            Path to the text file containing a list of file
                            names corresponding to the raw input documents
                            to be processed and stored; Multiple metadata
                            files must be separated by a space.
        --vocab_file VOCAB_FILE
                                Path to the vocabulary file.


        Optional arguments:
        -h, --help            Show this help message and exit.
        --multiple_docs_in_single_file
                                Pass this flag when a single text file contains
                                multiple documents separated by
                                <multiple_docs_separator> (default: False).
        --multiple_docs_separator MULTIPLE_DOCS_SEPARATOR
                                String which separates multiple documents in a single
                                text file. If newline character, pass `\n`.
                                There can only be one separator string for
                                all the documents.
                                (default: `\n`)
        --single_sentence_per_line
                                Pass this flag when the document is already
                                split into sentences, with one sentence in
                                each line. There is no requirement for further
                                sentence segmentation of a document
                                (default: False).
        --allow_cross_document_examples
                        Pass this flag when tokens for the same example can come from 
                        multiple documents (default: False).
        --document_separator_token DOCUMENT_SEPARATOR_TOKEN
                                If an example can span multiple documents, use this separator to 
                                indicate separate tokens of different documents 
                                (default: `[SEP]`).
        --overlap_size OVERLAP_SIZE
                                The overlap size between tokens of the current and previous 
                                example. Defaults to None, which sets the overlap to 
                                max_seq_len/4 (default: None).
        --buffer_size BUFFER_SIZE
                                Number of tokens to be processed at a time (default: 1000000).
        --input_files_prefix INPUT_FILES_PREFIX
                                Prefix to be added to paths of the input
                                files. For example, can be a directory where
                                raw data is stored if the paths are relative.
        --do_lower_case       Pass this flag to lower case the input text.
                                Must be True for uncased models and False for cased models. Note 
                                that if your vocab file has only lowercased letters, and you did 
                                not provide this flag, a lot of tokens will be mapped to `[UNK]` 
                                and vice versa (default: False).
        --max_seq_length MAX_SEQ_LENGTH
                                Maximum sequence length (default: 128).
        --dupe_factor DUPE_FACTOR
                                Number of times to duplicate the input data (with
                                different masks). For static masking, it is a common practice to 
                                duplicate the data, and provide different masking for the same 
                                input to learn more generalizable features (default: 10).
        --short_seq_prob SHORT_SEQ_PROB
                                Probability of creating sequences that are
                                shorter than the maximum sequence length
                                (default: 0.1).
        --min_short_seq_length MIN_SHORT_SEQ_LENGTH
                                The minimum number of tokens to be present in an
                                example if short sequence probability > 0. If None,
                                defaults to 2 + overlap_sizeAllowed values are between
                                [2 + overlap_size, max_seq_length-2) (default: None)
        --disable_masking     If False, TFRecords will be stored with
                                static masks. If True, masking will happen
                                dynamically during training (default: False).
        --masked_lm_prob MASKED_LM_PROB
                                Probability of replacing input tokens with a mask token `[MASK]` 
                                for a language modeling task (default: 0.15).
        --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                                Maximum number of masked LM predictions per
                                sequence (default: 20).
        --spacy_model SPACY_MODEL
                                The spaCy model to load (either a shortcut
                                link, a package name or a path). It is used to process the data 
                                files and segment them into sentences if the flag 
                                `single_sentence_per_line` is not set. Default model is set to 
                                the small English pipeline trained on written web text.
                                (default: en_core_web_sm).
        --mask_whole_word     Set to True to use whole word masking and
                                False to use per-WordPiece masking
                                (default: False).
        --output_dir OUTPUT_DIR
                                Directory where TFRecords will be stored.
                                (default: ./tfrecords/).
        --num_output_files NUM_OUTPUT_FILES
                                TFRecords will be separated into the
                                specified number of files on disk. The larger the number of 
                                files, the easier it becomes to parallelize writing/reading of 
                                the TFRecords (default: 10).
        --name NAME           Name of the dataset, i.e., prefix to use
                                for TFRecord names (default: "examples").
        --seed SEED           Seed for the random number generators (default: 0).


</details>


The following command can be used to generate TFRecords used for training the model.
```bash
python create_tfrecords_mlm_only.py --metadata_files=<full path to metadata.txt from step 2> --vocab_file=../../../../vocab/google_research_cased_L-12_H-768_A-12.txt --multiple_docs_in_single_file --single_sentence_per_line --overlap_size=0 --buffer_size=10000 --dupe_factor=1 --max_sequence_length=512 --disable_masking --output_dir=<full path to where the TFrecords should be written>
```
## Input function pipeline
We use [BertMlmOnlyTfRecordsDynamicMaskProcessor.py](../bert/input/BertMlmOnlyTfRecordsDynamicMaskProcessor.py) to read the raw unmasked examples from TFRecords and dynamically mask the sequences before passing them to the model for training.

## How to run

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our Developer Docs at:
https://docs.cerebras.net/en/latest/getting-started/tensorflow/index.html

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../PYTHON-SETUP.md), and simply run:

```
python run.py --mode train --params <path/to/yaml> --model_dir </path/to/model_dir>
```

For each of these commands
* `path/to/yaml` is a path to the YAML configuration file containing the model parameters. Parameters for the Linformer base configuration can be found in the [configs](./configs) directory. The parameter config files are for different maximum sequence lengths and supported Linformer variations as elaborated in [Model Overview](#model-overview).
* `path/to/model_dir` is the path to the model directory where compile and training artifacts will be saved.

Evaluation on the validation set can be run on GPU or Cerebras System with similar commands by passing the input arg `--mode=eval` to [run.py](./run.py).

## Configuration files included for this model

In the [configs](./configs/) directory, we have configuration files for Linformer. The configuration files provided use different variations of Linformer Attention as described in [Model overview](#model-overview) section and different projected lengths. The user can vary `model.projected_dims` parameter to change the projected dimension of attention. To use different variants of Linformer attention, the parameter `model.attention_style` can be set to either `linformer-shared-kv` or `linformer-shared-layers` or `linformer-shared-heads`.

All configs are meant for running in Pipeline mode with Appliance mode and Kubernetes. Slurm flow is available as a legacy support.

## References
[1] [Linformer: Self-Attention with Linear Complexity by Sinong Wang, et al.](https://arxiv.org/pdf/2006.04768.pdf)

[2] [VTS Conceptual Explanation Blog](https://www.cerebras.net/software/increasing-model-throughput-with-variable-tensor-shape-computations/)

[3] [Pipeline Execution Mode](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html#layer-pipelined-mode)

[4] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, et al.](https://arxiv.org/pdf/1810.04805.pdf)
