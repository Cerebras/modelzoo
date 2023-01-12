# Abstractive Summarization with GPT-J

- [Introduction](#introduction)
- [Sequence of the steps to perform](#sequence-of-the-steps-to-perform)
- [Structure of the code](#structure-of-the-code)
- [Dataset](#dataset)
  - [Download and extract the dataset](#download-and-extract-the-dataset)
  - [Create TFRecords](#create-tfrecords)
  - [Data processing](#data-processing)
- [Run fine-tuning on CS system](#run-fine-tuning-on-cs-system)
- [Run fine-tuning on GPU and CPU](#run-fine-tuning-on-gpu-and-cpu)
- [Configuration files included for this model](#configuration-files-included-for-this-model)
- [Citations](#citations)


## Introduction
In Natural Language Processing (NLP), [Abstractive Summarization](https://towardsdatascience.com/understanding-automatic-text-summarization-2-abstractive-methods-7099fa8656fe)  is a task that aims to generate a succinct summary of the source text. Unlike [Extractive Summarization](https://towardsdatascience.com/understanding-automatic-text-summarization-1-extractive-methods-8eb512b21ecc), abstractive summarization potentially paraphrases the source text instead of copying the relevant phrases from the source text to the summary. Abstractive summarization lends itself to a number of applications in various domains, such as books and literature paraphrasing, scientific literature comparison and summarization, financial markets research, legal and medical document analysis etc. 

Transformer [[1]](https://arxiv.org/pdf/1706.03762.pdf) based models such as GPT-J [[2]](https://github.com/kingoflolz/mesh-transformer-jax) are very effective in the abstractive summarization task when fine-tuned on a summarization dataset. Here we demonstrate how to easily summarize a text using a pre-trained GPT-J [[2]](https://github.com/kingoflolz/mesh-transformer-jax) model. 

## Sequence of the steps to perform

Following block diagram illutrates the sequence of steps you would perform to run abstractive summarization.

![](./images/Abstract-Summ.png)

## Structure of the code
* `configs/`: YAML configuration files for abstractive summarization using GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax).
* `input/`: contains scripts to generate TFRecords for the abstractive summarization dataset.


## Dataset
We use [CNN Daily Mail](https://arxiv.org/abs/1506.03340) dataset for abstrsactive summarization task. This dataset consists of news articles and the corresponding summaries.  

### Download and extract the dataset
Download CNN Daily Mail dataset from this [link](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail) and extract under `./raw` folder.
```bash
unzip cnn_stories_tokenized.zip -d ./raw
```

### Create TFRecords
We use the `vocab file` and `encoder file` of GPT-2 [[5]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). To download go to this [link](https://github.com/openai/gpt-2/blob/master/download_model.py). 

Train TFRecords are generated using the following commands and saved under `./cnn_dailymail/train`: 

```bash 
cd fine_tuning/abstractive_summarization
python create_tfrecords.py --data_file ./raw/train.bin --vocab_file </path/to/vocab-file> --encoder_file </path/to/encoder-file> --max_seq_length 1024 --output_dir ./cnn_dailymail/train --name train
```

And validation set is generated using the following commands and saved under `./cnn_dailymail/val`: 

```bash 
cd fine_tuning/abstractive_summarization
python create_tfrecords.py --data_file ./raw/val.bin --vocab_file </path/to/vocab-file> --encoder_file </path/to/encoder-file> --max_seq_length 1024 --output_dir ./cnn_dailymail/val --name val
```

`max_seq_length` is the maximum sequuence length or MSL which is set to be `1024` for GPT-J [[2]](https://github.com/kingoflolz/mesh-transformer-jax).

### Data processing
Each example consists of an article and abstract. If an example has a missing article or missing abstract, we skip it. If the length of the abstract is longer than the article, we also skip it (as this is likely an unrealistic example). Note that when generating a test set (i.e. from `./raw/test.bin`) for use in a publication, make sure to not skip or truncate the data (the command above is only for `train` and `val` set).

If `--truncate_article` is passed and if the total sequence length of the post-processed example is greater than the MSL and entire abstract fits, we truncate the article, otherwise we skip it.

By default every sequence is structured as: 

```
[eos_id, article_token_ids, sep_id, abstract_token_ids, pad_ids]
``` 
and the label is structured as,
```
[article_token_ids, eos_id, abstract_token_ids, eos_id, pad_ids]
```

where, `eos_id=pad_id` and `sep_id` is distinct. `EOS` is used at the start of the sentence as well. This is in contrast to this [reference](https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/blob/91963b12a59dc981f136f98df046f6dc584bd8a5/dataset.py#L49), where `EOS` is not added. We are using it for consistency as our pre-training dataset includes `EOS`.

We are also using a new `SEP` token and because we have some extra unused `vocab` slots available in GPT-J [[2]](https://github.com/kingoflolz/mesh-transformer-jax), we use one of those for `SEP`. We add this special token to the default tokenizer by passing in `special_tokens` (similar to this [reference](https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/blob/91963b12a59dc981f136f98df046f6dc584bd8a5/dataset.py#L49)). For clarity, we included "<|endoftext|>" in the `special_tokens` list but it already exists in the vocabulary and is not added again.

```
tokenizer = BPETokenizer(
  vocab_file, encoder_file, special_tokens=["<|sep|>", "<|endoftext|>"]
)
```

Data processing is largely the same as Cerebras GPT-2's [data processing](../../../gpt2/input/create_tfrecords.py). While the pre-training [TFRecord generation script](../../../gpt2/input/create_tfrecords.py) includes a `short_seq_prob` variable, we've removed this for now. We don't use the sliding window approach (i.e. there is no `overlap` variable when generating sequences, as there is in this [script](../../../gpt2/input/data_processor_utils.py)).

These examples are stored as TFRecords. Each record has an `input`, `label`, and `input mask` (which is set to `0` for the mask). We only consider loss over the abstract (i.e. the input mask has a `0` for every token in the article). To also consider loss over article, one can re-generate the dataset with `1's` in the mask for tokens corresponding to the article. In [./input/data_processor_utils.py](input/data_processor_utils.py) change the line:

```
input_mask = [0] * (1 + len(article_token_ids)) + [1] * (
                1 + len(abstract_token_ids)
            )
```

to:

```
input_mask = [1] * (1 + len(article_token_ids)) + [1] * (
                1 + len(abstract_token_ids)
            )
```

Note that at the time of TFRecord creation, `inverted_mask=False`, which makes actual tokens have to have `1's` and masks have `0's`. 

## Input function
GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax) uses the same input dataloader class as GPT-2 [[6]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) which is [`GptTfRecordsProcessor`](../gpt2/input/GptTfRecordsProcessor.py). It consumes the processed data and generates a features dictionary and a label tensor for the fine-tuning. The features dictionary for GPT-J fine-tuning is as follows:

- `input_ids`: Input token IDs, padded with `0` to `max_sequence_length`.
  - Shape: [`batch_size`, `max_sequence_length`].
  - Type:   `tf.int32`

- `input_mask`: Mask for padded positions. Has values 1 on the padded positions and 0 elsewhere.
  - Shape: [`batch_size`, `max_sequence_length`]
  - Type:  `tf.float32`

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our Developer Docs at:
https://docs.cerebras.net/en/latest/getting-started/tensorflow/index.html

### Run fine-tuning on GPU and CPU
To run pre-training on GPU/CPU, use the following command:
```bash
python run.py --mode train_and_eval --params fine_tuning/abstractive_summarization/configs/params_finetuning.yaml --model_dir </path/to/model_dir> --max_steps <num_train_steps>
```
Note that our model implementation and run scripts are compatible to run on GPU, however handling any GPU cluster related programming is up-to the user.

## Configuration files included for this model

This repository facilitates fine-tuning GPT-J [[2]](https://github.com/kingoflolz/mesh-transformer-jax) for abstractive summarization task. The config files are located under [./configs](./configs) directory.

* [configs/params_finetuning.yaml](configs/params_finetuning.yaml) does fine-tuning for GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax)  model using a pre-trained checkpoint on [CNN Daily Mail](https://arxiv.org/abs/1506.03340) dataset. The GPT-J model size is: `hidden_size=4096`, `num_hidden_layers=28`, `num_heads=16`.

## Citations
[1] [Attention Is All You Need by Vaswani, et al.](https://arxiv.org/pdf/1706.03762.pdf), 2017.

[2][Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX](https://github.com/kingoflolz/mesh-transformer-jax), May 2021.

[3] [Teaching machines to read and comprehend by Hermann et. al.](https://arxiv.org/pdf/1506.03340.pdf), NeurIPS-2015.

[4] [Generating Text Summaries Using GPT-2 on PyTorch with Minimal Training](https://blog.paperspace.com/generating-text-summaries-gpt-2/).

[5] [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019.