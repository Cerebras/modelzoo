## Transformer input function

This section describes the input data format expected by the model input function.

#### Transformer features dictionary
The features dictionary has the following key-value pairs:

`encoder_input_ids`: Encoder's input token IDs, padded with `0`s to `src_max_sequence_length`. Contains tokens for the source language that the model is expected to translate.

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `int32`.

`encoder_mask`: Mask for padded positions. Has `1`s on the encoder's input ids padded positions
and `0`s elsewhere.

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `int32`.

`encoder_input_length`: Length of `encoder_input_ids` excluding `pad` tokens

- Shape: `[batch_size, ]`.
- Type: `int32`.


`decoder_input_ids`: Decoder's input token IDs, padded with `0`s to `tgt_max_sequence_length`. Contains the tokens in target language corresponding to tokens in `encoder_input_ids`
- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `int32`.

`decoder_mask`: Mask for padded positions. Has `1`s on the decoder's input ids padded positions
and `0`s elsewhere.

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `int32`.

`decoder_input_length`: Length of `decoder_input_ids` excluding `pad` tokens

- Shape: `[batch_size, ]`.
- Type: `int32`.

`loss_scale`: The loss scaling factor computed as `batch_size/num_valid_tokens_in_batch_of_decoder_input_ids`. 
This scalar is broadcasted to be of same shape as `decoder_input_ids`
- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `float16`

#### Transformer label tensor

Decoder's output token IDs appended by the end of sequence token (teacher-forcing style). We shift the `decoder_input_ids` to the left and append it by a end sequence token.

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `int32`.

The Transformer model input dataloaders which generates the tf.data.Dataset object is located [here](./TransformerDynamicDataProcessor.py)

## Dataset preparation for Transformer.
In this section you can find an information about the dataset which was used for Transformer model pre-training.

### Workshop on Statistical Machine Translation - 2016 (WMT-16) English to German Translation dataset 
This is a publicly-available translation dataset hosted [here](https://www.statmt.org/wmt16/translation-task.html) and contains data from News commentary and European parlimentary proceedings.

The following tools allow the processing of the training data into tokenized format:

* Tokenizer tokenizer.perl
* Detokenizer detokenizer.perl
* Lowercaser lowercase.perl
* SGML Wrapper wrap-xml.perl
* [subword-nmt package](https://github.com/rsennrich/subword-nmt)

These tools are available in the [Moses git repository](https://github.com/moses-smt/mosesdecoder).


### How to generate the training dataset
Please use the script [`wmt16_en_de.sh`](./data_processing/wmt16_en_de.sh) to download, preprocess and tokenize the dataset.

Run the command: 
```bash
cd ./data_processing
source wmt16_en_de.sh
```

The steps in the above script can be summarized as below:
1. Download data from [wmt16 webpage](https://www.statmt.org/wmt16/translation-task.html). This consists of europarl-v7, commoncrawl and news-commentary corpora.
2. Concatenate these to form the training dataset
3. Concatenate newstest2015 and newstest2016 to create validation dataset
4. Process raw files using [Mosesdecoder](https://github.com/moses-smt/mosesdecoder)
5. Create shared byte-pair encoding vocabulary using 32000 merge operations
6. Add special tokens to the beginning of the vocab files


## Citation

[1] [Attention Is All You Need by Vaswani, et al.](https://arxiv.org/pdf/1706.03762.pdf)

[2] [WMT16 Dataset](https://www.statmt.org/wmt16/translation-task.html)

[3] [Subword Neural Machine Translation](https://github.com/rsennrich/subword-nmt)

