# Input data 
This file describes the inputs that are necessary for training the T5 and Transformer models. These are handled by the `T5DynamicDataProcessor` and `TransformerDynamicDataProcessor` classes, defined in the python scripts with the same name as the class. The descriptions below include the name of the features required, along with a short explanation of the feature, the shapes of the feature, and the data-type of the feature. 

Note that if you want to create your own dataloader, you will need to ensure that the input is prepared to match the following format. The model expects data of the specified format, and will not operate on other data.

## T5 model input

The features dictionary has the following key-value pairs:

`input_ids`: Encoder's input token IDs, padded with `0`s to `src_max_sequence_length`. 
For the unsupervised denoising spans objective on C4 dataset, these would be the tokenized original sentence with spans replaced by sentinel tokens (and padded). For example, before tokenization and padding, an input would be: "Thank you \<X\> me to your party \<Y\> week." See Figure 2 of the T5 paper for more detail.
For a supervised task, it would be the tokenized input sentence (and padding). For example, before tokenization and padding, a summarization input would be: "summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi...". See Figure 1 of the T5 paper for more detail.

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `torch.int32`.

`attention_mask`: Mask for padded positions. Has `1`s on the tokens that represent real text, and `0`s for padded positions.

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `torch.int32`.

`decoder_input_ids`: Decoder's input token IDs, padded with `0`s to `tgt_max_sequence_length`. 
For the C4 example, before tokenization and padding, the decoder input would be: "\<X\> for inviting \<Y\> last \<Z\>". See Figure 2 and Section 3.1.4 of the T5 paper for more detail. 
For the supervised example, before tokenization and padding, the decoder input would be: "six people hospitalized after a storm in attala county." See Figure 1 and Section 1 of the T5 paper for more detail.

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `torch.int32`.

`decoder_attention_mask`: Mask for padded positions. Has `1`s on the tokens that represent real text, and `0`s for padded positions.

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `torch.int32`.

`labels`: The labels are the decoder input token IDs shifted by one token. The `decoder_input_ids` have a specific start-of-sequence token prepended to the beginning, and the `labels` have a specific end-of-sequence token appended to the end. In this way, the label for the start-of-sequence token will be the first real token of the sentence. Then, the label for the first real token will be the second real token. This applies for all other tokens until finally, the label of the last real token will be the end-of-sequence token, so that the model learns to end sentences correctly.

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `torch.int32`.

## Transformer model input 

The features dictionary for Transformer has different specific examples of the data, but the same key-value pairs:

`input_ids`: Encoder's input token IDs, padded with `0`s to `src_max_sequence_length`. Contains tokens for the source language that the model is expected to translate. For example, before tokenization and padding, an input would be: "Das Parlament erhebt sich zu einer Schweigeminute." 

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `torch.int32`.

`attention_mask`: Mask for padded positions. Has `1`s on the tokens that represent real text, and `0`s for the padding tokens.

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `torch.int32`.

`decoder_input_ids`: Decoder's input token IDs, padded with `0`s to `tgt_max_sequence_length`. Contains the tokens in the target language corresponding to tokens in `input_ids`. For example, before tokenization and padding, a decoder input would be: "The House rose and observed a minute's silence."
- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `torch.int32`.

`decoder_attention_mask`: Mask for padded positions. Has `1`s on the tokens that represent real text, and `0`s for the padding tokens.

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `torch.int32`.


`labels`: The labels are the decoder input token IDs shifted by one token. Refer to the explanation of the T5 `labels` for more detail. 

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `torch.int32`.


