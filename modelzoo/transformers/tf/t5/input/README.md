## T5 model input
This file describes each of the inputs that the T5 model requires, which are managed by the `T5DynamicDataProcessor` class defined in `T5DynamicDataProcessor.py`. Each input description below includes the name of the feature, a short explanation, the shape of the tensor, and the data-type of the tensor.

Note that if you write your own dataloader, you must structure the data to match the following format. The model expects data in this format, and will not work otherwise.

The features dictionary has the following key-value pairs:

`encoder_input_ids`: Encoder's input token IDs, padded with `0`s to `src_max_sequence_length`.

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `tf.int32`.

`encoder_mask`: Mask for padded positions. Has `1`s on the tokens representing real text, 
and `0`s elsewhere.

- Shape: `[batch_size, src_max_sequence_length]`.
- Type: `tf.int32`.

`decoder_input_ids`: Decoder's input token IDs with a start-of-sequence token prepended, and padded with `0`s to `tgt_max_sequence_length`. 

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `tf.int32`.

`decoder_mask`: Mask for padded positions. Has `1`s on the tokens representing real text,  
and `0`s elsewhere.

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `tf.int32`.

`labels`: Decoder's input token IDs with an end-of-sequence token appended (the mapping between `decoder_input_ids` and `labels` follows the common training regime called [teacher forcing](https://en.wikipedia.org/wiki/Teacher_forcing)).

- Shape: `[batch_size, tgt_max_sequence_length]`.
- Type: `tf.int32`.

