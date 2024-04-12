# BERT Summarization Input

## BertSumCSVDataProcessor.py

The features dictionary has the following key-value pairs (shown in [BertSumCSVDataProcessor.py](BertSumCSVDataProcessor.py)):

`input_ids`: Input token IDs, padded with `0`s to `max_sequence_length`.

- Shape: `[batch_size, max_sequence_length]`.
- Type: `torch.int32`.

`attention_mask`: Mask for padded positions. Has `0`s on the padded positions
and `1`s elsewhere.

- Shape: `[batch_size, max_sequence_length]`.
- Type: `torch.int32`.

`token_type_ids`: Segment IDs. A segment is equal to `0` or `1`
 conditioned on index of the sentence is odd or even. For example, for `[sent1, sent2, sent3, sent4, sent5]` we will assign `[0, 1, 0, 1, 0]`.

`label_ids`: The label tensor. Carries the labels for each sentence. It's binary labels that indicate if each sub-sentences should be included in the summary. Please note that `max_cls_tokens` indicates the number of sub-sentences in our input text.

- Shape: `[batch_size, max_cls_tokens]`.
- Type: `torch.int32`.

`cls_indices`: CLS indices. Specifies indices in the `[CLS]` tokens in the `input_ids`.

- Shape: `[batch_size, max_cls_tokens]`.
- Type: `torch.int32`.

`cls_weights`: CLS tokens weights. Equal to `1` for all real `[CLS]` tokens, and `0` for all padded.

- Shape: `[batch_size, max_cls_tokens]`.
- Type: `torch.float32`.



A demo example of the input string and segment ID structure for single extractive_summarization:

``` bash
Tokens:   [CLS] sent one [SEP][CLS] sent two [SEP][CLS] sent three [SEP][CLS] sent four [SEP][PAD] [PAD] ...
Segments:   0    0    0    0    1    1    1    1   0     0     0     0    1    1    1    1     0    0    ...
label       0                   1                  0                      0
```

NOTE: The input tokens will be converted to IDs using the vocab file.

___

## Generating CSV files for summarization task

1. Download and unzip CNN and DailyMail stories from [DeepMind Q&A Dataset](https://cs.nyu.edu/~kcho/DMQA/). Specifically, in that page there are two categories of data, i.e., CNN and DailyMail, each of them has one section of data called stories. Download these two section of data through the given link, then unzip them and store all story files into one directory.

2. Download and unzip Stanford CoreNLP from [CoreNLP](https://stanfordnlp.github.io/CoreNLP/). Then execute the following command:

```bash
export CLASSPATH=/path/to/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar
```

by replacing ``/path/to`` with the location where the unzipped Stanford CoreNLP was saved.

3. Sentence splitting and tokenization. Execute:

```python
python write_csv_data.py --mode tokenize --input_path INPUT --output_path OUTPUT
```

by replacing:

- ``INPUT`` with the location where the unzipped and merged CNN/DailyMail stories are saved, and
- ``OUTPUT`` with the location where the tokenized CNN/DailyMail should be saved.

4. Format to simpler JSON files. Execute:

```python
python write_csv_data.py --mode convert_to_json_files --input_path INPUT --output_path OUTPUT --map_path MAP --lower_case
```

by replacing:

- ``INPUT`` with the location where tokenized CNN/DailyMail were saved, and
- ``OUTPUT`` with the location where simplified JSON files should be saved, and
- ``MAP`` with where URLs of the stories are stored, which provide split into training, testing and validation parts (these can be downloaded [here](https://github.com/nlpyang/BertSum/tree/05f8c634197d0ed1be8157d71f29aa7765abdd2a/urls).
- Specify ``--lower_case`` flag if the text need to be lower-cased.

5. Format to CSV files. Execute:

```python
python write_csv_data.py --mode convert_to_bert_format_files --input_path INPUT --output_path OUTPUT --lower_case --vocab_file /path/to/vocab.txt
```

by replacing:

- ``INPUT`` to where simplified JSON files were stored,
- ``OUTPUT`` to where BERT CSV files should be stored.
- Specify ``--lower_case`` flag if the text need to be lower-cased.
- Specify ``vocab_file`` by replacing ``/path/to`` to where the vocab file is stored.
