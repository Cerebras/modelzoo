# Introduction

Transformer model is trained on [WMT-2016](https://www.statmt.org/wmt16/it-translation-task.html) which is a publicly available English to German machine translation dataset. To preprocess the raw WMT-2016, [wmt16_en_de.sh](./wmt16_en_de.sh) can be used. This script downloads the individual dataset components, concatenates these components, processes the raw text files, tokenizes the processed files, creates the byte-pair encoded vocabulary and adds special tokens to the vocabulary.

## Prerequisites

The preprocessing script uses Github repositories which are self contained, therefore, we don't need to install any special setup for the script to run.

## Input data

### Data download

WMT-2016 consists of `3` components:

- [Europarl-v7](http://www.statmt.org/europarl/v7/de-en.tgz)
- [CommonCrawl](http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz)
- [News Commentary corpora](http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz). 

These can be downloaded following the embedded links.

[Validation](http://data.statmt.org/wmt16/translation-task/dev.tgz) and [test](http://data.statmt.org/wmt16/translation-task/test.tgz) sets can also be downloaded following these links.

The files associated with each of the components will be downloaded and stored in the `./wmt16_en_de/data` directory. You can change the default directory and provide your own by setting the environment variable `$OUTPUT_DIR` as follows:

```bash
export OUTPUT_DIR="</preprocess/data/dir>"
```

### Input files format

As such there is no special input file required for the preprocessing, we just document the intermediate files downloaded here.

There are `5` files that are downloaded: 

```
europarl-v7-de-en.tgz
common-crawl.tgz
nc-v11.tgz
dev.tgz
test.tgz
```

`dev.tgz` and `test.tgz` are the validation and the test sets, respectively. Once downloaded, all the files are extracted and are put under their respective directories:

```
${OUTPUT_DIR}/data/europarl-v7-de-en
${OUTPUT_DIR}/data/common-crawl
${OUTPUT_DIR}/data/nc-v11
${OUTPUT_DIR}/data/dev
${OUTPUT_DIR}/data/test
```

The uncompressed data files, namely, `europarl-v7.de-en.en`, `commoncrawl.de-en.en` and `news-commentary-v11.de-en.en` are concatenated to form `train.en` which is the source language training data. `europarl-v7.de-en.de`, `commoncrawl.de-en.de` and `news-commentary-v11.de-en.de` form the target language data `train.de`. Following is the directory structure of the two generated files.

```
${OUTPUT_DIR}/train.en
${OUTPUT_DIR}/train.de
```

If you want to use your custom machine translation dataset, or any sequence to sequence dataset, first it needs to extracted if it is in a compressed format and needs to be stored under `"$( pwd )"/wmt16_de_en` as `train.en` (source language data) and `train.de` (target language data).

## Running the script

The script can be run using the following command:
```bash
source wmt16_en_de.sh
```

`wmt16_en_de.sh` uses [Mosesdecoder]((https://github.com/moses-smt/mosesdecoder)) to process raw files and tokenization. Tokenization process uses `8` threads and there can be a slowdown in this step if the number of cores available are not sufficient.

Once raw text is tokenized, [subword-nmt](https://github.com/rsennrich/subword-nmt.git) repo is used to generate subword units (Byte-Pair Encoding). And finally vocabulary is generated from the subword units.

This processing script will generate the following files:

- `wmt16_en_de/train.tok.clean.bpe.32000.en` is the source data file which contains tokenized `English` language sentences used for training.
- `wmt16_en_de/train.tok.clean.bpe.32000.de` is the target data file which contains tokenized `German` language sentences used for training.
- `wmt16_en_de/newstest2014.tok.clean.bpe.32000.en` is the source data file which contains tokenized `English` language sentences used for evaluation. 
- `wmt16_en_de/newstest2014.tok.clean.bpe.32000.de` is the target data file which contains tokenized `German` language sentences used for evaluation.
- `wmt16_en_de/vocab.bpe.32000.en` is the source data vocabulary which contains byte pair encoded tokens for the `English` language text.
- `wmt16_en_de/vocab.bpe.32000.de` is the target data vocabulary which contains byte pair encoded tokens for the `German` language text.

After the dataset is generated, please run the next two commands to split up the input dataset into multiple files, and generate `meta.dat` file.
We split the data into many files so that multiple worker nodes can read the files in parallel.
We want the speed of passing data to the CS system to keep up with the speed of the compute of the CS system, which requires multiple workers handling data.

```bash
python split_files.py --src_file <path-to-source-data-in-one-file> --tgt_file {path-to-target-translated-data-in-one-file}
--src_dir <path-to-source-directory> --tgt_dir {path-to-target-translated-directory}
--buffer_len <number-of-examples-to-store-in-one-output-file>
```

The `--buffer_len` argument describes how many examples will be in each output file. It defaults to `17581`, which gives us `256` output files for this dataset. If you change datasets, please consider updating these settings.

```bash
python create_meta.py --src_dir <path-to-source-directory> --tgt_dir {path-to-target-translated-directory}
```

The meta file will be generated inside the source directory.

Now `--src_dir` and `--tgt_dir` contains examples that can be used for model training.

## Output data

### Output directory structure

By default the output directory `src_dir`, contains `257` files, `256` text files and `1` meta file. The `src_dir` structure is as below (only first and last few files shown):

```
meta.dat
train.tok.clean.bpe.32000.en-0
train.tok.clean.bpe.32000.en-1
train.tok.clean.bpe.32000.en-10
...
train.tok.clean.bpe.32000.en-97
train.tok.clean.bpe.32000.en-98
train.tok.clean.bpe.32000.en-99
```

By default the output directory `tgt_dir`, contains `256` text files and meta file is shared from the `src_dir` . The `tgt_dir` structure is as below (only first and last few files shown):

```
train.tok.clean.bpe.32000.en-0
train.tok.clean.bpe.32000.en-1
train.tok.clean.bpe.32000.en-10
...
train.tok.clean.bpe.32000.en-97
train.tok.clean.bpe.32000.en-98
train.tok.clean.bpe.32000.en-99
```


### Output file structure

Each of source file and target file contains the raw text which is then dynamically processed into token ids in the [TransformerDynamicDataProcessor](../../../data/nlp/transformer/TransformerDynamicDataProcessor.py). 

Sample few lines in the `src_file` is shown below: 

```
At the same time , Z@@ uma ’ s revolutionary generation still seems un@@ easy leading South Africa in a post-@@ apar@@ thei@@ d era that is now 15 years old .
In a region that rever@@ es the elderly , Z@@ uma ’ s attach@@ ment to his rural traditions must be matched by an equal openness to the ap@@ peti@@ tes of the country ’ s youth .
Three in ten South Afri@@ cans are younger than 15 , meaning that they did not live a day under apar@@ thei@@ d .
Som@@ ehow Z@@ uma must find a way to honor his own generation ’ s commitment to ra@@ cial justice and national liber@@ ation , while em@@ power@@ ing the m@@ asses who daily suffer the sting of class differences and y@@ earn for material gain .
```

The `src_file` contains 4500962 lines.

Sample few lines in the `tgt_file` is shown below: 

```
Dem politischen Ökonomen Mo@@ el@@ et@@ si M@@ be@@ ki zufolge , ist Z@@ uma im Grunde seines Herz@@ ens „ ein Kon@@ serv@@ ativer “ . In diesem Sinne vertritt Z@@ uma das Südafrika von gestern .
Er ist Mitglied einer st@@ ol@@ zen Generation , die die Apar@@ thei@@ d be@@ zw@@ ang – und der anschließend ein frie@@ dlicher Übergang zu einer schwarzen Mehrheits@@ regierung gelang .
Das bleibt eine der größten Errungenschaften in der jün@@ geren Geschichte .
Gleichzeitig scheint sich Z@@ um@@ as revolution@@ äre Generation mit der Führung Sü@@ dafri@@ kas in der nun seit 15 Jahren dau@@ ernden Ära nach der Apar@@ thei@@ d noch immer un@@ wohl zu fühlen .
In einer Region , wo die älteren Menschen sehr ver@@ ehrt werden , muss Z@@ um@@ as Bin@@ dung an lan@@ dest@@ yp@@ ische Traditionen eine gleich@@ wer@@ tige Offenheit gegenüber den Bedürfnissen der Jugend des Landes gegenüber@@ stehen .
Drei von zehn Sü@@ dafri@@ kan@@ ern sind jün@@ ger als 15 und das bedeutet , dass sie nicht einen Tag unter der Apar@@ thei@@ d gel@@ ebt haben .
Ir@@ gend@@ wie muss Z@@ uma einen Weg finden , einerseits das Engagement seiner Generation hinsichtlich ethn@@ ischer Gerechtigkeit und nationaler Befreiung zu würdigen und andererseits den M@@ assen , die täglich unter Klassen@@ unterschie@@ den leiden und sich nach materi@@ ellen Verbesserungen seh@@ nen , mehr Mit@@ wirkungs@@ möglichkeiten einzuräumen .
```

The `tgt_file` contains 4500962 lines, corresponding to the lines in `src_file`.
