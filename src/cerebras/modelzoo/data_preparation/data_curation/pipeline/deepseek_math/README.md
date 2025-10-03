# Math Curation Pipeline

This directory contains scripts and configuration files for curating high-quality mathematical data, inspired by the DeepSeek Math data curation process.

## Directory Structure

* **`configs/`**
  Contains YAML configuration files for:

  * FastText model training
  * Text extraction from raw WARC files
  * URL mining, downloading, and iterative retraining using FastText

* **`extractors/`**
  Contains extractor modules used as pipeline components to extract mathematical content.

* **`filters/`**
  Contains filter modules used as pipeline components to refine and clean extracted data.

* **`writers/`**
  Contains writers modules used as pipeline components to write extracted data.

* **`readers/`**
  Contains reader modules used as pipeline components to read raw input data.

* **`utils/`**
  Contains utility modules used for latex parsing and table parsing.

* **`deduplication/`**
  Contains modules for minhash and url based deduplication.

## Main Scripts

* **`train_fasttext_math_classifier.py`**
  Trains a FastText classifier to identify mathematical content.

* **`text_extraction_pipeline.py`**
  Extracts high-quality mathematical text from raw WARC files obtained from Common Crawl.

* **`extraction_and_mining_pipeline.py`**
  Mines additional math-related URLs from previously extracted data, downloads the corresponding WARC files, and extracts mathematical content to iteratively expand the dataset.

## Quick Start

Run the following scripts to execute various stages of the pipeline:

1. **Train FastText Classifier**

   ```bash
   bash src/models/src/cerebras/modelzoo/data_preparation/data_curation/env_files/fasttext_training.sh
   ```

2. **Extract Text from Common Crawl**

   ```bash
   bash src/models/src/cerebras/modelzoo/data_preparation/data_curation/env_files/text_extraction_pipeline.sh
   ```

3. **Mine and Re-Extract Additional URLs**

   ```bash
   bash src/models/src/cerebras/modelzoo/data_preparation/data_curation/env_files/extraction_mining_pipeline.sh
   ```

