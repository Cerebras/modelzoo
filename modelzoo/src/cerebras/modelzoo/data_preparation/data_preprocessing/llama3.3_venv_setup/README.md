# Background

The current modelzoo release uses transformer version 4.40.0 and tokenizers version 0.19.1. We need higher versions of transformers and tokenizers to be able to support data preprocessing with LLaMA 3.3 tokenizer. Therefore, we have created the following steps to setup a virtual environment which will allow the users to preprocess data for LLaMA 3.3 tokenizers.

# Steps to Preprocess Data with LLaMA 3.3 Tokenizer

## 1. Navigate to the Setup Directory
```bash
cd modelzoo/data_preparation/data_preprocessing/llama3.3_venv_setup
```

## 2. Create a Virtual Environment
```bash
python3.8 -m venv llama3.3-data-preprocessing
```

## 3. Install the Requirements
```bash
python -m pip install --upgrade pip
pip install -r requirements_llama3.3.txt
```

## 4. Setup the Data Config in a YAML File

### Example Config:
```yaml

setup:
  data:
    type: "local"
    source: "input/dir/here"
  output_dir: "output/dir/here"
  processes: 16
  mode: "pretraining"

processing:
  huggingface_tokenizer: "meta-llama/Llama-3.3-70B-Instruct"
  tokenizer_params:
    token: <insert_token_here>
  read_hook: "cerebras.modelzoo.data_preparation.data_preprocessing.hooks:chat_read_hook"
  read_hook_kwargs:
      multi_turn_key: "messages"
      multi_turn_content_key: "content"
      multi_turn_role_key: "role"

  shuffle_seed: 0
  shuffle: False
  use_ftfy: True
  ftfy_normalizer: "NFC"
  wikitext_detokenize: False
  UNSAFE_skip_jsonl_decoding_errors: False

dataset:
  use_vsl: False
```

For detailed instructions on data preprocessing, refer to the [Cerebras Documentation](https://training-docs.cerebras.ai/model-zoo/components/data-preprocessing/data-preprocessing).

## 5. Run the Preprocessor
```bash
python modelzoo/data_preparation/data_preprocessing/preprocess_data.py --config <data_config>.yaml
```
