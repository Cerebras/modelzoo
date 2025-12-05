# KenLM Perplexity Scoring Pipeline

A comprehensive pipeline for training KenLM language models and computing perplexity scores for data quality assessment. This tool helps evaluate text quality by measuring how well a language model predicts the text, with lower perplexity indicating higher quality.

## Features

- **Flexible Pipeline**: Modular design supporting multiple input/output formats
- **Quality Assessment**: Perplexity-based text quality evaluation
- **Model Training**: KenLM model training with quantization support
- **Tokenization**: SentencePiece tokenizer training and integration
- **Performance**: Parallel processing for efficient data handling
- **Logging**: Comprehensive logging and statistics reporting


## Installation

1. Clone and install KenLM (Optional):
```bash
# Clone KenLM repository
git clone https://github.com/kpu/kenlm.git
cd kenlm
./bjam -j4
```

2. Set up the virtual environment using the provided script:
```bash
# Make the script executable
chmod +x src/models/src/cerebras/modelzoo/data_preparation/data_curation/env_files/install_requirements.sh

# Run the script with requirements.txt
./src/models/src/cerebras/modelzoo/data_preparation/data_curation/env_files/install_requirements.sh venv requirements.txt
```

3. Activate the virtual environment:
```bash
source venv/bin/activate  # On Unix/macOS

```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

1. Here's a complete example configuration file (`configs/train_and_eval.yaml`) with all entries filled:

```yaml
# Base runtime configuration
run_config:
  cpus_per_task: 4 # Cpus per task (slurm)
  num_tasks: 2    # Number of tasks per worker (slurm)
  executor_type: 'slurm'  # Type of executor to use (slurm/local)
  # num_workers: 4  (Only use if executor is local)

# Data input/output configuration
data_config:
  input_folder: "/path/to/input_data/"  # Path to your input data
  output_folder: "/path/to/preproc_output/"
  input_format: "parquet"  # Input file format (jsonl/csv/txt/parquet)
  text_column: "text"  # Column containing text data

# Pipeline steps configuration
pipelines:
  # This step picks up the config only from the data_config dataclass
  KenLMPreprocessor: {}
  
  KenLMTrainer:
    input_folder: "/path/to/preproc_output/"
    output_path: "/path/to/kenlm_model"
    kenlm_path: "/path/to/kenlm/repo/"
    vocab_estimate: 15000  # Estimated vocabulary size
    quantize: 8  # Quantization level (0 = no quantization)
    offsets: 255  # Maximum number of offsets
  
  SentencePieceTrainer:
    input_folder: "/path/to/preproc_output/"
    output_path: "/path/to/sentencepiece_model/"
    model_prefix: "tokenizer"
  
  CCPerplexityStats:
    model_dataset: "dataset-name"
    domain: "domain-name"
    model_path: "/path/to/kenlm_model/model.binary"
    tokenizer_path: "/path/to/sentencepiece_model/tokenizer.model"
    output_folder: "/path/to/perplexity/output/"
```

2. Run the pipeline:

```bash
python run.py --config configs/train_and_eval.yaml
```

## Pipeline Components

### 1. Preprocessor (`KenLMPreprocessor`)
- Tokenizes text into sentences and words
- Normalizes text (lowercase, whitespace)
- Handles multiple input formats (JSONL, CSV, Parquet, text)
- Supports parallel processing

### 2. KenLM Trainer (`KenLMTrainer`)
- Trains language models using KenLM
- Supports model quantization
- Configurable n-gram order and parameters
- Automatic model optimization

### 3. SentencePiece Trainer (`SentencePieceTrainer`)
- Trains subword tokenizers
- Supports multiple tokenization algorithms
- Configurable vocabulary size
- Handles large datasets efficiently

### 4. Perplexity Stats (`CCPerplexityStats`)
- Computes perplexity scores for text quality assessment
- Supports both custom and pre-trained models
- Handles multiple languages
- Provides detailed statistics

## Configuration Details

### Input/Output Formats
Supported formats:
- JSONL (`jsonl`)
- CSV (`csv`)
- Parquet (`parquet`)
- Text (`text`)
- HuggingFace Datasets (`huggingface`)

## Implementation Details

### Text Processing
- Uses NLTK for sentence and word tokenization
- Normalizes text using datatrove's text normalization
- Handles Unicode characters and special cases
- Supports parallel processing for large datasets

### Model Training
- KenLM models are trained using the official KenLM toolkit
- Supports both ARPA and binary model formats
- Includes model quantization for memory efficiency
- Validates model quality after quantization

### Perplexity Calculation
- Computes perplexity using KenLM's scoring
- Supports both custom and pre-trained models
- Handles multiple languages and domains
- Provides detailed statistics and metrics

## Examples

### Basic Usage
```python
from pipeline import BasePipeline
from config import KenLMPipelineConfig

# Load configuration
config = KenLMPipelineConfig.from_yaml("config.yaml")

# Create and run pipeline
pipeline = BasePipeline(**config.dict())
pipeline.run()
```

### Custom Model Training
```python
from train import KenLMTrainer
from config import KenLMTrainerConfig

config = KenLMTrainerConfig(
    input_folder="data/preprocessed",
    output_path="models/custom",
    quantize=8,
    backoff=1
)

trainer = KenLMTrainer(config)
trainer.run()
```

### Perplexity Evaluation
```python
from perplexity import CCNetPerplexityStats
from config import PerplexityStatsConfig

config = PerplexityStatsConfig(
    output_folder="results",
    model_dataset="cc_net",
    language="english"
)

evaluator = CCNetPerplexityStats(config)
evaluator.run()
```

## Logging and Monitoring

The pipeline provides comprehensive logging:
- Progress tracking for each pipeline step
- Error reporting and handling
- Performance metrics and statistics
- Model quality assessment results

Logs are stored in the `<output_folder>/logs` directory, organized by pipeline step.

> Slurm logs are stored in <current_working_directory>/slurm_logs
