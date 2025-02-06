# Using HuggingFace datasets for auto-regressive LM

There are two methods to use HuggingFace dataset in training/evaluation of language models such as GPT, [Converting HuggingFace dataset to HDF5 format](#converting-huggingface-dataset-to-hdf5-format) or [Using HuggingFace dataset On-the-fly without conversion to HDF5](#using-huggingface-dataset-on-the-fly-without-conversion-to-hdf5)

## Converting HuggingFace dataset to HDF5 format
In this approach, we iterate over the dataset once and write the samples in `h5` files, to be used by `HDF5Dataset` class, implemented in `dataset.py` [here](../../data/common/h5_map_dataset/dataset.py). This can be done by calling the function [`preprocess_data.py`](../data_preprocessing/preprocess_data.py). Please refer to the section on [Data Preprocessing](https://docs.cerebras.net/en/latest/wsc/Model-zoo/Components/Data-preprocessing/data_preprocessing.html) for a more detailed explanation.
We have two examples in this folder, using this conversion:
- [`HF_converter_example_Eli5.py`](./HF_converter_example_Eli5.py), Eli5 dataset defined in [`HuggingFace_Eli5.py`](./HuggingFace_Eli5.py) based on this HuggingFace [tutorial](https://huggingface.co/docs/transformers/tasks/language_modeling). 
- [`HF_converter_example_BookCorpus.py`](./HF_converter_example_BookCorpus.py), BookCorpus dataset defined in [`HuggingFace_BookCorpus.py`](./HuggingFace_BookCorpus.py). This is another example of HuggingFace dataset similar to Eli5 but larger in size.

The following code snippet shows how `HuggingFace_Eli5` dataset is defined:

```python
"""HuggingFace Eli5 Dataset"""

from datasets import load_dataset
from transformers import AutoTokenizer

from cerebras.modelzoo.data_preparation.huggingface.CSDataCollatorForLanguageModeling import (
    CSDataCollatorForLanguageModeling,
)

def HuggingFace_Eli5(
    split="train", num_workers=8, sequence_length=128
):
    eli5 = load_dataset("eli5", split="train_asks[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2, seed=0)
    eli5 = eli5[split]  # Select dataset split
    eli5 = eli5.flatten()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=eli5.column_names,
    )

    block_size = sequence_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_eli5.map(
        group_texts, batched=True, num_proc=num_workers
    )

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = CSDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    return dataset, data_collator
```

> **NOTE**: Cerebras GPT models expects the labels to be shifted in the dataloader rather the model (in contrast to HuggingFace). To address this difference, we have implemented a custom data collator function [CSDataCollatorForLanguageModeling](./CSDataCollatorForLanguageModeling.py) based on [DataCollatorForLanguageModeling](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling).

The following code snippet shows how to convert `HuggingFace_Eli5` dataset to HDF5 format:

```python
from cerebras.modelzoo.data_preparation.huggingface.HuggingFace_Eli5 import (
    HuggingFace_Eli5,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.convert_dataset_to_HDF5 import (
    convert_dataset_to_HDF5,
)

dataset, data_collator = HuggingFace_Eli5(split="train", num_workers=8, sequence_length=128)

convert_dataset_to_HDF5(
    dataset=dataset,
    data_collator=data_collator,
    output_dir="./eli5_hdf5_dataset/",
    num_workers=8,
)
```

## Using HuggingFace dataset On-the-fly without conversion to HDF5
You may choose to stream samples directly from your HuggingFace dataset without converting to HDF5. Since the preprocessing/tokenization can be CPU intensive and hurt the performance of the dataloader, this is usually more suitable for small datasets.

The class [`HuggingFaceDataProcessor.py`](./HuggingFaceDataProcessor.py) provides the required tools (sharding between the workers, shuffling and etc.) to connect any HuggingFace dataset (Map-Style or Iterable) to CS GPT models.

The DataProcessor [`HuggingFaceDataProcessorEli5`](../../data/nlp/gpt/HuggingFaceDataProcessorEli5.py) showcases HuggingFace Eli5 (Map-Style) dataset directly streamed to GPT-2 model and the DataProcessor [`HuggingFaceIterableDataProcessorEli5`](../../data/nlp/gpt/HuggingFaceIterableDataProcessorEli5.py) is an example of using the same dataset in Iterable format. The following code snippet shows how to define `HuggingFaceDataProcessorEli5` to be used by GPT models:

```python
class HuggingFaceDataProcessorEli5(HuggingFaceDataProcessor):
    """
    A HuggingFace Eli5 map-style Data Processor.
    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of batches loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    """

    def __init__(self, params):
        num_workers = params.get("num_workers", 0)
        split = params["split"]

        self.dataset, self.data_collator = HuggingFace_Eli5(
            split=split, num_workers=num_workers, sequence_length=128
        )

        # The super class will take care of sharding the dataset and creating the dataloader
        super().__init__(params)
```

To use the DataProcessor [`HuggingFaceDataProcessorEli5`](../../data/nlp/gpt/HuggingFaceDataProcessorEli5.py), specify the following in the model YAML config:

```yaml
train_input:
    data_processor: "HuggingFaceDataProcessorEli5"
    split: "train"
    batch_size: 128
    shuffle: True
    shuffle_seed: 1337
    num_workers: 8
    prefetch_factor: 10
    persistent_workers: True # Important to avoid seeding at each epoch
```
