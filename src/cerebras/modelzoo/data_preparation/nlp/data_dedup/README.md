# Data Deduplication Pipeline

# Environment Setup

**NOTE:** Skip this if you have already setup the model zoo environment as described in [PYTHON-SETUP.md](../../../../../../PYTHON-SETUP.md) if you're running on a Cerebras Wafer-Scale Cluster. If trying to run this locally, please follow the below steps.

The file [requirements.txt](./requirements.txt) contains the pre-requisites that are needed to enable a clean run of the scripts. Below is how to setup the environment:
```bash
virtualenv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt
```
# Step-by-step Guide
To perform deduplication, we used the [datasketch](http://ekzhu.com/datasketch/minhash.html) library and applied further optimizations to reduce memory consumption and increase parallelism. Our implementation is using producer-consumer schema in order to parallelize I/O operations that dominate our runtime. 

In addition, we applied code changes to reduce the memory utilization by keeping only one document per set of duplicates in memory.

The deduplication process includes multiple stages such as building MinHashLSH index, performing querying into the index to locate duplicates, building a graph representation to locate connected components with duplicates and finally filtering duplicates in each component.

Below you can find commands to reproduce each step in the pipeline:
### Step 3.1: MinHash Generation

MinHash generation can be a very slow process. We recommend running it separately before creating a MinHashLSH index. 

To calculate MinHash object for each document, we strip, lowercase, remove punctuation, consecutive spaces, newlines and tabs from each document.
Afterwards, we construct a list of 13-grams that are later used as features to create a document signature to add into MinHashLSH
index. 

(More details about MinHash can be found at [Identifying and Filtering Near-Duplicate Documents](https://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf).)

We also apply NFC normalisation, as well as filter out short documents, before yielding the documents. 

For custom datasets, you also need to specify the `jsonl_key` as well as the format of the dataset. By default, the `jsonl_key` is set to be `text` and the format to be `jsonl`.

Here is an example command to run MinHash generation: 
```
python to_hash.py --dataset_name <dataset-name> --input_dir <input-dir> --output_dir <output-dir> --job_id <job-id> --jsonl_key <jsonl-key> --format <format-of-the-dataset>
```

To reduce the total processing time, multiple jobs can be run for each corpus in parallel by using multiple job IDs - starting from 0. By default, the script works expecting one job, but if you wish, you can replicate the scripts across multiple machines and the script chunks the list of files to be processed equally, to do the MinHash generation. 

NOTE: This assumes that the dataset is present at a common place that is accessible by all the machines. 

### Step 3.2: Duplicate Pairs Generation 
In this step, we build a MinHashLSH index and query it to locate near duplicates 

(More reading here: [Chapter 3, Mining of Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf). )

We use a Jaccard similarity threshold of 0.8 by default to determine whether a pair of documents should be considered as a duplicate, but you can specify it according to your own needs.
```bash
python generate_duplicate_pairs.py --input_dir <output-directory-from-previous-step> --out_file <output-directory>/duplicates/duplicate_pairs.txt
```
### Step 3.3: Duplicate Graph Construction & Search for Connected Components 
After locating duplicate pairs, we need to find connected components containing documents that are duplicates with each other. 

To make it more illustrative, consider  these pairs: `(A, B), (A, C), (A, E)`. 

We are going to form a cluster of `(A, B, C, E)` and keep only one document from the component. 

We evaluated the performance and memory consumption of [networkx](https://networkx.org/), [graphtool](https://graph-tool.skewed.de/), and [networkit](https://networkit.github.io/). [networkit](https://networkit.github.io/) offered most efficient implementation as it is designed to work with large graphs and features great parallelism.

Below you can find an example command how to construct a graph from documents pairs: 
```bash
python generate_connected_components.py --input_dir <output-directory-from-previous-step}/duplicates --out_file <output-directory>/duplicates/connected_components.pickle
```

### Step 3.4: Generate Final List of Duplicates 
In this step, we generate the final deduplicated dataset. We dump the original dataset in fixed-size files, in the `jsonl.zst` format, whose size is configurable in the `deduplicate_dataset.py` file. By default, we create files of 16 MB. 

Below you can find an example command on how to generate the final deduplicated dataset:
```bash
python deduplicate_dataset.py --input_file <output-directory-from-previous-step>/duplicates/connected_components.pickle --input_dir <input-dir> --output_dir <final-output-dir> --format <format-of-dataset>
```