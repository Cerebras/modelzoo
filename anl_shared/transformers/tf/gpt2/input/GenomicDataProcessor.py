import tensorflow as tf

from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset
from modelzoo.common.tf.input.utils import transform_dataset
from anl_shared.transformers.tf.gpt2.input.gene_processor_utils import (
    training_data_generator,
)


class GenomicDataProcessor:
    """
    A text dataset processor for GPT pre-training.
    Performs on-the-fly processing of data from text.
    Functionality includes:
        Reading data from text documents
        Creating creating input sequences and masks, and
        autoregressive LM labels
    :param dict params: dict containing training 
        input parameters for creating dataset.
    Expects the following fields:
    - "metadata_files" (str or list of str): A string or strings list each
      pointing to a metadata file. A metadata file contains file paths for
      flat text cleaned documents. It has one file path per line. 
      The cleaned cleaned files have one paragraph per line and are
      separated by an empty line. 
    - "data_dir" (str):  Directory path containing individual fasta files, each with 
        one or more genomic sequences. One sequence = one sample.
    - "vocab_file" (str): Vocabulary file, to build tokenization from
    - "max_sequence_length" (int): Maximum length of the sequence to generate
    - "n_gram" (int): Number of BP in a token. Default = 1
    - "batch_size" (int): Batch size.
    - "seed" (int): Seed for random number generator.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_buffer" (int): Shuffle buffer size.
    - "shuffle_seed" (int): Shuffle seed.
    - "repeat" (bool): Flag to enable data repeat.
    - "add_special_tokens" (bool): Flag to add BOS and EOS tokens. 
    - "eos_token" (str): EOS token.
    - "pad_token" (str): PAD token. 
    - "mixed_precision" (bool): Casts input mask to fp16 if set to True.
      Otherwise, the generated mask is float32.
    """

    def __init__(self, params):

        self.data_dir = params["data_dir"]
        self.vocab_file = params["vocab_file"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]
        self.n_gram = params.get("n_gram",1)

        self.shuffle = params["shuffle"]
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", None)
        self.repeat = params["repeat"]

        self.add_special_tokens = params.get("add_special_tokens", True)
        self.eos_token = params.get("eos_tokens", "[SEP]")
        self.pad_token = params.get("pad_tokens", "[PAD]")

        self.seed = params.get("seed", None)  # used for random number generator
        self.mp_type = "float16" if params.get("mixed_precision") else "float32"
        # for sharding on the Cerebras System, we need to explicitly retrieve TF_CONFIG
        self.use_multiple_workers = params.get("use_multiple_workers", False)

        self.skip_large_MSL = params.get("skip_large_MSL",False)

        assert self.batch_size > 0, "Batch size should be positive."

    def create_tf_dataset(self, is_training=True, input_context=None):
        """
        Create tf dataset.
        :param bool is_training: Specifies whether the data is for training
        :param dict input_context: Given by distributed strategy for training
        :returns: tf dataset
        """

        def _data_generator():
            return training_data_generator(
                self.data_dir,
                self.vocab_file,
                self.max_sequence_length,
                n_gram = self.n_gram,
                skip_large_MSL=self.skip_large_MSL,
                inverted_mask=True,
                add_special_tokens=self.add_special_tokens,
                eos_token=self.eos_token,
                pad_token=self.pad_token,
                input_ids_dtype="int32",
                input_mask_dtype=self.mp_type,
                labels_dtype="int32",
                seed=self.seed,
            )

        features_types = {
            "input_ids": tf.int32,
            "input_mask": getattr(tf, self.mp_type),
        }

        features_shapes = {
            "input_ids": [self.max_sequence_length],
            "input_mask": [self.max_sequence_length],
        }

        dataset = tf.data.Dataset.from_generator(
            _data_generator,
            output_types=(features_types, tf.int32),
            output_shapes=(features_shapes, [self.max_sequence_length]),
        )

        dataset = shard_dataset(
            dataset, self.use_multiple_workers, input_context
        )

        return transform_dataset(
            dataset=dataset,
            map_fn=None,
            batch_size=self.batch_size,
            is_training=is_training,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
            map_before_batch=True,
        )
