import torch

from anl_shared.transformers.pytorch.gpt2.input.gene_processor_utils import (
    training_data_generator,
)


class GenomicDataProcessor(torch.utils.data.IterableDataset):
    """
    A genomic dataset processor for GPT pre-training.
    Performs on-the-fly processing of data from genomic data files.

    Functionality includes:
        Reading data from text documents
        Creating creating input sequences and masks, and
        autoregressive LM labels

    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (str): Input directory containing fasta files with genomic sequences.
    - "vocab_file" (str): Vocabulary file, to build tokenization from
    - "max_sequence_length (int): Maximum length of the sequence to generate
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of samples loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    - "add_special_tokens" (bool): Flag to add BOS and EOS tokens.
    - "eos_token" (str): EOS token.
    - "pad_token" (str): PAD token.
    """

    def __init__(self, params):
        super(GenomicDataProcessor, self).__init__()

        self.data_dir = params["data_dir"]
        self.vocab_file = params["vocab_file"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]
        self.n_gram = params.get("n_gram", 3)
        self.skip_large_MSL = params.get("skip_large_MSL", False)

        self.shuffle = params["shuffle"]

        self.num_workers = params.get("num_workers", 8)
        self.drop_last = params.get("drop_last", True)

        self.add_special_tokens = params.get("add_special_tokens", True)
        self.eos_token = params.get("eos_tokens", "[SEP]")
        self.pad_token = params.get("pad_tokens", "[PAD]")

        assert self.batch_size > 0, "Batch size should be positive."


    def __iter__(self):
        """
        Iterating over the data to construct input features.
        """
        for example in training_data_generator( 
            data_dir=self.data_dir,
            vocab_file=self.vocab_file,
            max_sequence_length=self.max_sequence_length,
            n_gram=self.n_gram,
            skip_large_MSL=self.skip_large_MSL,
            inverted_mask=False,
            add_special_tokens=self.add_special_tokens,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            input_ids_dtype="int32",
            input_mask_dtype="int32",
            labels_dtype="int32",
        ):

            yield example

    def create_dataloader(self, is_training):
        """
        Classmethod to create the dataloader object.
        """
        data_loader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        return data_loader
