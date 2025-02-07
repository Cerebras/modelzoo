# Data processing 
This directory collects python modules for data preprocessing stage. During this stage, we 
prepare the raw input data into the format that can be fed into the deep learning model.

Preprocessing stages differ from model to model, but on the high-level, most of them are targeting on the next tasks: 
1. Removing special symbols that are not part of the vocabulary (for example, Chinese symbols in the English text).
2. Converting words into the tokens (with tokenizers such as [BPE](#bpe), [WordPiece](#wordpiece), [FullTokenizer](#fulltokenizer)).
3. Adding special symbols to indicate the beginning, middle or the end of sequence.
4. Converting tokens into features that can be fed into the model (such as pytorch or tensorflow tensors with numerical values representing tokens in the vocabulary).

Here is an example of the input/output for stages above:
1. Removing Chinese symbols from English text:
   * `input: "Hello, tokenization world! 很高兴见到你!"`
   * `output: "Hello, tokenization world!"`
   
2. Adding special symbols (`[CLS]` in the beginning of the sequence, and `[SEP]` at the end): 
   * `input: "Hello, tokenization world!"`
   * `output: "[CLS] Hello, tokenization world! [SEP]"`
   
3. Converting words into the list of tokens with BPE: 
   * `input: "Hello, tokenization world!"`
   * `output: ["[CLS]", "Hello", ",", "token", "##ization", world, "!", "[SEP]"]`

4. Convert tokens into features (for [BERT](https://arxiv.org/abs/1810.04805)):
   * `input_ids: [101, 7592, 1010, 19204, 3989, 2088, 999, 102]`
   * `token_type_ids: [0, 0, 0, 0, 0, 0, 0, 0]`
   * `attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]`

## Tokenizer
Now let's talk about stage #2 in details. There is a list of ways to tokenize your data. 
In our preprocessing modules we are using some of the most popular algorithms for data tokenization such as 
[BPE](#bpe), [FullTokenizer](#fulltokenizer) (which is using [WordPiece](#wordpiece) tokenizer), and will also discuss how you can create your own tokenizer from 
scratch. 

A tokenizer is in charge of preparing the inputs for a model. 
Above listed tokenizers are used across the majority of our transformer-based models. You can find a different 
tokenization methods created for [T5](https://arxiv.org/abs/1910.10683) and [Transformer](https://arxiv.org/abs/1706.03762) networks. Please follow their specific documentation: [T5-README](nlp/t5/input), [Transformer-README](nlp/transformer/README.md).
If you want to add your own version, feel free to do so with the steps provided in the section [Creating your own Tokenizer](#creating-your-own-tokenizer).

### BPE 
Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model.
It’s used by a lot of Transformer models, including BERT, GPT, GPT-2, RoBERTa, BART, and DeBERTa. 

On a high-level, BPE is a simple form of data compression algorithm in which the most common pair of consecutive tokens of data is replaced with a token that does not occur in that data.
This algorithm creates a vocabulary, that can be used later to tokenize a bulk of text, thus compressing it into the smaller size 
to ease the further work such as model training. It is based on the idea that non-frequent tokens 
are redundant, and only confuses the models during training. 
For the full details about this algorithm, please refer to this blog post that has a very detailed 
explanation with examples: [Byte-Pair Encoding: Subword-based tokenization algorithm](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0#:~:text=BPE%20ensures%20that%20the%20most,subword%2Dbased%20tokenization%20algorithm%20does.).

Here is the example of the BPE algorithm at work:

* `input: "i am converting this sentence into bpe tokens!"`

* `output: ["i", "Ġam", "Ġconverting", "Ġthis", "Ġsentence", "Ġinto", "Ġb", "pe", "Ġtokens", "!"]`

In this example we used vocabulary with tokens `["!", "i", "pe", "Ġam", "Ġb", "Ġconverting", "Ġinto", "Ġsentence", "Ġthis", "Ġtokens"]`, and you can see the input is transformed from words into tokens.
Since this algorithm encodes on the token (or byte) level, during training the token `"Ġ"` was added to identify the word boundary so that the algorithm knows where each word begins.

You can find our own reference implementation at [BPETokenizer.py](./nlp/tokenizers/BPETokenizer.py).
In order to use it, please provide the next list of params:
```
:param str vocab_file: File containing vocabulary, each token in new line;
:param str encoder_file: File providing the mapping between tokens in the text and corresponding BPE tokens;
:param str errors: Optional param specifying how to handle errors in the decoding stage (default value: "replace").
:param iterable special_tokens: Optional param specifying special tokens to use (such as `[PAD]`, `[MASK]`, etc.) (default value: None).
```
### FullTokenizer
This tokenizer consists of two stages: additional pre-processing and tokenization with [WordPiece](#wordpiece) tokenizer.
During the first stage we perform additional text filtering such as removing strip accents, and checking for Chinese 
characters. At the second stage we apply [WordPiece](#wordpiece) tokenizer. 

#### WordPiece
On a high-level, WordPiece is another way to perform tokenization on the raw bulk of text. The key 
difference from BPE lies in the way the symbol pairs are chosen for adding to the vocabulary. Instead of relying on the frequency of the pairs, WordPiece chooses the one which maximises the likelihood of the training data.
There is a separate language model that is trained and while building a vocabulary, this algorithm adds 
new pairs of tokens maximizing this model likelihood. 
For the full details about this algorithm, please refer to this blog post: [WordPiece: Subword-based tokenization algorithm](https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7).

Here is the example of the WordPiece algorithm at work:

* `input: "i am converting this sentence into wordpiece tokens!"`

* `output: ["i", "am", "converting", "this", "sentence", "into", "word", "##piece", "token", "##s", "!"]`

In this example we used vocabulary with tokens `[..., "!", "##e", "##s", "am", "converting", "i", "into", "sentence", "this", "token", "word", ...]`, and you can see the input is transformed from words into tokens.
As in our previous example with BPE, we can see a special token `"##"` added to identify the word boundaries.

You can find our own reference implementation at [Tokenization.py](./nlp/tokenizers/Tokenization.py#L302).
In order to use it, please provide the next list of params:
```
:param str vocab_file: File containing vocabulary, each token in new line;
:param bool do_lower_case: Whether or not your text should be lower-cased (default: True).
```

### Creating your own Tokenizer 
In this section, you can see where and how you can add your own tokenizer, following examples from the 
previous sections in this documentation. 

The easiest way to add a new tokenizer is to add another class in this file [Tokenization.py](./nlp/tokenizers/Tokenization.py), and inherit it from [`BaseTokenizer`](./nlp/tokenizers/Tokenization.py#L37).
If you want to change the tokenizer class for your models, you need to update which tokenization class you're calling 
in the dataloader script. For example, for [BERT](../models/nlp/bert) these lines would need to be replaced with a call of a different 
class.

`BaseTokenizer` is what we called the stage #1 in the [`FullTokenizer`](#fulltokenizer), where we perform some basic 
grammar filtering on the raw bulk of text. By inheriting, you don't have to implement this basic text pre-processing from
scratch. However, there are a few methods that still need to be created: tokenization method, and a method to convert 
tokens to the indices. 

By default `BaseTokenizer` provides you with the [tokenize](./nlp/tokenizers/Tokenization.py#L207)
method that accepts a text as an input, and tokenizes it into tokens. If this method requires any changes for 
your tokenization algorithm, please overwrite it inside your own tokenizer. 

By default `BaseTokenizer` does not provide any method to convert your input tokens into the indices that can be 
fed into the model. You need to create you own following an example provided in the `FullTokenizer`: [convert_tokens_to_ids](./nlp/tokenizers/Tokenization.py#L321).

Below we are going to create an example of your own tokenizer, given `BaseTokenizer` as a base class: 

```python
from collections import defaultdict 
from tokenizers.Tokenization import BaseTokenizer

class MyOwnTokenizer(BaseTokenizer):
    def __init__(self):
       self.vocab = self._create_vocab()
       super(MyOwnTokenizer, self).__init__(vocab_file="my_own_vocab.txt", do_lower_case=True)
    
    def _create_vocab(self):
       # `0` is reserved for unknown token
       vocab = defaultdict(int)
       vocab["apple"] = 1
       vocab["bottle"] = 2
       vocab["it"] = 4
       vocab["as"] = 5
       vocab["characters"] = 6
       vocab["is"] = 7
       vocab["my"] = 8
       vocab["own"] = 9
       vocab["program"] = 10
       vocab["removes"] = 11
       vocab["such"] = 12
       vocab["tokenizer"] = 13
       vocab["using"] = 14 
       with open("my_own_vocab.txt", "w") as fout:
          for token in vocab.keys():
              fout.write(f"{token}\n")
       return vocab
        
    def tokenize(self, text):
         """
         Given the text input, perform: 
             1. convertion to lowercase;
             2. removal of special characters `?!`;
             3. splitting the words into tokens by the space characters.
         """
         text_lowered = text.lower()
         text_filtered = text_lowered.replace("?!", "")
         text_splitted = self._run_split_on_punctuation(text_filtered)
         output_tokens = []
         for text in text_splitted:
            output_tokens.extend(text.split())
         return output_tokens

    def convert_tokens_to_ids(self, tokens):
         indices = [self.vocab[token] for token in tokens]
         return indices  
```

Now, let's use this tokenizer: 
```python
   text = """
   This program is using my own tokenizer. It removes characters such as ?!, 
   and split words on punctuation symbols.
   """

   tokenizer = MyOwnTokenizer()
   tokens = tokenizer.tokenize(text)
   print(tokens)
   # ["this", "program", "is", "using", "my", "own", "tokenizer", ".", "it",
   # "removes", "characters", "such", "as", ",", "and", "split", "words", "on",
   # "punctuation", "symbols", "."]
   token_ids = tokenizer.convert_tokens_to_ids(tokens)
   print(token_ids)
   # [0, 10, 7, 14, 8, 9, 13, 0, 4, 11, 6, 12, 5, 0, 0, 0, 0, 0, 0, 0, 0]
```

As we can see the input text is tokenized, and the results can be later on used into the model training. 