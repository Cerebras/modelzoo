# SantaCoder

SantaCoder is a 1.1B parameter decoder-only transformer model developed by the BigCode community. 

Architecturally, SantaCoder uses Multi-Query Attention (MQA) for efficient inference and supports Fill-in-the-Middle (FIM) generation, allowing the model to complete and infill code based on context. It employs a 49K BPE tokenizer trained on raw bytes and achieves strong performance on MultiPL-E and HumanEval benchmarks despite being significantly smaller than competing models.

For more information on using our SantaCoder implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/santacoder) in our documentation.
