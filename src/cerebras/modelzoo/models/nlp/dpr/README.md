# Dense Passage Retrieval (DPR)

Dense Passage Retriever (DPR) is a technique introduced by Facebook Research that marked one of the biggest early successes in applying neural networks for retrieval. The goal of retrieval is to find passages that will help in answering a question. To accomplish this, DPR is composed of two sub-models: a question encoder, and a passage encoder. The idea is that questions and passages have different properties, so we can optimize each model (usually a BERT-based model) for each sub-domain.

For more information on using our DPR implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/dpr) in our documentation.
