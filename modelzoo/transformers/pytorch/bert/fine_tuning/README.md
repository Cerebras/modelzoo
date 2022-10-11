# BERT fine-tuning
This README is created to list the supported model configurations under this folder, and to provide some high-level details about their implementations.

We support Classifier, Regression, QA, Summarization, and Token classification tasks. 

For the Classifier and Regression models, the new output head is constructed by using two dense layers, feeding the `[CLS]` representation to a these layers. The first dense layer is loaded from the pre-trained NSP head and the second dense layer is initialized with random weights. The QA, Token Classifier and Summarization fine-tuning heads contain only one dense layer, which is initialized with random weights during fine-tuning.
