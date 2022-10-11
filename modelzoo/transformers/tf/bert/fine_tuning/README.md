# BERT fine-tuning
This README is created to list the supported model configurations under this folder, and to provide some high-level details about their implementations.

We support Classifier, Regression, QA, Summarization, and Token classification tasks. 

For the Classifier and Regression models, the new output head is constructed by using two dense layers, feeding the `[CLS]` representation to a these layers. The first dense layer is loaded from the pre-trained NSP head and the second dense layer is initialized with random weights. The QA, Token Classifier and Summarization fine-tuning heads contain only one dense layer, which is initialized with random weights during fine-tuning.

When fine-tuning a pre-trained model, the model does not need to be redefined in YAML. Instead, it is possible to load the YAML section from the pre-training config using `bert_pretrain_params_path`. See [qa/configs/params_bert_base_squad.yaml](qa/configs/params_bert_base_squad.yaml#L20) for example. If `bert_pretrain_params_path` is not used, then the model section of the config needs to exactly match the model used during pre-training (with the exception of the fine-tuning heads) in order to ensure that weights can be loaded properly.
