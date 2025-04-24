## StarCoder

StarCoder is a family of decoder-only transformer models developed by the BigCode initiative, optimized for high-quality code generation. The flagship model, StarCoder (15.5B), was trained on 1 trillion tokens, with an emphasis on multilingual support and strong performance in Python.

Architecturally, StarCoder builds on the transformer decoder backbone with several enhancements: it uses multi-query attention (MQA) for fast inference, supports fill-in-the-middle (FIM) generation, and extends context lengths to 8K tokens. Variants of StarCoder have been fine-tuned for specific domains such as SQL, OctoPack, and WizardCoder-style instruction following.

For more information on using our StarCoder implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/starcoder) in our documentation.
