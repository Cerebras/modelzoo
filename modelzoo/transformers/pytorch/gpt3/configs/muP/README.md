# GPT-3 &mu;P configs

[&mu;P](https://arxiv.org/abs/2203.03466) allows for hyperparameter transfer from a smaller base model to large scale language models without the need of tuning at the large scale. For further details refer to the [GPT-3 documentation](../../../README.md) (Section: *Maximal Update Parameterization*).
Here we provide three [GPT-3](https://arxiv.org/abs/2005.14165) &mu;P configs which serve as an example of &mu;-Transfer of hyperparameters. 

## muP hyperparameter search
We considered [40 Million](./params_gpt3_40m.yaml) GPT-3 model as the base model which is used for hyperparameter sweep. This model was trained on [Pile](https://arxiv.org/abs/2101.00027) dataset for 800 Million tokens with the sequence length of 2048. We performed a random search for three hyperparameters which consituted approximately `200` samples or combinations of hyperparameters. The three hyperparameters that were tuned in the search were:

|||
|-------|-------|
|$\eta_{base}$|Base learning rate|
|$\sigma_{base}$|Base initialization standard deviation|
|$m_{emb}$| Embedding output multiplier|

## mu-Transfer to larger models
The optimal values for $\sigma_{base}$, $\eta_{base}$ and $m_{emb}$ were used for &mu;-Transfer to [111 Million](./params_gpt3_111m.yaml) and [256 Million](./params_gpt3_256m.yaml) parameter models. 


## Appendix

**Reference**: Brown, T.B. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).

**Reference**: Yang, G. et al. (2022). [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466).

**Reference**: Dey, N. et al. (2023). [Cerebras-GPT: Open Compute-Optimal Language Models
Trained on the Cerebras Wafer-Scale Cluster](https://arxiv.org/abs/2304.03208).
