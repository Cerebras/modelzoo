# GPT-3 &mu;P configs

&mu;P allows for hyperparameter transfer from a smaller base model to large scale language models without the need of tuning at the large scale. For further details refer to the [GPT-3 documentation](../../../README.md) (Section: *Maximal Update Parameterization*).
Here we provide three GPT-3 &mu;P configs which serve as an example of &mu;-Transfer of hyperparameters. 

## muP hyperparameter search
We considered [40 Million](./params_gpt3_40M.yaml) GPT-3 model as the base model which is used for hyperparameter sweep. This model was trained on [Pile](https://arxiv.org/abs/2101.00027) dataset for 800 Million tokens with the sequence length of 2048. We performed a random search for three hyperparameters which consituted approximately `200` samples or combinations of hyperparameters. The three hyperparameters that were tuned in the search were:

|||
|-------|-------|
|$\eta_{base}$|Base learning rate|
|$\sigma_{base}$|Base initialization standard deviation|
|$m_{emb}$| Embedding output multiplier|

## mu-Transfer to larger models
The optimal values for $\sigma_{base}$, $\eta_{base}$ and $m_{emb}$ were used for &mu;-Transfer to [111 Million](./params_gpt3_111M.yaml) and [256 Million](./params_gpt3_256M.yaml) parameter models. 
