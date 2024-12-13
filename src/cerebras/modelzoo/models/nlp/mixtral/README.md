# Mixtral Language Models

## Overview of the model

[Mixtral of Experts](https://arxiv.org/abs/2401.04088) models (8x7B and 8x22B) have similar architecture to [Mistral](../mistral) model, except that they use Mixture of Expert (MoE) instead of regular FFNs. Each MoE block has 8 experts and each token is routed to top-2 expert FFNs.

For more details on each technique we refer to the original papers in the `References` section.

## Structure of the code

The code for Mixtral uses the same infrastructure as our implementation of [GPT-2](../gpt2/); we refer to the README under GPT-2 for most instructions. The code in this directory contains:

- `configs/`: YAML configuration files.
- `run.py`: Training script. Performs training and validation.

## Configs included for this model

For convenience, we provide two different configurations of common model setups for Mixtral models:

- [params_mixtral_8x7B.yaml](./configs/params_mixtral_8x7b.yaml): A 46.7B parameter model configured as described in the original paper.
- [params_mixtral_8x22B.yaml](./configs/params_mixtral_8x22b.yaml): Larger Mixtral model (141B parameters), announced in [Cheaper, Better, Faster, Stronger](https://mistral.ai/news/mixtral-8x22b/), the arXiv paper for this configuration is not published yet.
- [params_moe_111M_base_cszoov2.yaml](./configs/params_moe_111M_base_cszoov2.yaml): This is a 111M config developed by internal team in Cerebras
- [params_moe_111M_with_shared_expert_cszoov2.yaml](./configs/params_moe_111M_with_shared_expert_cszoov2.yaml): This config is to demonstrate how shared experts can be used in a MoE model.

## Expert Configuration Details

These YAML settings allow flexibility in training with different numbers of experts and levels of specialization.

`num_experts`: Defines the total number of experts in the model.

`top_k`: Specifies how many experts are selected for each token during routing.

## Additional Features for Enhanced MoE Training

Our models include extra features built on top of the Mixtral base model to improve Mixture of Experts (MoE) training. These features can be configured in the YAML file:

`num_shared_experts` (Optional[int]):
Specifies the number of experts shared across all tokens. These shared experts are always activated and help capture common knowledge across different contexts. This concept is inspired by [DeepseekMoe](https://arxiv.org/pdf/2401.06066).

`null_expert_bias` (Optional[float]):
Adds an optional bias to the "null expert" probability in the routing process, which improves loss when top_k=1. The null expert represents the model’s uncertainty or its decision that "none of the above" is the best option. This bias enhances gradient flow back to the router, leading to better performance.

`routing_algorithm` (Literal["hash", "learned"]):
Allows users to choose between hash-based routing and learned routing methods for determining which experts to activate.

`router_selection_nonlinearity` (Literal["sigmoid", "sinkhorn", "softmax"]):
Specifies the type of non-linearity used in the routing algorithm to generate expert probabilities. This option is applicable when using the "learned" routing method.

## Appendix

**Reference**: Jiang, Albert, et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825)

**Reference**: Jiang, Albert, et al. (2024). [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
