from copy import deepcopy
from typing import Literal, Optional, Union

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.losses.GRPOLoss import GRPOLoss
from cerebras.modelzoo.models.nlp.llama.model import LlamaModelConfig
from cerebras.modelzoo.models.nlp.gpt2.model import GPT2ModelConfig
from cerebras.modelzoo.config import ModelConfig
from typing_extensions import Annotated


class RLModelConfig(ModelConfig):
    name: Literal["RL"]
    policy_model : Annotated[Union[LlamaModelConfig, GPT2ModelConfig], Field(discriminator="name"),] = ...

    use_kl_loss : Optional[bool] = True
    kl_loss_coef : Optional[float] = 0.005
    clip_ratio : Optional[float]
    clip_ratio_low : Optional[float] = 0.2
    clip_ratio_high : Optional[float] = 0.28

class RLModel(torch.nn.Module):
    def __init__(self, config: RLModelConfig):
        super().__init__()

        self.policy_model = config.policy_model()

        self.loss_fn = GRPOLoss(config.clip_ratio, config.clip_ratio_low, config.clip_ratio_high, config.use_kl_loss, config.kl_loss_coef)

    def forward(self, data):
        _, curr_log_probs = self.policy_model(
            data={
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "labels": data["input_ids"],
                "position_ids" : data["position_ids"]
            },
            output_logits=True,
        )

        curr_log_probs = torch.log_softmax(curr_log_probs, dim=-1)
        # curr_log_probs = torch.gather(
        #    input=curr_log_probs,
        #    dim=-1,
        #    index=data["input_ids"].to(torch.int64).unsqueeze(-1),
        # ).squeeze(-1)
        one_hot = cstorch.nn.functional.one_hot(
            data["input_ids"].to(torch.int64),
            num_classes=curr_log_probs.size(-1),
        )  # .to(curr_log.probs.dtype)  # (8,128,50257)
        # Multiply and sum over vocab dimension
        curr_log_probs = torch.sum(curr_log_probs * one_hot, dim=-1)  # (8,128)
        loss = self.loss_fn(
            data["old_log_probs"],
            curr_log_probs,
            data["advantages"],
            data["loss_mask"],
            data["ref_log_probs"],
        )
        return loss
