from copy import deepcopy
from typing import Literal, Optional, Union
from pydantic import Field

import torch
import torch.nn.functional as F

import cerebras.pytorch as cstorch
from cerebras.modelzoo.losses.GRPOLoss import GRPOLoss
from cerebras.modelzoo.models.nlp.llama.model import LlamaModelConfig
from cerebras.modelzoo.models.nlp.gpt2.model import GPT2ModelConfig
from cerebras.modelzoo.config import ModelConfig
from typing_extensions import Annotated
import logging

def logprobs_from_logits(logits: torch.Tensor, labels) -> torch.Tensor:
    """
    Implementation taken from verL; modified to fix compile issues on our stack.
    """
    logp = F.log_softmax(logits, dim=-1) # batch_Size x 3840 X vocab_size
    one_hot = cstorch.nn.functional.one_hot(
        labels.to(torch.int64), num_classes=logp.size(-1)
    ).to(logp.dtype) # batch_size x 3840 x vocab_size
    return (logp * one_hot).sum(dim=-1)


class RLModelConfig(ModelConfig):
    name: Literal["RL"]
    policy_model : Annotated[Union[LlamaModelConfig, GPT2ModelConfig], Field(discriminator="name"),] = ...

    use_kl_loss : Optional[bool] = True
    kl_loss_coef : Optional[float] = 0.005
    clip_ratio : Optional[float] = None
    clip_ratio_low : Optional[float] = 0.2
    clip_ratio_high : Optional[float] = 0.28

class RLModel(torch.nn.Module):
    def __init__(self, config: RLModelConfig):
        super().__init__()

        self.policy_model = config.policy_model()

        self.loss_fn = GRPOLoss(config.clip_ratio, config.clip_ratio_low, config.clip_ratio_high, config.use_kl_loss, config.kl_loss_coef)

    def forward(self, data):
        if "old_log_probs" not in data:
            logging.info("Rahul inside old log probs")
            _, logits = self.policy_model(
                data={
                    "input_ids": data["input_ids"],
                    "attention_mask": data["attention_mask"],
                    "labels": data["input_ids"],
                    #"position_ids": data["position_ids"],
                },
                output_logits=True,
            )

            # TODO: We don't divide by temperature here. veRL does something like logits.div_(temperature),
            # where temperature is present in data.meta_info.
            logging.info(f"rahul logits shape is {logits.shape}")
            response_length = data["responses"].size(-1)
            logging.info(f"rahul response len is {response_length}")

            B, MSL, V = logits.shape

            # Create batch indices [0, 1, ..., B-1]
            batch_idx = torch.arange(B, dtype=torch.int, device=logits.device).unsqueeze(1)  # [B, 1]
            # For each batch, compute the token indices to extract
            token_idx = (
                torch.arange(response_length, dtype=torch.int, device=logits.device).unsqueeze(0)  # [1, 3840]
                + (data["prompts_len"].unsqueeze(1) - 1)  # shift start position per batch
            )      # shape [B, 3840]
            # Gather logits using advanced indexing
            selected_logits = logits[batch_idx, token_idx, :]  # [B, 3840, vocab_size]
            old_log_probs = logprobs_from_logits(selected_logits, data["responses"])
            return {"old_log_probs": old_log_probs, "input_ids": data["input_ids"], "attention_mask" : data["attention_mask"], "responses" : data["responses"], "prompts_len" : data["prompts_len"]}

            #logits = logits[:, 165:(165+response_length), :]  # [batch_size, response_length, vocab_size]
            #old_log_probs = logprobs_from_logits(logits, data["responses"])
            #return {"logits": logits.to(torch.float32), "input_ids": data["input_ids"], "attention_mask" : data["attention_mask"], "responses" : data["responses"], "prompts_len" : data["prompts_len"]}

        logging.info("Rahul executing training")
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
