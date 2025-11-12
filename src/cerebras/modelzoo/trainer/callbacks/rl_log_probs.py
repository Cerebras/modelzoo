import os
import numpy as np
import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback
import torch.nn.functional as F
import torch

def logprobs_from_logits(logits: torch.Tensor, labels) -> torch.Tensor:
    """
    Implementation taken from verL; modified to fix compile issues on our stack.
    """
    logp = F.log_softmax(logits, dim=-1) # batch_Size x 3840 X vocab_size
    one_hot = cstorch.nn.functional.one_hot(
        labels.to(torch.int64), num_classes=logp.size(-1)
    ).to(logp.dtype) # batch_size x 3840 x vocab_size
    return (logp * one_hot).sum(dim=-1)


class SaveOldLogProbs(Callback):
    """
        Saves model outputs to out_dir as old_log_probs when in calc-old mode.
        This saves the log-probs, per batch.
    """

    def __init__(self, prefix: str = "oldlp"):
        self.out_dir = "/n0/lab/sota-rl-inference/eval_rollouts"
        self.prefix = prefix
        self._count = 0

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        os.makedirs(self.out_dir, exist_ok=True)
        self._ready = True

    def on_after_forward(self, trainer, model, outputs, batch):
        self.post_process(outputs)

    @cstorch.step_closure
    def post_process(self, outputs):
        # Save tensor as old_log_probs shard
        #arr = outputs['output'].numpy()
        path = os.path.join(self.out_dir, f"{self.prefix}_{self._count:07d}.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        #B, MSL, V = outputs['logits'].shape
        #response_length = 3840

        #batch_idx = torch.arange(B, dtype=torch.int).unsqueeze(1)  # [B, 1]

        # For each batch, compute the token indices to extract
        '''token_idx = (
            torch.arange(response_length, dtype=torch.int).unsqueeze(0)  # [1, 3840]
            + (outputs["prompts_len"].unsqueeze(1) - 1)  # shift start position per batch
        )  '''    # shape [B, 3840]

        # Gather logits using advanced indexing
        #selected_logits = outputs['logits'][batch_idx, token_idx, :]  # [B, 3840, vocab_size]

        #logits = logits[:, prompt_len:(prompt_len+response_length), :]  # [batch_size, response_length, vocab_size]
        #old_log_probs = logprobs_from_logits(selected_logits, outputs["responses"])
        #return {"old_log_probs": old_log_probs, "input_ids": data["input_ids"], "attention_mask" : data["attention_mask"], "responses" : data["responses"], "prompts_len" : data["prompts_len"]}

        np.savez(path, old_log_probs=outputs['old_log_probs'].numpy(), inputs=outputs['input_ids'].numpy(), mask=outputs['attention_mask'].numpy(), responses=outputs['responses'], promptlen=outputs['prompts_len'])
        self._count += 1
