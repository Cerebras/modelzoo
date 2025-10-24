import os
import numpy as np
import torch
from cerebras.modelzoo.trainer.callbacks import Callback

class SaveOldLogProbs(Callback):
    """
        Saves model outputs to out_dir as old_log_probs when in calc-old mode.
        This saves the log-probs, per batch.
    """

    def __init__(self, prefix: str = "oldlp"):
        self.out_dir = os.getcwd()
        self.prefix = prefix
        self._count = 0

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        os.makedirs(self.out_dir, exist_ok=True)
        self._ready = True

    def on_after_forward(self, trainer, model, outputs, batch):
        if batch is not None and isinstance(batch, dict) and ("old_log_probs" in batch):
            # Training step -- can be ignored.
            return  

        if not isinstance(outputs, torch.Tensor):
            # When we're calculating old log probs, we should get a tensor.
            return 

        # Save tensor as old_log_probs shard
        arr = outputs.detach().cpu().numpy()
        path = os.path.join(self.out_dir, f"{self.prefix}_{self._count:07d}.npz")
        np.savez(path, old_log_probs=arr)
        self._count += 1
