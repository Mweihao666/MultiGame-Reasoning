import random
import numpy as np
from contextlib import contextmanager
from omegaconf import OmegaConf
import dataclasses
from typing import Callable, Optional

import torch
import torch.nn.functional as F

@contextmanager
def all_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)

def register_resolvers():
    try:
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
        OmegaConf.register_new_resolver("int_div", lambda x, y: int(float(x) / float(y)))
        OmegaConf.register_new_resolver("not", lambda x: not x)
    except:
        pass # already registered



@dataclasses.dataclass
class GenerationsLogger:

    def log(self, loggers, samples, step, _type='val'):
        if 'wandb' in loggers:
            self.log_generations_to_wandb(samples, step, _type)
        if 'swanlab' in loggers:
            self.log_generations_to_swanlab(samples, step, _type)

    def log_generations_to_wandb(self, samples, step, _type='val'):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'table'):
            # Initialize the table on first call
            self.table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"{_type}/generations": new_table}, step=step)
        self.table = new_table

    def log_generations_to_swanlab(self, samples, step, _type='val'):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i+1}"))

        # Log to swanlab
        swanlab.log({f"{_type}/generations": swanlab_text_list}, step=step)




def _gather_from_labels(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities that correspond to the provided labels."""

    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _stable_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Compute log-softmax in float32 with a max-subtraction trick for stability."""

    logits_float = logits.float()
    shifted = logits_float - logits_float.max(dim=-1, keepdim=True).values
    return shifted - shifted.logsumexp(dim=-1, keepdim=True)


def safe_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logprobs_fn: Optional[Callable[..., torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Compute log-probs while guarding against NaNs from fused kernels.

    The function first tries the provided ``logprobs_fn`` (typically the high-
    performance flash-attn implementation). If the result contains non-finite
    values, it falls back to a numerically stable float32 computation based on
    ``torch.log_softmax`` and replaces only the problematic positions.
    """

    log_probs = None
    if logprobs_fn is not None:
        try:
            log_probs = logprobs_fn(logits=logits, labels=labels, **kwargs)
        except TypeError:
            # Some implementations expect positional arguments
            log_probs = logprobs_fn(logits, labels, **kwargs)

    if log_probs is None:
        log_probs = _gather_from_labels(F.log_softmax(logits.float(), dim=-1), labels)

    if torch.isfinite(log_probs).all():
        return log_probs

    fallback_log_probs = _gather_from_labels(_stable_log_softmax(logits), labels)
    repaired_log_probs = torch.where(torch.isfinite(log_probs), log_probs, fallback_log_probs)
    return torch.nan_to_num(repaired_log_probs, neginf=-float("inf"))