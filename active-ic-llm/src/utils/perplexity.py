"""Batch perplexity computation."""

from typing import List

from ..models.model_utils import ModelUtils


def compute_perplexities(texts: List[str], model_name: str, device: str, num_gpus: int) -> List[float]:
    mu = ModelUtils(model_name, device=device, num_gpus=num_gpus)
    return [mu.compute_perplexity(t) for t in texts]
