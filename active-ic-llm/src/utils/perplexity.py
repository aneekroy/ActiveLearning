"""Batch perplexity computation."""

from typing import List

from ..models.model_utils import ModelUtils


def compute_perplexities(
    texts: List[str],
    model_name: str,
    device: str,
    num_gpus: int,
    batch_size: int,
) -> List[float]:
    """Compute perplexities for a list of texts using batched inference."""
    mu = ModelUtils(model_name, device=device, num_gpus=num_gpus)
    return mu.compute_batch_perplexity(texts, batch_size=batch_size)
