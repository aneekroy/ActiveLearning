"""Uncertainty-based sampling using perplexity."""

from typing import List

from ..config import cfg
from ..utils.perplexity import compute_perplexities


class UncertaintySampler:
    def __init__(self):
        self.model_name = cfg.model_name
        self.device = cfg.device
        self.num_gpus = cfg.num_gpus

    def select(self, pool_dataset, k: int) -> List[int]:
        texts = pool_dataset.get_all_texts()
        perplexities = compute_perplexities(texts, self.model_name, self.device, self.num_gpus)
        ranked = sorted(range(len(texts)), key=lambda i: perplexities[i], reverse=True)
        return ranked[:k]
