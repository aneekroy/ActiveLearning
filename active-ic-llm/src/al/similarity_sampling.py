"""Similarity-based sampling for each test instance."""

from typing import List
import numpy as np
from ..utils.embeddings import embed_texts


class SimilaritySampler:
    """Select demonstrations most similar to a given test instance."""

    def __init__(self):
        # Cache embeddings for datasets so we don't recompute them for every
        # test example which drastically slows down inference.
        self._pool_cache = {}

    def select_for_one_test(self, pool_dataset, test_text: str, k: int) -> List[int]:
        """Return indices of ``k`` pool examples most similar to ``test_text``."""

        dataset_id = id(pool_dataset)
        if dataset_id not in self._pool_cache:
            self._pool_cache[dataset_id] = embed_texts(pool_dataset.get_all_texts())
        pool_emb = self._pool_cache[dataset_id]

        test_emb = embed_texts([test_text])
        sims = np.dot(pool_emb, test_emb.T).squeeze()
        topk = np.argsort(-sims)[:k]
        return topk.tolist()
