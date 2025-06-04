"""Similarity-based sampling for each test instance."""

from typing import List
import numpy as np
from ..utils.embeddings import embed_texts


class SimilaritySampler:
    def __init__(self):
        pass

    def select_for_one_test(self, pool_dataset, test_text: str, k: int) -> List[int]:
        pool_emb = embed_texts(pool_dataset.get_all_texts())
        test_emb = embed_texts([test_text])
        sims = np.dot(pool_emb, test_emb.T).squeeze()
        topk = np.argsort(-sims)[:k]
        return topk.tolist()
