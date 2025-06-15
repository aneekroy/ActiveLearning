"""Similarity-based sampling for each test instance."""

from typing import List
import numpy as np
from pathlib import Path
from ..config import cfg
from ..utils.embeddings import embed_texts


class SimilaritySampler:
    """Select demonstrations most similar to a given test instance."""

    def __init__(self):
        # Cache embeddings for datasets so we don't recompute them for every
        # test example which drastically slows down inference.
        self._pool_cache = {}
        self._cache_dir = Path(cfg.embedding_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def select_for_one_test(self, pool_dataset, test_text: str, k: int) -> List[int]:
        """Return indices of ``k`` pool examples most similar to ``test_text``."""

        dataset_id = id(pool_dataset)
        if dataset_id not in self._pool_cache:
            cache_file = self._cache_dir / f"{pool_dataset.task}_{dataset_id}.npy"
            if cache_file.exists():
                emb = np.load(cache_file)
            else:
                emb = embed_texts(pool_dataset.get_all_texts())
                np.save(cache_file, emb)
            self._pool_cache[dataset_id] = emb
        pool_emb = self._pool_cache[dataset_id]

        test_emb = embed_texts([test_text])
        sims = np.dot(pool_emb, test_emb.T).squeeze()
        topk = np.argsort(-sims)[:k]
        return topk.tolist()
