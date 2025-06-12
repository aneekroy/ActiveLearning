"""Sentence embedding utilities using SentenceTransformer."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

_sbert_model = None


def _load_model() -> SentenceTransformer:
    """Lazily load the sentence transformer to avoid unnecessary downloads."""
    global _sbert_model
    if _sbert_model is None:
        model_name = os.environ.get("SBERT_MODEL", "/home/aneek/models/all-MiniLM-L6-v2")
        local = Path(model_name).exists()
        _sbert_model = SentenceTransformer(model_name, local_files_only=local)
    return _sbert_model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _load_model()
    emb = model.encode(texts, normalize_embeddings=True)
    return np.array(emb)
