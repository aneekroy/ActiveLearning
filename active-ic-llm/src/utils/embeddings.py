"""Sentence embedding utilities using :mod:`sentence_transformers`."""

from typing import List
import numpy as np
from pathlib import Path
import os

# Ensure transformers does not pull in TensorFlow/Flax which may
# cause verbose GPU initialisation warnings when imported.
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORT", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Import SentenceTransformer lazily to avoid loading heavy backends when the
# embedding utilities are not used.
SentenceTransformer = None

_sbert_model = None


def _load_model() -> SentenceTransformer:
    """Lazily load :class:`SentenceTransformer` to avoid unnecessary downloads."""
    global _sbert_model, SentenceTransformer
    if _sbert_model is None:
        if SentenceTransformer is None:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
        model_name = os.environ.get("SBERT_MODEL", "/home/aneek/models/all-MiniLM-L6-v2")
        local = Path(model_name).exists()
        _sbert_model = SentenceTransformer(model_name, local_files_only=local)
    return _sbert_model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _load_model()
    emb = model.encode(texts, normalize_embeddings=True)
    return np.array(emb)
