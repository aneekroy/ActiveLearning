"""KMeans clustering helper."""

from typing import List
import numpy as np
from sklearn.cluster import KMeans


def kmeans_cluster_pool(embeddings: np.ndarray, k: int, random_state: int = 42) -> List[int]:
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(embeddings)
    centroids = kmeans.cluster_centers_
    indices = []
    for c in centroids:
        dists = np.linalg.norm(embeddings - c, axis=1)
        indices.append(int(dists.argmin()))
    return indices
