"""Rank gallery images for each query by cosine similarity."""
from __future__ import annotations

import numpy as np


def top_k_indices(query_emb: np.ndarray, gallery_emb: np.ndarray, k: int,
                  chunk_size: int = 256) -> np.ndarray:
    """Indices of the k most similar gallery images per query, best first.

    Both inputs must be L2-normalised, so the dot product is the cosine similarity.
    Returns an int array of shape (n_queries, k).
    """
    k = min(k, gallery_emb.shape[0])
    out = np.empty((query_emb.shape[0], k), dtype=np.int64)
    for start in range(0, query_emb.shape[0], chunk_size):
        scores = query_emb[start : start + chunk_size] @ gallery_emb.T
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
        out[start : start + chunk_size] = np.take_along_axis(part, order, axis=1)
    return out
