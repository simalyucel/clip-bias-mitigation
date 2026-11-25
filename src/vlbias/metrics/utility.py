"""Retrieval quality: Recall@K in both directions.

text -> image : for each caption, is its image among the K most similar images?
image -> text : for each image, is at least one of its captions among the K most similar captions?

Reported in percent, as in the CLIP and bias-mitigation papers. Higher is better.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def text_to_image_ranks(text_emb: np.ndarray, image_emb: np.ndarray,
                        caption_to_image: np.ndarray) -> np.ndarray:
    """0-based rank of the correct image for every caption (0 = retrieved first)."""
    scores = text_emb @ image_emb.T
    correct = scores[np.arange(len(scores)), caption_to_image]
    return (scores > correct[:, None]).sum(axis=1)


def image_to_text_ranks(text_emb: np.ndarray, image_emb: np.ndarray,
                        caption_to_image: np.ndarray) -> np.ndarray:
    """0-based rank of the best-ranked correct caption for every image."""
    scores = image_emb @ text_emb.T
    ranks = np.empty(len(image_emb), dtype=int)
    for i in range(len(image_emb)):
        best = scores[i, caption_to_image == i].max()
        ranks[i] = int((scores[i] > best).sum())
    return ranks


def retrieval_recall(text_emb: np.ndarray, image_emb: np.ndarray, caption_to_image: np.ndarray,
                     ks: Sequence[int] = (1, 5, 10)) -> pd.DataFrame:
    """Recall@K (%) for text->image and image->text. One row per direction x K."""
    caption_to_image = np.asarray(caption_to_image)
    rows = []
    for direction, ranks in (("text_to_image", text_to_image_ranks(text_emb, image_emb, caption_to_image)),
                             ("image_to_text", image_to_text_ranks(text_emb, image_emb, caption_to_image))):
        for k in ks:
            rows.append({"direction": direction, "k": k, "recall": 100.0 * float(np.mean(ranks < k))})
    return pd.DataFrame(rows)


def recall_from_top_k(top: np.ndarray, caption_to_image: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> list[dict]:
    """text -> image Recall@K (%) from an already ranked top-k list (one row of image ids per caption).

    Used for re-ranking (post-processing) methods, whose order is not given by similarity scores.
    ``top`` must have at least max(ks) columns.
    """
    top, caption_to_image = np.asarray(top), np.asarray(caption_to_image)
    hit_rank = np.full(len(top), np.inf)
    rows_, cols_ = np.nonzero(top == caption_to_image[:, None])
    hit_rank[rows_] = cols_
    return [{"direction": "text_to_image", "k": k, "recall": 100.0 * float(np.mean(hit_rank < k))} for k in ks]
