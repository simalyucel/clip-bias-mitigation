"""Fairness metrics for ranked retrieval results.

All functions take ``ranked_labels``: the protected-attribute label (an integer group id
in ``0 .. n_groups-1``) of each retrieved image, in ranked order (position 0 = most
similar to the query). Only the first ``k`` entries are used.

Definitions follow Geyik et al. (2019), as used for text-to-image retrieval by
Berg et al. (2022) and Chuang et al. (2023):

    Skew_a@k  = ln( p_a(top-k) / p_a(desired) )
    MaxSkew@k = max_a Skew_a@k      largest over-representation   (0 is ideal)
    MinSkew@k = min_a Skew_a@k      largest under-representation  (0 is ideal)
    NDKL@k    = (1/Z) * sum_{i=1..k} 1/log2(i+1) * KL( D(top-i) || D(desired) )
                with Z = sum_{i=1..k} 1/log2(i+1)

Notes
-----
* Skew is unbounded below: if a group has no image in the top-k its Skew is -inf.
  To keep averages finite, a missing group's share is floored at ``zero_floor / k``
  (default: half an image). The number of absent groups is reported separately so
  this choice stays visible.
* NDKL is >= 0 and weights early positions more; it cannot tell *which* group is
  favoured, so it complements Skew rather than replacing it. Geyik et al. define it
  over the whole ranking; here it is computed over the top-k for the same k as Skew.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

REFERENCES = ("uniform", "gallery", "custom")


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def group_proportions(labels: Sequence[int], n_groups: int) -> np.ndarray:
    """Share of each group among ``labels`` (length ``n_groups``, sums to 1)."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        raise ValueError("labels is empty")
    counts = np.bincount(labels, minlength=n_groups)[:n_groups]
    return counts / labels.size


def reference_distribution(
    kind: str,
    n_groups: int,
    gallery_labels: Sequence[int] | None = None,
    custom: Sequence[float] | None = None,
) -> np.ndarray:
    """The *desired* distribution the top-k is compared against.

    uniform : 1/n_groups for every group (demographic parity)
    gallery : composition of the collection being searched (isolates the model
              from an imbalanced gallery, cf. Kong et al. 2023)
    custom  : any user-given distribution, e.g. real-world statistics
    """
    if kind == "uniform":
        return np.full(n_groups, 1.0 / n_groups)
    if kind == "gallery":
        if gallery_labels is None:
            raise ValueError("gallery reference needs gallery_labels")
        return group_proportions(gallery_labels, n_groups)
    if kind == "custom":
        if custom is None:
            raise ValueError("custom reference needs custom=")
        d = np.asarray(custom, dtype=float)
        if d.shape != (n_groups,) or np.any(d < 0) or not np.isclose(d.sum(), 1.0):
            raise ValueError("custom distribution must be non-negative, length n_groups, sum to 1")
        return d
    raise ValueError(f"unknown reference {kind!r}, expected one of {REFERENCES}")


# --------------------------------------------------------------------------- #
# Skew family
# --------------------------------------------------------------------------- #
def _check_k(ranked_labels: Sequence[int], k: int) -> np.ndarray:
    ranked = np.asarray(ranked_labels, dtype=int)
    if k < 1:
        raise ValueError("k must be >= 1")
    if ranked.size < k:
        raise ValueError(f"ranking has {ranked.size} items, fewer than k={k}")
    return ranked[:k]


def skew_at_k(
    ranked_labels: Sequence[int],
    desired: np.ndarray,
    k: int,
    zero_floor: float = 0.5,
) -> np.ndarray:
    """Skew@k for every group (array of length ``len(desired)``).

    Groups with a desired share of 0 get ``nan`` (the ratio is undefined).
    """
    desired = np.asarray(desired, dtype=float)
    top = _check_k(ranked_labels, k)
    p = group_proportions(top, len(desired))
    p = np.where(p == 0, zero_floor / k, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.log(p / desired)
    skew[desired == 0] = np.nan
    return skew


def max_skew_at_k(ranked_labels, desired, k, zero_floor: float = 0.5) -> float:
    return float(np.nanmax(skew_at_k(ranked_labels, desired, k, zero_floor)))


def min_skew_at_k(ranked_labels, desired, k, zero_floor: float = 0.5) -> float:
    return float(np.nanmin(skew_at_k(ranked_labels, desired, k, zero_floor)))


# --------------------------------------------------------------------------- #
# NDKL
# --------------------------------------------------------------------------- #
def ndkl(ranked_labels: Sequence[int], desired: np.ndarray, k: int | None = None) -> float:
    """Normalized Discounted cumulative KL-divergence over the top-k (k=None: full ranking)."""
    desired = np.asarray(desired, dtype=float)
    ranked = np.asarray(ranked_labels, dtype=int)
    k = ranked.size if k is None else k
    top = _check_k(ranked, k)
    n_groups = len(desired)

    one_hot = np.zeros((k, n_groups))
    one_hot[np.arange(k), top] = 1.0
    cum = np.cumsum(one_hot, axis=0) / np.arange(1, k + 1)[:, None]  # D(top-i), i=1..k

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(cum > 0, cum * np.log(cum / desired), 0.0)
    kl = terms.sum(axis=1)  # inf if a group with desired share 0 appears

    weights = 1.0 / np.log2(np.arange(1, k + 1) + 1)
    return float((weights * kl).sum() / weights.sum())


# --------------------------------------------------------------------------- #
# Convenience: all metrics for one ranking
# --------------------------------------------------------------------------- #
def evaluate_ranking(
    ranked_labels: Sequence[int],
    desired: np.ndarray,
    ks: Iterable[int],
    zero_floor: float = 0.5,
) -> list[dict]:
    """All metrics for one ranked list, one dict per k.

    Each dict has: k, max_skew, min_skew, max_abs_skew, ndkl, n_absent, skew (per group).
    ``max_abs_skew`` = max(|Skew|), the variant used by Zhao et al. (2026).
    """
    rows = []
    for k in ks:
        skew = skew_at_k(ranked_labels, desired, k, zero_floor)
        top = _check_k(ranked_labels, k)
        present = np.bincount(top, minlength=len(desired))[: len(desired)] > 0
        rows.append(
            {
                "k": k,
                "max_skew": float(np.nanmax(skew)),
                "min_skew": float(np.nanmin(skew)),
                "max_abs_skew": float(np.nanmax(np.abs(skew))),
                "ndkl": ndkl(ranked_labels, desired, k),
                "n_absent": int(np.sum(~present & (np.asarray(desired) > 0))),
                "skew": skew,
            }
        )
    return rows
