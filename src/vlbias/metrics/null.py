"""Chance level and significance for the fairness metrics.

Idea: a model with NO preference ranks the gallery in random order. Drawing many random
rankings of the same gallery gives, for every metric, the distribution of values that can
occur by chance alone (the *null distribution*). An observed value is significant when it
is unlikely under that distribution.

Under random ranking, the null distribution depends only on the gallery labels, the
reference distribution and k, not on the query. It is computed once per
attribute x reference x k and reused for every query.

p-values use the standard permutation-test estimate (1 + #more extreme) / (1 + n), and
many tests are corrected with Benjamini-Hochberg (false discovery rate).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Metrics for many rankings at once
# --------------------------------------------------------------------------- #
def ranking_metrics_batch(rankings: np.ndarray, desired: np.ndarray, ks: Sequence[int],
                          zero_floor: float = 0.5) -> dict[int, dict[str, np.ndarray]]:
    """Skew per group, MaxSkew, MinSkew and NDKL for many rankings (vectorised).

    rankings : int array (n_rankings, K) of group labels in ranked order, K >= max(ks)
    returns  : {k: {"skew": (n, G), "max_skew": (n,), "min_skew": (n,), "ndkl": (n,)}}
    Same definitions as ``vlbias.metrics.fairness`` (tested to agree).
    """
    rankings = np.asarray(rankings, dtype=int)
    desired = np.asarray(desired, dtype=float)
    n, K = rankings.shape
    G = len(desired)
    if K < max(ks):
        raise ValueError(f"rankings have {K} items, fewer than k={max(ks)}")

    counts = np.zeros((n, K, G))
    counts[np.arange(n)[:, None], np.arange(K)[None, :], rankings] = 1.0
    counts = np.cumsum(counts, axis=1)                       # counts in top-1..top-K
    shares = counts / np.arange(1, K + 1)[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(shares > 0, shares * np.log(shares / desired), 0.0).sum(axis=2)  # (n, K)
    weights = 1.0 / np.log2(np.arange(1, K + 1) + 1)

    out = {}
    for k in ks:
        p = shares[:, k - 1, :]
        p = np.where(p == 0, zero_floor / k, p)
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.log(p / desired)
        skew[:, desired == 0] = np.nan
        out[k] = {
            "skew": skew,
            "max_skew": np.nanmax(skew, axis=1),
            "min_skew": np.nanmin(skew, axis=1),
            "ndkl": (kl[:, :k] * weights[:k]).sum(axis=1) / weights[:k].sum(),
        }
    return out


def chance_distributions(labels: np.ndarray, desired_by_ref: dict[str, np.ndarray],
                         ks: Sequence[int], n_null: int = 1000, seed: int = 0,
                         zero_floor: float = 0.5, chunk: int = 200) -> dict[tuple[str, int], dict]:
    """Null distributions of all metrics under random rankings of a gallery.

    labels         : group label of every gallery image (only valid labels, no -1)
    desired_by_ref : {"uniform": array, "gallery": array, ...}
    returns        : {(reference, k): {"skew": (n_null, G), "max_skew": ..., "min_skew": ..., "ndkl": ...}}
    """
    labels = np.asarray(labels, dtype=int)
    K = min(max(ks), len(labels))
    rng = np.random.default_rng(seed)
    parts: dict[tuple[str, int], list[dict]] = {}
    done = 0
    while done < n_null:
        m = min(chunk, n_null - done)
        idx = np.stack([rng.choice(len(labels), K, replace=False) for _ in range(m)])
        rankings = labels[idx]
        for ref, desired in desired_by_ref.items():
            for k, res in ranking_metrics_batch(rankings, desired, ks, zero_floor).items():
                parts.setdefault((ref, k), []).append(res)
        done += m
    return {key: {m: np.concatenate([p[m] for p in ps]) for m in ps[0]} for key, ps in parts.items()}


# --------------------------------------------------------------------------- #
# p-values
# --------------------------------------------------------------------------- #
def p_greater(observed, null) -> np.ndarray:
    """P(chance value >= observed): small = observed is unusually LARGE."""
    s = np.sort(np.asarray(null))
    obs = np.atleast_1d(np.asarray(observed, dtype=float))
    n_ge = len(s) - np.searchsorted(s, obs - _TOL, side="left")
    return (1 + n_ge) / (1 + len(s))


def p_less(observed, null) -> np.ndarray:
    """P(chance value <= observed): small = observed is unusually SMALL."""
    s = np.sort(np.asarray(null))
    obs = np.atleast_1d(np.asarray(observed, dtype=float))
    n_le = np.searchsorted(s, obs + _TOL, side="right")
    return (1 + n_le) / (1 + len(s))


def p_two_sided(observed, null) -> np.ndarray:
    """Unusually large OR small (e.g. a group's Skew: over- or under-represented)."""
    return np.minimum(1.0, 2 * np.minimum(p_less(observed, null), p_greater(observed, null)))


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls the false discovery rate).

    Testing hundreds of query x group cells at alpha = 0.05 would flag ~5% of them by
    chance alone; the adjusted p-values keep the expected share of false findings at alpha.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / np.arange(1, len(q) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(q)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def null_of_mean(null: np.ndarray, n_items: int, n_boot: int = 4000,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Chance distribution of the MEAN of ``n_items`` independent random-ranking values.

    Used when a result averages several queries (e.g. one concept over 8 templates):
    the average of random values varies less than a single one, so its chance range is
    narrower.
    """
    rng = rng or np.random.default_rng(0)
    null = np.asarray(null)
    idx = rng.integers(0, len(null), size=(n_boot, n_items))
    return null[idx].mean(axis=1)
