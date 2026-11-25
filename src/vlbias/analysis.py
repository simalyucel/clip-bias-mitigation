"""Per-concept analysis with significance.

Category averages (positive / negative / occupation) can cancel out opposite effects
(e.g. "criminal" -> men, "nasty" -> women). These helpers average over templates only,
one value per concept x group, and test each value against chance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vlbias.metrics.null import benjamini_hochberg, null_of_mean, p_two_sided


def concept_skew(per_group: pd.DataFrame, nulls: dict | None, attribute: str,
                 reference: str = "gallery", k: int = 100, query_set: str | None = None,
                 categories=None, alpha: float = 0.05, n_boot: int = 4000,
                 seed: int = 0) -> pd.DataFrame:
    """Mean Skew per concept x group (over templates), with chance range and significance.

    Columns: category, concept, group, skew (mean over templates), n_templates,
             chance_lo, chance_hi (95% range of the mean of n_templates random rankings),
             p_value, p_adj (Benjamini-Hochberg over all cells returned), significant.
    """
    d = per_group.query("attribute == @attribute and reference == @reference and k == @k")
    if query_set is not None:
        d = d[d["query_set"] == query_set]
    if categories is not None:
        d = d[d["category"].isin(list(categories))]
    out = (d.groupby(["category", "concept", "group"], sort=False)["skew"]
             .agg(skew="mean", n_templates="count").reset_index())
    if nulls is None:
        return out

    null = nulls[(attribute, reference, k)]
    rng = np.random.default_rng(seed)
    cache: dict[tuple[str, int], np.ndarray] = {}
    lo, hi, p = [], [], []
    for _, r in out.iterrows():
        key = (r["group"], int(r["n_templates"]))
        if key not in cache:
            col = null["skew"][:, null["groups"].index(r["group"])]
            cache[key] = null_of_mean(col, key[1], n_boot, rng)
        dist = cache[key]
        lo.append(np.quantile(dist, 0.025))
        hi.append(np.quantile(dist, 0.975))
        p.append(float(p_two_sided(r["skew"], dist)[0]))
    out["chance_lo"], out["chance_hi"], out["p_value"] = lo, hi, p
    out["p_adj"] = benjamini_hochberg(out["p_value"].values)
    out["significant"] = out["p_adj"] < alpha
    return out


def to_matrix(cs: pd.DataFrame, groups: list[str], value: str = "skew") -> pd.DataFrame:
    """Concept x group matrix (rows ordered by category, then as they appear)."""
    m = cs.pivot_table(index=["category", "concept"], columns="group", values=value,
                       aggfunc="first", sort=False)
    return m[[g for g in groups if g in m.columns]]
