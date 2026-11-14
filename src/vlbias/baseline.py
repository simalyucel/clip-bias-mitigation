"""Bias evaluation: rank a labelled gallery for every query, compute metrics + significance.

The core (``compute_nulls`` + ``evaluate``) works on precomputed embeddings, so it is
model-agnostic and reused unchanged by every mitigation method (only the embeddings change).

Labels may contain -1 (e.g. children for "age_adult"): those images are left out of the
ranking for that attribute, i.e. that attribute is evaluated on the sub-gallery of images
that have a valid label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vlbias.data.queries import Query, load_queries
from vlbias.metrics import evaluate_ranking, reference_distribution
from vlbias.metrics.null import (benjamini_hochberg, chance_distributions, p_greater,
                                 p_less, p_two_sided)
from vlbias.retrieval.rank import top_k_indices

QUERY_METRICS = ("max_skew", "min_skew", "max_abs_skew", "ndkl", "n_absent")


def _references(attr_labels, n_groups, references, custom=None) -> dict[str, np.ndarray]:
    refs = {}
    for ref in references:
        if ref == "custom" and custom is None:
            continue
        refs[ref] = reference_distribution(ref, n_groups, gallery_labels=attr_labels, custom=custom)
    return refs


def compute_nulls(labels: dict[str, np.ndarray], label_names: dict[str, list[str]],
                  ks: Sequence[int], references: Sequence[str], n_null: int = 1000,
                  seed: int = 0, zero_floor: float = 0.5,
                  custom_references: dict | None = None) -> dict[tuple[str, str, int], dict]:
    """Chance distributions per (attribute, reference, k) from random rankings."""
    nulls = {}
    for attr, lab in labels.items():
        lab = lab[lab >= 0]
        names = label_names[attr]
        refs = _references(lab, len(names), references, (custom_references or {}).get(attr))
        for (ref, k), d in chance_distributions(lab, refs, ks, n_null, seed, zero_floor).items():
            nulls[(attr, ref, k)] = {**d, "groups": names}
    return nulls


def chance_summary(nulls: dict) -> pd.DataFrame:
    """Mean and 95% chance range of every metric (what a model with no preference scores)."""
    rows = []
    for (attr, ref, k), d in nulls.items():
        for m in ("max_skew", "min_skew", "ndkl"):
            v = d[m]
            rows.append({"attribute": attr, "reference": ref, "k": k, "metric": m,
                         "chance_mean": float(np.mean(v)),
                         "chance_lo": float(np.quantile(v, 0.025)),
                         "chance_hi": float(np.quantile(v, 0.975))})
    return pd.DataFrame(rows)


def evaluate(
    query_emb: np.ndarray,
    queries: Sequence[Query],
    gallery_emb: np.ndarray,
    labels: dict[str, np.ndarray],
    label_names: dict[str, list[str]],
    ks: Sequence[int] = (100, 1000),
    references: Sequence[str] = ("uniform", "gallery"),
    zero_floor: float = 0.5,
    custom_references: dict[str, Sequence[float]] | None = None,
    nulls: dict | None = None,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_query, per_group) DataFrames.

    per_query : one row per query x attribute x reference x k with
                max_skew, min_skew, max_abs_skew, ndkl, n_absent
                (+ p-values and significance flags when ``nulls`` is given)
    per_group : one row per query x attribute x reference x k x group with skew
                (+ chance range, p-value, significance when ``nulls`` is given)
    """
    full_top = None
    q_rows, g_rows = [], []
    for attr, attr_labels in labels.items():
        names = label_names[attr]
        valid = attr_labels >= 0
        if valid.all():
            if full_top is None:
                full_top = top_k_indices(query_emb, gallery_emb, max(ks))
            top, lab = full_top, attr_labels
        else:  # rank only the images with a valid label for this attribute
            top, lab = top_k_indices(query_emb, gallery_emb[valid], max(ks)), attr_labels[valid]
        refs = _references(lab, len(names), references, (custom_references or {}).get(attr))
        for qi, q in enumerate(queries):
            ranked = lab[top[qi]]
            for ref, desired in refs.items():
                for r in evaluate_ranking(ranked, desired, ks, zero_floor):
                    base = {**q.to_dict(), "attribute": attr, "reference": ref, "k": r["k"]}
                    q_rows.append({**base, **{m: r[m] for m in QUERY_METRICS}})
                    for name, s in zip(names, r["skew"]):
                        g_rows.append({**base, "group": name, "skew": s})
    per_query, per_group = pd.DataFrame(q_rows), pd.DataFrame(g_rows)
    if nulls is not None:
        _add_significance(per_query, per_group, nulls, alpha)
    return per_query, per_group


def _add_significance(per_query: pd.DataFrame, per_group: pd.DataFrame, nulls: dict, alpha: float) -> None:
    """Adds p-values (raw + Benjamini-Hochberg within each attribute x reference x k) in place."""
    for col in ("max_skew_p", "min_skew_p", "ndkl_p"):
        per_query[col] = np.nan
    for (attr, ref, k), idx in per_query.groupby(["attribute", "reference", "k"]).groups.items():
        d = nulls[(attr, ref, k)]
        rows = per_query.loc[idx]
        per_query.loc[idx, "max_skew_p"] = p_greater(rows["max_skew"].values, d["max_skew"])
        per_query.loc[idx, "min_skew_p"] = p_less(rows["min_skew"].values, d["min_skew"])
        per_query.loc[idx, "ndkl_p"] = p_greater(rows["ndkl"].values, d["ndkl"])
        for m in ("max_skew", "min_skew", "ndkl"):
            adj = benjamini_hochberg(per_query.loc[idx, f"{m}_p"].values)
            per_query.loc[idx, f"{m}_p_adj"] = adj
            per_query.loc[idx, f"{m}_sig"] = adj < alpha

    for col in ("chance_lo", "chance_hi", "p_value"):
        per_group[col] = np.nan
    for (attr, ref, k, group), idx in per_group.groupby(["attribute", "reference", "k", "group"]).groups.items():
        d = nulls[(attr, ref, k)]
        null = d["skew"][:, d["groups"].index(group)]
        per_group.loc[idx, "chance_lo"] = np.nanquantile(null, 0.025)
        per_group.loc[idx, "chance_hi"] = np.nanquantile(null, 0.975)
        per_group.loc[idx, "p_value"] = p_two_sided(per_group.loc[idx, "skew"].values, null)
    for _, idx in per_group.groupby(["attribute", "reference", "k"]).groups.items():
        adj = benjamini_hochberg(per_group.loc[idx, "p_value"].values)
        per_group.loc[idx, "p_adj"] = adj
        per_group.loc[idx, "significant"] = adj < alpha
    per_group["significant"] = per_group["significant"].astype(bool)


def summarize(per_query: pd.DataFrame, chance: pd.DataFrame | None = None,
              by: Sequence[str] = ("query_set", "attribute", "reference", "k")) -> pd.DataFrame:
    """Mean (and std) of each metric over queries, + chance level and share significant."""
    metrics = ["max_skew", "min_skew", "max_abs_skew", "ndkl"]
    agg = per_query.groupby(list(by))[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    sig_cols = [c for c in ("max_skew_sig", "min_skew_sig", "ndkl_sig") if c in per_query]
    if sig_cols:
        share = per_query.groupby(list(by))[sig_cols].mean().add_prefix("share_").reset_index()
        agg = agg.merge(share, on=list(by))
    if chance is not None:
        wide = chance.pivot_table(index=["attribute", "reference", "k"], columns="metric",
                                  values="chance_mean").add_suffix("_chance").reset_index()
        agg = agg.merge(wide, on=["attribute", "reference", "k"], how="left")
    return agg


def run_utility(vlm, util_cfg: dict, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Retrieval quality (Recall@K) on the dataset in the ``utility:`` config block."""
    from vlbias.data.flickr30k import load_flickr30k_test
    from vlbias.metrics.utility import retrieval_recall
    from vlbias.retrieval.embed import encode_texts, gallery_embeddings

    if util_cfg.get("dataset", "flickr30k") != "flickr30k":
        raise NotImplementedError(util_cfg["dataset"])
    data = load_flickr30k_test(util_cfg.get("hf_id"))
    img_emb = gallery_embeddings(vlm, data, cache_dir)
    txt_emb = encode_texts(vlm, data.captions)
    return retrieval_recall(txt_emb, img_emb, data.caption_to_image, util_cfg.get("ks", (1, 5, 10)))


def run_baseline(cfg: dict, root: str | Path = ".") -> dict:
    """Full baseline from an experiment config dict (see configs/experiments/)."""
    import yaml

    from vlbias.data.attributes import resolve_attributes
    from vlbias.data.fairface import load_gallery
    from vlbias.models.load import load_model
    from vlbias.retrieval.embed import encode_texts, gallery_embeddings
    from vlbias.utils.config import load_yaml, set_seed

    root = Path(root)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    cache = root / cfg.get("cache_dir", "data/embeddings")

    vlm = load_model(load_yaml(root / cfg["model"]))
    queries = [q for p in cfg["query_sets"] for q in load_queries(root / p)]
    gallery = load_gallery(cfg["dataset"], seed)
    labels, names = resolve_attributes(gallery, cfg["attributes"])

    g_emb = gallery_embeddings(vlm, gallery, cache, batch_size=cfg.get("batch_size", 256))
    q_emb = encode_texts(vlm, [q.text for q in queries])

    m, sig = cfg["metrics"], cfg.get("significance", {})
    nulls = None
    if sig.get("n_null", 0) > 0:
        nulls = compute_nulls(labels, names, m["k"], m["references"], sig["n_null"],
                              sig.get("seed", 0), m.get("zero_floor", 0.5), m.get("custom"))
    per_query, per_group = evaluate(q_emb, queries, g_emb, labels, names, ks=m["k"],
                                    references=m["references"], zero_floor=m.get("zero_floor", 0.5),
                                    custom_references=m.get("custom"), nulls=nulls,
                                    alpha=sig.get("alpha", 0.05))
    for df in (per_query, per_group):
        df.insert(0, "model", vlm.name)
        df.insert(1, "dataset", gallery.name)
    chance = chance_summary(nulls) if nulls else None
    summary = summarize(per_query, chance)

    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(out / "per_query.csv", index=False)
    per_group.to_csv(out / "per_group.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    if chance is not None:
        chance.to_csv(out / "chance.csv", index=False)
    utility = None
    if "utility" in cfg:
        utility = run_utility(vlm, cfg["utility"], cache)
        utility.insert(0, "model", vlm.name)
        utility.to_csv(out / "utility.csv", index=False)
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return {"per_query": per_query, "per_group": per_group, "summary": summary,
            "chance": chance, "nulls": nulls, "utility": utility, "gallery": gallery,
            "labels": labels, "label_names": names, "gallery_emb": g_emb,
            "query_emb": q_emb, "queries": queries, "model": vlm}
