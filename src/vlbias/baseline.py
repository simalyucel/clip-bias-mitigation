"""Baseline bias evaluation: rank a labelled gallery for every query, compute metrics.

The core, ``evaluate``, works on precomputed embeddings so it is model-agnostic and
reusable for every mitigation method (only the embeddings change).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vlbias.data.queries import Query, load_queries
from vlbias.metrics import evaluate_ranking, reference_distribution
from vlbias.retrieval.rank import top_k_indices


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_query, per_group) DataFrames.

    per_query : one row per query x attribute x reference x k with
                max_skew, min_skew, max_abs_skew, ndkl, n_absent
    per_group : one row per query x attribute x reference x k x group with skew
    """
    top = top_k_indices(query_emb, gallery_emb, max(ks))
    q_rows, g_rows = [], []
    for attr, attr_labels in labels.items():
        names = label_names[attr]
        refs = {}
        for ref in references:
            custom = (custom_references or {}).get(attr) if ref == "custom" else None
            if ref == "custom" and custom is None:
                continue
            refs[ref] = reference_distribution(ref, len(names), gallery_labels=attr_labels, custom=custom)
        for qi, q in enumerate(queries):
            ranked = attr_labels[top[qi]]
            for ref, desired in refs.items():
                for r in evaluate_ranking(ranked, desired, ks, zero_floor):
                    base = {**q.to_dict(), "attribute": attr, "reference": ref, "k": r["k"]}
                    q_rows.append({**base, **{m: r[m] for m in
                                  ("max_skew", "min_skew", "max_abs_skew", "ndkl", "n_absent")}})
                    for name, s in zip(names, r["skew"]):
                        g_rows.append({**base, "group": name, "skew": s})
    return pd.DataFrame(q_rows), pd.DataFrame(g_rows)


def summarize(per_query: pd.DataFrame,
              by: Sequence[str] = ("query_set", "attribute", "reference", "k")) -> pd.DataFrame:
    """Mean (and std) of each metric over queries."""
    metrics = ["max_skew", "min_skew", "max_abs_skew", "ndkl"]
    agg = per_query.groupby(list(by))[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def run_baseline(cfg: dict, root: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Full baseline from an experiment config dict (see configs/experiments/)."""
    from vlbias.data.fairface import load_fairface
    from vlbias.models.load import load_model
    from vlbias.retrieval.embed import encode_texts, gallery_embeddings
    from vlbias.utils.config import load_yaml, set_seed

    root = Path(root)
    set_seed(cfg.get("seed", 0))

    vlm = load_model(load_yaml(root / cfg["model"]))
    queries = [q for p in cfg["query_sets"] for q in load_queries(root / p)]

    ds = cfg["dataset"]
    if ds["name"] != "fairface":
        raise NotImplementedError(ds["name"])
    gallery = load_fairface(ds.get("split", "validation"), str(ds.get("padding", "0.25")),
                            ds.get("max_images"), cfg.get("seed", 0))
    labels = {a: gallery.labels[a] for a in cfg["attributes"]}

    g_emb = gallery_embeddings(vlm, gallery, root / cfg.get("cache_dir", "data/embeddings"),
                               batch_size=cfg.get("batch_size", 256))
    q_emb = encode_texts(vlm, [q.text for q in queries])

    m = cfg["metrics"]
    per_query, per_group = evaluate(q_emb, queries, g_emb, labels, gallery.label_names,
                                    ks=m["k"], references=m["references"],
                                    zero_floor=m.get("zero_floor", 0.5),
                                    custom_references=m.get("custom"))
    for df in (per_query, per_group):
        df.insert(0, "model", vlm.name)
        df.insert(1, "dataset", gallery.name)
    summary = summarize(per_query)

    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(out / "per_query.csv", index=False)
    per_group.to_csv(out / "per_group.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    return {"per_query": per_query, "per_group": per_group, "summary": summary,
            "gallery": gallery, "gallery_emb": g_emb, "query_emb": q_emb,
            "queries": queries, "model": vlm}
