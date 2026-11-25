"""Common evaluation pipeline: every mitigation method is tested under exactly the same conditions.

Fixed for every method (set once, from the experiment config):
    queries, image gallery, attributes, reference distributions (uniform + gallery),
    ranking cut-offs k, chance level / significance, utility dataset (Flickr30K).
Measured for every method:
    fairness  (per_query.csv, per_group.csv, summary.csv)
    utility   (utility.csv: Flickr30K Recall@K)
    cost      (costs.csv: set-up time, transform time, latency per query, GPU memory, added parameters)

Usage::

    pipe = EvaluationPipeline(load_yaml("configs/experiments/pipeline_v1.yaml"))
    base = pipe.run(NoMitigation())          # C0, the unmitigated reference
    res  = pipe.run(MyMethod())              # any Mitigation subclass
    compare([base, res])                     # one table, changes vs the baseline

What changes between methods is only the Mitigation object (see vlbias/mitigation/base.py).
Everything that does not depend on the method (gallery embeddings, chance level) is computed once
and shared, so all methods are compared against the same random-ranking chance level.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vlbias.baseline import chance_summary, compute_nulls, evaluate, summarize
from vlbias.mitigation.base import FitContext, Mitigation, NoMitigation

BIAS_METRICS = ("max_skew_mean", "min_skew_mean", "ndkl_mean")


@dataclass
class RunResult:
    method: str
    category: str
    per_query: pd.DataFrame
    per_group: pd.DataFrame
    summary: pd.DataFrame
    utility: pd.DataFrame | None
    costs: pd.DataFrame
    params: dict = field(default_factory=dict)
    output_dir: Path | None = None


class _Timer:
    def __init__(self):
        self.rows = []

    @contextmanager
    def __call__(self, stage: str):
        t0 = time.perf_counter()
        yield
        self.rows.append({"metric": f"{stage}_seconds", "value": time.perf_counter() - t0, "unit": "s"})

    def add(self, metric, value, unit):
        self.rows.append({"metric": metric, "value": value, "unit": unit})


def _cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


class EvaluationPipeline:
    """Loads everything that is shared by all methods once, then ``run(method)`` for each method."""

    def __init__(self, cfg: dict, root: str | Path = ".", cache_dir: str | Path | None = None,
                 vlm=None, gallery=None, verbose: bool = True):
        from vlbias.data.attributes import resolve_attributes
        from vlbias.data.fairface import load_gallery
        from vlbias.data.queries import load_queries
        from vlbias.models.load import load_model
        from vlbias.utils.config import load_yaml, set_seed

        self.cfg, self.root, self.verbose = cfg, Path(root), verbose
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / cfg.get("cache_dir", "data/embeddings")
        set_seed(cfg.get("seed", 0))

        # vlm / gallery can be passed in (tests, or to reuse what a notebook already loaded)
        self.base_vlm = vlm if vlm is not None else load_model(load_yaml(self.root / cfg["model"]))
        self.queries = [q for p in cfg["query_sets"] for q in load_queries(self.root / p)]
        self.gallery = gallery if gallery is not None else load_gallery(cfg["dataset"], cfg.get("seed", 0))
        self.attributes = list(cfg["attributes"])
        self.labels, self.label_names = resolve_attributes(self.gallery, self.attributes)

        m, sig = cfg["metrics"], cfg.get("significance", {})
        self.ks, self.references = list(m["k"]), list(m["references"])
        self.zero_floor, self.custom = m.get("zero_floor", 0.5), m.get("custom")
        self.alpha = sig.get("alpha", 0.05)
        self.nulls = None
        if sig.get("n_null", 0) > 0:
            self._log(f"chance level: {sig['n_null']} random rankings per attribute ...")
            self.nulls = compute_nulls(self.labels, self.label_names, self.ks, self.references,
                                       sig["n_null"], sig.get("seed", 0), self.zero_floor, self.custom)
        self.chance = chance_summary(self.nulls) if self.nulls else None

        self._emb_cache: dict = {}      # (model name, what) -> embeddings
        self._flickr = None
        self.results: dict[str, RunResult] = {}

    # ------------------------------------------------------------------ helpers
    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _gallery_emb(self, vlm):
        from vlbias.retrieval.embed import gallery_embeddings

        key = (vlm.name, "gallery")
        if key not in self._emb_cache:
            self._emb_cache[key] = gallery_embeddings(vlm, self.gallery, self.cache_dir,
                                                      batch_size=self.cfg.get("batch_size", 256))
        return self._emb_cache[key]

    def _query_emb(self, vlm):
        from vlbias.retrieval.embed import encode_texts

        key = (vlm.name, "queries")
        if key not in self._emb_cache:
            self._emb_cache[key] = encode_texts(vlm, [q.text for q in self.queries])
        return self._emb_cache[key]

    def _fit_data(self, vlm):
        """Labelled embeddings from the fit split (FairFace train by default), for methods that learn from labels."""
        from vlbias.data.attributes import resolve_attributes
        from vlbias.data.fairface import load_gallery
        from vlbias.retrieval.embed import gallery_embeddings

        key = (vlm.name, "fit")
        if key not in self._emb_cache:
            fit_cfg = self.cfg.get("fit_data")
            if fit_cfg is None:
                raise ValueError("this method needs labelled fit data: add a `fit_data:` block to the config")
            self._log(f"loading fit data ({fit_cfg.get('split')}, max {fit_cfg.get('max_images')} images) ...")
            g = load_gallery(fit_cfg, self.cfg.get("seed", 0))
            labels, names = resolve_attributes(g, self.attributes)
            emb = gallery_embeddings(vlm, g, self.cache_dir, batch_size=self.cfg.get("batch_size", 256))
            self._emb_cache[key] = (emb, labels, names)
        return self._emb_cache[key]

    def _flickr_data(self):
        if self._flickr is None:
            from vlbias.data.flickr30k import load_flickr30k_test
            self._flickr = load_flickr30k_test(self.cfg["utility"].get("hf_id"))
        return self._flickr

    def _utility(self, vlm, method: Mitigation) -> pd.DataFrame:
        from vlbias.metrics.utility import recall_from_top_k, retrieval_recall
        from vlbias.retrieval.embed import encode_texts, gallery_embeddings

        util_cfg = self.cfg["utility"]
        ks = list(util_cfg.get("ks", (1, 5, 10)))
        data = self._flickr_data()
        key_i, key_t = (vlm.name, "flickr_img"), (vlm.name, "flickr_txt")
        if key_i not in self._emb_cache:
            self._emb_cache[key_i] = gallery_embeddings(vlm, data, self.cache_dir)
        if key_t not in self._emb_cache:
            self._emb_cache[key_t] = encode_texts(vlm, data.captions)
        img = method.transform_image(self._emb_cache[key_i])
        txt = method.transform_text(self._emb_cache[key_t])
        util = retrieval_recall(txt, img, data.caption_to_image, ks)
        if method.overrides("rank"):  # re-ranking changes the text->image order: recompute it from the ranking
            top = method.rank(txt, img, max(ks))
            t2i = pd.DataFrame(recall_from_top_k(top, data.caption_to_image, ks))
            util = pd.concat([t2i, util[util.direction != "text_to_image"]], ignore_index=True)
        return util

    # ------------------------------------------------------------------ main entry
    def run(self, method: Mitigation | None = None, save: bool = True, utility: bool = True) -> RunResult:
        """Evaluate one method (default: no mitigation) under the common protocol."""
        method = method or NoMitigation()
        timer, cuda = _Timer(), _cuda()
        if cuda:
            cuda.cuda.reset_peak_memory_stats()
        self._log(f"== {method.name} ({method.category})")

        # 1) in-processing: possibly a different model -> new embeddings
        with timer("adapt_model"):
            vlm = method.adapt_model(self.base_vlm)
        if vlm is not self.base_vlm and vlm.name == self.base_vlm.name:
            raise ValueError("an adapted model must get its own `name` (embeddings are cached by name)")
        g_raw, q_raw = self._gallery_emb(vlm), self._query_emb(vlm)

        # 2) set-up (fit). Time spent embedding the fit images is reported separately.
        fit_load = {"s": 0.0}

        def fit_data():
            t0 = time.perf_counter()
            out = self._fit_data(vlm)
            fit_load["s"] += time.perf_counter() - t0
            return out

        from vlbias.retrieval.embed import encode_texts
        ctx = FitContext(vlm=vlm, encode_texts=lambda texts: encode_texts(vlm, list(texts)),
                         attributes=self.attributes, fit_data=fit_data)
        t0 = time.perf_counter()
        method.fit(ctx)
        timer.add("fit_seconds", time.perf_counter() - t0 - fit_load["s"], "s")
        timer.add("fit_data_loading_seconds", fit_load["s"], "s")

        # 3) pre-processing: transform embeddings
        with timer("transform_gallery"):
            g = method.transform_image(g_raw)
        with timer("transform_queries"):
            q = method.transform_text(q_raw)

        # 4) latency: transform + rank every query over the whole gallery (top max k)
        t0 = time.perf_counter()
        method.rank(method.transform_text(q_raw), g, max(self.ks))
        timer.add("latency_per_query_ms", 1000 * (time.perf_counter() - t0) / len(self.queries), "ms")

        # 5) fairness, with the same queries, gallery, references, k and chance level for all methods
        with timer("fairness_eval"):
            per_query, per_group = evaluate(q, self.queries, g, self.labels, self.label_names, ks=self.ks,
                                            references=self.references, zero_floor=self.zero_floor,
                                            custom_references=self.custom, nulls=self.nulls,
                                            alpha=self.alpha, rank_fn=method.rank)
        for df in (per_query, per_group):
            df.insert(0, "method", method.name)
            df.insert(1, "method_category", method.category)
            df.insert(2, "model", vlm.name)
            df.insert(3, "dataset", self.gallery.name)
        summary = summarize(per_query, self.chance)
        summary.insert(0, "method", method.name)

        # 6) utility
        util = None
        if utility and "utility" in self.cfg:
            with timer("utility_eval"):
                util = self._utility(vlm, method)
            util.insert(0, "method", method.name)

        timer.add("extra_parameters", method.extra_parameters(), "count")
        if cuda:
            timer.add("gpu_peak_memory_mb", cuda.cuda.max_memory_allocated() / 2**20, "MB")
        costs = pd.DataFrame(timer.rows)
        costs.insert(0, "method", method.name)

        res = RunResult(method.name, method.category, per_query, per_group, summary, util, costs,
                        method.params())
        if save:
            res.output_dir = self._save(res, vlm)
        self.results[method.name] = res
        return res

    def _save(self, res: RunResult, vlm) -> Path:
        import yaml

        out = self.root / self.cfg["output_dir"] / res.method
        out.mkdir(parents=True, exist_ok=True)
        res.per_query.to_csv(out / "per_query.csv", index=False)
        res.per_group.to_csv(out / "per_group.csv", index=False)
        res.summary.to_csv(out / "summary.csv", index=False)
        res.costs.to_csv(out / "costs.csv", index=False)
        if self.chance is not None:
            self.chance.to_csv(out / "chance.csv", index=False)
        if res.utility is not None:
            res.utility.to_csv(out / "utility.csv", index=False)
        with open(out / "method.yaml", "w") as f:
            yaml.safe_dump({"name": res.method, "category": res.category, "model": vlm.name,
                            "params": _plain(res.params)}, f, sort_keys=False)
        with open(out / "config_used.yaml", "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        return out


def _plain(x):
    """Make params YAML-safe (numpy scalars/arrays -> python)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def load_result(path: str | Path) -> RunResult:
    """Read a saved run back from its folder (results/<experiment>/<method>/)."""
    import yaml

    path = Path(path)
    meta = yaml.safe_load(open(path / "method.yaml"))
    read = lambda f: pd.read_csv(path / f) if (path / f).exists() else None
    return RunResult(meta["name"], meta["category"], read("per_query.csv"), read("per_group.csv"),
                     read("summary.csv"), read("utility.csv"), read("costs.csv"), meta.get("params", {}), path)


def compare(results: Sequence[RunResult], baseline: str = "none", reference: str = "gallery", k: int = 1000,
            query_set: str | None = None, targets: dict | None = None) -> pd.DataFrame:
    """One table: every method x attribute, bias + utility, and the change vs the baseline.

    Bias columns are means over queries (lower is better). ``*_change_%`` is relative to the
    baseline: -20 means 20% less bias. Utility is text->image Recall@K. With ``targets``
    (Deliverable 2, Table 12), two yes/no columns say whether a method meets them:
        bias_target_met    : primary bias metric reduced by at least ``min_bias_reduction``
        utility_target_met : primary utility metric dropped by at most ``max_utility_drop``
    """
    results = list(results)
    names = [r.method for r in results]
    if baseline not in names:
        raise ValueError(f"baseline {baseline!r} not among {names}")
    rows = []
    for r in results:
        s = r.summary.query("reference == @reference and k == @k")
        if query_set is not None:
            s = s.query("query_set == @query_set")
        g = s.groupby("attribute")[list(BIAS_METRICS)].mean()   # mean over query sets if several
        for attr, vals in g.iterrows():
            row = {"method": r.method, "category": r.category, "attribute": attr, **vals.to_dict()}
            if r.utility is not None:
                for _, u in r.utility.query("direction == 'text_to_image'").iterrows():
                    row[f"R@{int(u.k)}"] = u.recall
            rows.append(row)
    table = pd.DataFrame(rows)
    base = table[table.method == baseline].set_index("attribute")
    for col in [c for c in table.columns if c in BIAS_METRICS or c.startswith("R@")]:
        ref = table["attribute"].map(base[col])
        if col == "min_skew_mean":   # MinSkew is negative: "less bias" = closer to 0
            table[f"{col}_change_%"] = 100 * (table[col].abs() - ref.abs()) / ref.abs()
        else:
            table[f"{col}_change_%"] = 100 * (table[col] - ref) / ref.abs()
    if targets:
        b, u = f"{targets['bias_metric']}_mean_change_%", f"{targets['utility_metric']}_change_%"
        table["bias_target_met"] = table[b] <= -100 * targets["min_bias_reduction"]
        if u in table:
            table["utility_target_met"] = table[u] >= -100 * targets["max_utility_drop"]
        flags = [c for c in ("bias_target_met", "utility_target_met") if c in table]
        table[flags] = table[flags].astype(object)
        table.loc[table.method == baseline, flags] = None   # targets don't apply to the baseline itself
    return table


def compare_costs(results: Sequence[RunResult]) -> pd.DataFrame:
    """Cost of every method side by side (one row per method)."""
    c = pd.concat([r.costs for r in results], ignore_index=True)
    return c.pivot_table(index="method", columns="metric", values="value", sort=False)
