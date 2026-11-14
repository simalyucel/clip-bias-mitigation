"""Smoke tests for queries, ranking and the evaluate() pipeline (no model needed)."""
from pathlib import Path

import numpy as np

from vlbias.baseline import evaluate, summarize
from vlbias.data.queries import fill_template, load_queries
from vlbias.retrieval.rank import top_k_indices

ROOT = Path(__file__).resolve().parents[1]


def test_fill_template_article():
    assert fill_template("a photo of {a} {}", "engineer") == "a photo of an engineer"
    assert fill_template("a photo of {a} {}", "doctor") == "a photo of a doctor"
    assert fill_template("A photo of a {} person", "smart") == "A photo of a smart person"


def test_query_sets_load():
    chuang = load_queries(ROOT / "configs/queries/chuang2023.yaml")
    assert len(chuang) == 10
    berg = load_queries(ROOT / "configs/queries/berg2022_test.yaml")
    assert len(berg) == 33 * 8
    occ = load_queries(ROOT / "configs/queries/occupations.yaml")
    assert len(occ) == 30 * 3


def test_top_k_indices_order():
    gallery = np.eye(4, dtype=np.float32)
    query = np.array([[0.1, 0.9, 0.5, 0.0]], dtype=np.float32)
    assert top_k_indices(query, gallery, 3).tolist() == [[1, 2, 0]]


def test_evaluate_detects_planted_bias():
    # Group 1 is shifted along axis 0 only. A query along axis 0 should favour group 1;
    # a query along axis 1 is independent of the group. (Not normalised on purpose:
    # normalising would shrink group 1 on every other axis and leak the label.)
    rng = np.random.default_rng(0)
    n, d = 2000, 16
    labels = rng.integers(0, 2, n)
    gallery = rng.normal(size=(n, d))
    gallery[:, 0] += 1.5 * labels
    q_emb = np.eye(d, dtype=np.float32)[:2]

    queries = load_queries(ROOT / "configs/queries/chuang2023.yaml")[:2]
    per_query, per_group = evaluate(q_emb, queries, gallery.astype(np.float32),
                                    {"gender": labels}, {"gender": ["Male", "Female"]},
                                    ks=[100], references=["uniform", "gallery"])
    assert len(per_query) == 2 * 2 * 1
    assert len(per_group) == len(per_query) * 2
    uni = per_query[per_query.reference == "uniform"].set_index("concept")
    assert uni.loc[queries[0].concept, "max_skew"] > 0.5      # biased query: strong skew
    assert uni.loc[queries[1].concept, "max_skew"] < 0.3      # neutral query: little skew
    s = summarize(per_query)
    assert {"max_skew_mean", "ndkl_mean"} <= set(s.columns)
