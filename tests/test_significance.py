"""Tests for chance distributions, p-values, BH correction and concept-level significance."""
import numpy as np
import pandas as pd
import pytest

from vlbias.analysis import concept_skew
from vlbias.baseline import compute_nulls, evaluate
from vlbias.data.queries import Query
from vlbias.metrics import evaluate_ranking
from vlbias.metrics.null import (benjamini_hochberg, chance_distributions, null_of_mean,
                                 p_greater, p_less, p_two_sided, ranking_metrics_batch)


def test_batch_agrees_with_single_ranking():
    rng = np.random.default_rng(0)
    rankings = rng.integers(0, 3, size=(20, 50))
    desired = np.array([0.5, 0.3, 0.2])
    batch = ranking_metrics_batch(rankings, desired, ks=[10, 50])
    for i in range(20):
        for row in evaluate_ranking(rankings[i], desired, ks=[10, 50]):
            b = batch[row["k"]]
            assert np.allclose(b["skew"][i], row["skew"])
            assert b["max_skew"][i] == pytest.approx(row["max_skew"])
            assert b["min_skew"][i] == pytest.approx(row["min_skew"])
            assert b["ndkl"][i] == pytest.approx(row["ndkl"])


def test_chance_level_of_balanced_gallery_is_small():
    labels = np.array([0, 1] * 500)
    d = chance_distributions(labels, {"uniform": np.array([0.5, 0.5])}, ks=[100], n_null=300)
    m = d[("uniform", 100)]["max_skew"]
    assert m.shape == (300,)
    assert 0 < m.mean() < 0.15          # random top-100 of a 50/50 gallery: small skew


def test_chance_level_reflects_gallery_imbalance():
    labels = np.array([0] * 900 + [1] * 100)
    d = chance_distributions(labels, {"uniform": np.array([0.5, 0.5])}, ks=[100], n_null=200)
    assert d[("uniform", 100)]["max_skew"].mean() == pytest.approx(np.log(0.9 / 0.5), abs=0.05)


def test_p_values():
    null = np.arange(100, dtype=float)      # 0..99
    assert p_greater(99, null)[0] == pytest.approx(2 / 101)
    assert p_less(0, null)[0] == pytest.approx(2 / 101)
    assert p_greater(-5, null)[0] == pytest.approx(1.0)
    assert p_two_sided(50, null)[0] == pytest.approx(1.0)
    assert p_two_sided(1000, null)[0] == pytest.approx(2 / 101)


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # sorted: .01 .03 .04 .20 -> *4/rank: .04 .06 .0533 .20 -> monotone: .04 .0533 .0533 .20
    assert np.allclose(benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.20])
    assert np.isnan(benjamini_hochberg([np.nan, 0.5])[0])


def test_null_of_mean_is_narrower():
    null = np.random.default_rng(0).normal(size=5000)
    assert null_of_mean(null, 8).std() < null.std() / 2


def _planted(n=3000, d=16, shift=1.5, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    gallery = rng.normal(size=(n, d))
    gallery[:, 0] += shift * labels
    return labels, gallery.astype(np.float32), np.eye(d, dtype=np.float32)[:2]


def test_significance_flags_planted_bias_only():
    labels, gallery, q_emb = _planted()
    queries = [Query("biased", "biased", "{}", "negative", "toy"),
               Query("neutral", "neutral", "{}", "positive", "toy")]
    lab, names = {"gender": labels}, {"gender": ["Male", "Female"]}
    nulls = compute_nulls(lab, names, ks=[100], references=["gallery"], n_null=500)
    pq, pg = evaluate(q_emb, queries, gallery, lab, names, ks=[100], references=["gallery"], nulls=nulls)
    sig = pq.set_index("concept")["max_skew_sig"]
    assert bool(sig["biased"]) and not bool(sig["neutral"])
    assert {"chance_lo", "chance_hi", "p_value", "p_adj", "significant"} <= set(pg.columns)
    fem = pg.query("concept == 'biased' and group == 'Female'").iloc[0]
    assert fem["significant"] and fem["skew"] > fem["chance_hi"]


def test_excluded_labels_are_left_out():
    labels, gallery, q_emb = _planted()
    labels = labels.copy()
    labels[:500] = -1                                   # e.g. children for age_adult
    queries = [Query("q", "q", "{}", "c", "toy")]
    lab, names = {"x": labels}, {"x": ["A", "B"]}
    nulls = compute_nulls(lab, names, ks=[100], references=["gallery"], n_null=100)
    pq, pg = evaluate(q_emb[:1], queries, gallery, lab, names, ks=[100], references=["gallery"], nulls=nulls)
    assert len(pq) == 1 and set(pg["group"]) == {"A", "B"}


def test_concept_skew_table():
    labels, gallery, q_emb = _planted()
    queries = [Query(f"t{i}", c, f"tpl{i}", cat, "toy")
               for c, cat, i in [("biased", "neg", 0), ("biased", "neg", 1), ("neutral", "pos", 0)]]
    emb = np.stack([q_emb[0], q_emb[0], q_emb[1]])
    lab, names = {"gender": labels}, {"gender": ["Male", "Female"]}
    nulls = compute_nulls(lab, names, ks=[100], references=["gallery"], n_null=300)
    _, pg = evaluate(emb, queries, gallery, lab, names, ks=[100], references=["gallery"], nulls=nulls)
    cs = concept_skew(pg, nulls, "gender", k=100)
    assert set(cs["concept"]) == {"biased", "neutral"}
    assert cs.loc[cs.concept == "biased", "n_templates"].eq(2).all()
    row = cs.query("concept == 'biased' and group == 'Female'").iloc[0]
    assert row["significant"]
