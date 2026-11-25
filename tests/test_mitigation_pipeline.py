"""Tests for the common evaluation pipeline and the Mitigation interface (no model or download needed).

A fake gallery with a planted gender direction (axis 0) is searched with queries that point
partly along that direction. Toy methods then show that each plug-in point works:
a pre-processing projection removes axis 0, a post-processing re-ranker alternates groups.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlbias.baseline import evaluate
from vlbias.data.flickr30k import RetrievalDataset
from vlbias.data.gallery import LabelledGallery
from vlbias.mitigation.base import Compose, Mitigation, NoMitigation, normalize
from vlbias.pipeline import EvaluationPipeline, compare, compare_costs, load_result
from vlbias.retrieval.rank import top_k_indices

ROOT = Path(__file__).resolve().parents[1]
N, D = 1200, 16


class FakeVLM:
    name = "fake"


def _setup(tmp_path, utility=True):
    rng = np.random.default_rng(0)
    gender = rng.integers(0, 2, N)
    race = rng.integers(0, 3, N)
    g = rng.normal(size=(N, D))
    g[:, 0] += 2.0 * gender                        # planted bias direction
    g = normalize(g)
    gallery = LabelledGallery(images=None, labels={"gender": gender, "race": race},
                              label_names={"gender": ["Male", "Female"], "race": ["A", "B", "C"]},
                              name="fake-gallery")
    cfg = {
        "seed": 0, "model": None,
        "query_sets": ["configs/queries/chuang2023.yaml"],
        "attributes": ["gender", "race"],
        "metrics": {"k": [100], "references": ["uniform", "gallery"], "zero_floor": 0.5},
        "significance": {"n_null": 50, "seed": 0, "alpha": 0.05},
        "output_dir": str(tmp_path / "results"),
    }
    if utility:
        cfg["utility"] = {"ks": [1, 5, 10]}
    pipe = EvaluationPipeline(cfg, root=ROOT, cache_dir=tmp_path, vlm=FakeVLM(), gallery=gallery, verbose=False)
    q = rng.normal(size=(len(pipe.queries), D))
    q[:, 0] += 3.0                                  # every query leans towards group 1
    pipe._emb_cache[("fake", "gallery")] = g
    pipe._emb_cache[("fake", "queries")] = normalize(q)
    if utility:                                     # tiny retrieval set: caption i describes image i // 2
        img = normalize(rng.normal(size=(50, D)))
        txt = normalize(np.repeat(img, 2, axis=0) + 0.3 * rng.normal(size=(100, D)))
        pipe._flickr = RetrievalDataset(images=None, captions=[str(i) for i in range(100)],
                                        caption_to_image=np.repeat(np.arange(50), 2), name="fake-flickr")
        pipe._emb_cache[("fake", "flickr_img")] = img
        pipe._emb_cache[("fake", "flickr_txt")] = txt
    return pipe, gender


class DropAxis0(Mitigation):
    """Toy pre-processing: remove the planted bias direction from text and image embeddings."""
    name, category = "drop_axis0", "pre"

    def transform_text(self, emb):
        emb = emb.copy(); emb[:, 0] = 0
        return normalize(emb)

    def transform_image(self, emb):
        return self.transform_text(emb)


class AlternateGender(Mitigation):
    """Toy post-processing: re-rank so groups alternate (Kong et al.'s idea, with true labels)."""
    name, category = "alternate", "post"

    def __init__(self, gallery_emb, labels):
        self.lookup = {tuple(np.round(e, 6)): l for e, l in zip(gallery_emb, labels)}

    def rank(self, query_emb, gallery_emb, k):
        lab = np.array([self.lookup.get(tuple(np.round(e, 6)), 0) for e in gallery_emb])
        full = top_k_indices(query_emb, gallery_emb, len(gallery_emb))
        out = []
        for row in full:
            a, b = row[lab[row] == 0], row[lab[row] == 1]
            if len(a) < k // 2 or len(b) < k // 2:   # unknown images (e.g. Flickr): keep the plain order
                out.append(row[:k])
            else:
                out.append(np.ravel(np.column_stack([a[: k // 2], b[: k // 2]]))[:k])
        return np.array(out)


def _mean(res, attr="gender", ref="uniform"):
    return res.summary.query("attribute == @attr and reference == @ref")["max_skew_mean"].iloc[0]


def test_no_mitigation_reproduces_baseline_evaluate(tmp_path):
    pipe, _ = _setup(tmp_path)
    res = pipe.run(NoMitigation())
    direct, _ = evaluate(pipe._emb_cache[("fake", "queries")], pipe.queries, pipe._emb_cache[("fake", "gallery")],
                         pipe.labels, pipe.label_names, ks=[100], references=["uniform", "gallery"],
                         nulls=pipe.nulls)
    cols = ["max_skew", "min_skew", "ndkl", "max_skew_p_adj"]
    np.testing.assert_allclose(res.per_query[cols].values, direct[cols].values)
    assert _mean(res) > 0.4                         # planted bias is visible


def test_pre_processing_removes_planted_bias(tmp_path):
    pipe, _ = _setup(tmp_path)
    base, pre = pipe.run(NoMitigation()), pipe.run(DropAxis0())
    assert _mean(pre) < 0.5 * _mean(base)


def test_post_processing_reranking_balances(tmp_path):
    pipe, gender = _setup(tmp_path)
    base = pipe.run(NoMitigation())
    post = pipe.run(AlternateGender(pipe._emb_cache[("fake", "gallery")], gender))
    assert _mean(post) < 0.05 < _mean(base)         # exactly 50/50 in the top 100
    assert post.utility.query("direction == 'text_to_image'")["recall"].notna().all()


def test_compose_and_compare(tmp_path):
    pipe, gender = _setup(tmp_path)
    g = pipe._emb_cache[("fake", "gallery")]
    runs = [pipe.run(NoMitigation()), pipe.run(DropAxis0()),
            pipe.run(Compose([DropAxis0(), AlternateGender(DropAxis0().transform_image(g), gender)]))]
    assert runs[2].category == "combined" and runs[2].method == "drop_axis0+alternate"
    table = compare(runs, reference="uniform", k=100,
                    targets={"bias_metric": "max_skew", "utility_metric": "R@5",
                             "min_bias_reduction": 0.05, "max_utility_drop": 0.05})
    gend = table[table.attribute == "gender"].set_index("method")
    assert gend.loc["none", "max_skew_mean_change_%"] == pytest.approx(0)
    assert gend.loc["drop_axis0", "max_skew_mean_change_%"] < -50
    assert bool(gend.loc["drop_axis0", "bias_target_met"])
    costs = compare_costs(runs)
    assert {"fit_seconds", "latency_per_query_ms", "extra_parameters"} <= set(costs.columns)


def test_two_rerankers_cannot_be_combined():
    with pytest.raises(ValueError):
        Compose([AlternateGender(np.zeros((1, 2)), [0]), AlternateGender(np.zeros((1, 2)), [0])])


def test_results_saved_and_reloaded(tmp_path):
    pipe, _ = _setup(tmp_path)
    res = pipe.run(DropAxis0())
    for f in ("per_query.csv", "per_group.csv", "summary.csv", "costs.csv", "utility.csv", "method.yaml"):
        assert (res.output_dir / f).exists()
    back = load_result(res.output_dir)
    assert back.method == "drop_axis0" and back.category == "pre"
    pd.testing.assert_frame_equal(back.summary, res.summary, check_dtype=False)


def test_utility_unchanged_without_mitigation(tmp_path):
    pipe, _ = _setup(tmp_path)
    u = pipe.run(NoMitigation()).utility
    assert set(u.direction) == {"text_to_image", "image_to_text"}
    assert (u.recall >= 0).all() and (u.recall <= 100).all()
