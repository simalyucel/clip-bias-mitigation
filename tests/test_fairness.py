"""Unit tests for the fairness metrics, with values worked out by hand."""
import math

import numpy as np
import pytest

from vlbias.metrics import (
    evaluate_ranking,
    group_proportions,
    max_skew_at_k,
    min_skew_at_k,
    ndkl,
    reference_distribution,
    skew_at_k,
)

UNIFORM2 = np.array([0.5, 0.5])


# ---- distributions -------------------------------------------------------- #
def test_group_proportions():
    assert np.allclose(group_proportions([0, 0, 1, 2], 3), [0.5, 0.25, 0.25])


def test_group_proportions_missing_group():
    assert np.allclose(group_proportions([0, 0], 3), [1.0, 0.0, 0.0])


def test_reference_uniform():
    assert np.allclose(reference_distribution("uniform", 4), [0.25] * 4)


def test_reference_gallery():
    d = reference_distribution("gallery", 2, gallery_labels=[0, 0, 0, 1])
    assert np.allclose(d, [0.75, 0.25])


def test_reference_custom_validation():
    assert np.allclose(reference_distribution("custom", 2, custom=[0.85, 0.15]), [0.85, 0.15])
    with pytest.raises(ValueError):
        reference_distribution("custom", 2, custom=[0.5, 0.6])
    with pytest.raises(ValueError):
        reference_distribution("nonsense", 2)


# ---- skew ------------------------------------------------------------------ #
def test_skew_balanced_is_zero():
    ranking = [0, 1] * 50
    assert np.allclose(skew_at_k(ranking, UNIFORM2, k=100), 0.0)


def test_skew_hand_computed():
    # top-10: 8 of group 0, 2 of group 1, desired 50/50
    ranking = [0] * 8 + [1] * 2 + [1] * 90
    skew = skew_at_k(ranking, UNIFORM2, k=10)
    assert skew[0] == pytest.approx(math.log(0.8 / 0.5))
    assert skew[1] == pytest.approx(math.log(0.2 / 0.5))
    assert max_skew_at_k(ranking, UNIFORM2, 10) == pytest.approx(math.log(1.6))
    assert min_skew_at_k(ranking, UNIFORM2, 10) == pytest.approx(math.log(0.4))


def test_skew_only_uses_top_k():
    ranking = [0] * 10 + [1] * 1000
    assert skew_at_k(ranking, UNIFORM2, k=10)[0] == pytest.approx(math.log(2.0))


def test_skew_absent_group_is_floored():
    ranking = [0] * 10
    skew = skew_at_k(ranking, UNIFORM2, k=10, zero_floor=0.5)
    assert np.isfinite(skew).all()
    assert skew[1] == pytest.approx(math.log((0.5 / 10) / 0.5))


def test_skew_gallery_reference():
    # gallery is 75/25, top-4 is also 75/25 -> no skew
    desired = reference_distribution("gallery", 2, gallery_labels=[0, 0, 0, 1])
    assert np.allclose(skew_at_k([0, 1, 0, 0], desired, k=4), 0.0)


def test_k_larger_than_ranking_raises():
    with pytest.raises(ValueError):
        skew_at_k([0, 1], UNIFORM2, k=5)


# ---- ndkl ------------------------------------------------------------------ #
def test_ndkl_hand_computed():
    # i=1: D=[1,0], KL=ln2, weight 1 ; i=2: D=[.5,.5], KL=0, weight 1/log2(3)
    z = 1 + 1 / math.log2(3)
    assert ndkl([0, 1], UNIFORM2, k=2) == pytest.approx(math.log(2) / z)


def test_ndkl_perfect_interleaving_is_small_and_nonnegative():
    alternating = [0, 1] * 50
    blocked = [0] * 50 + [1] * 50
    a, b = ndkl(alternating, UNIFORM2), ndkl(blocked, UNIFORM2)
    assert 0 <= a < b


def test_ndkl_sees_position_but_skew_does_not():
    # same top-10 composition, different order: Skew identical, NDKL different
    early = [0] * 5 + [1] * 5
    late = [1] * 5 + [0] * 5
    assert np.allclose(skew_at_k(early, UNIFORM2, 10), skew_at_k(late, UNIFORM2, 10))
    mixed = [0, 1] * 5
    assert ndkl(mixed, UNIFORM2, 10) < ndkl(early, UNIFORM2, 10)


# ---- evaluate_ranking ------------------------------------------------------ #
def test_evaluate_ranking_rows():
    ranking = [0] * 8 + [1] * 2 + [0, 1] * 50
    rows = evaluate_ranking(ranking, UNIFORM2, ks=[10, 100])
    assert [r["k"] for r in rows] == [10, 100]
    r10 = rows[0]
    assert r10["max_skew"] == pytest.approx(math.log(1.6))
    assert r10["max_abs_skew"] == pytest.approx(abs(math.log(0.4)))
    assert r10["n_absent"] == 0


def test_evaluate_ranking_counts_absent():
    rows = evaluate_ranking([0] * 20, np.array([1 / 3] * 3), ks=[10])
    assert rows[0]["n_absent"] == 2
