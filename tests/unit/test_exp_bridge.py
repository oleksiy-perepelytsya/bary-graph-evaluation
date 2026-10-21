from __future__ import annotations

import math

import numpy as np
import pytest

from lib.vector import cosine, normalize
from scripts.exp_bridge import (
    auc_from_pairs,
    band_tolerances,
    is_degenerate,
    jaccard,
    oracle_bridge,
    rank_metrics,
    rank_of_true,
    residual,
)


def unit(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    return normalize(rng.normal(size=dim))


def test_oracle_exactness() -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        c1, c2, b = unit(rng), unit(rng), unit(rng)
        w1, w2, w3 = (float(x) for x in rng.uniform(0.05, 1.5, size=3))
        m = normalize(w1 * c1 + w2 * c2 + w3 * b)
        solved = oracle_bridge(m, c1, c2, w1, w2, w3)
        assert np.linalg.norm(solved) == pytest.approx(1.0, abs=1e-4)
        assert cosine(solved, b) == pytest.approx(1.0, abs=1e-5)


def test_oracle_scale_invariance_all_three_weights() -> None:
    """Scaling children by s requires dividing *all three* weights by s.

    This is the handoff's known-bug guard: multiplying only w3 describes a
    different vector.
    """
    rng = np.random.default_rng(11)
    for s in (0.5, 2.0, 10.0):
        c1, c2, b = unit(rng), unit(rng), unit(rng)
        w1, w2, w3 = 0.7, 0.9, 0.4
        m = normalize(w1 * c1 + w2 * c2 + w3 * b)
        solved = oracle_bridge(m, s * c1, s * c2, w1 / s, w2 / s, w3 / s)
        assert cosine(solved, b) == pytest.approx(1.0, abs=1e-5)


def test_oracle_zero_bridge_weight_returns_query() -> None:
    rng = np.random.default_rng(13)
    c1, c2, b = unit(rng), unit(rng), unit(rng)
    m = normalize(c1 + c2)
    solved = oracle_bridge(m, c1, c2, 1.0, 1.0, 0.0)
    assert np.allclose(solved, m)


def test_residual_zero_for_faithful_triad_and_positive_for_swap() -> None:
    rng = np.random.default_rng(17)
    c1, c2, b, b2 = unit(rng), unit(rng), unit(rng), unit(rng)
    w1, w2, w3 = 0.6, 0.8, 0.3
    m = normalize(w1 * c1 + w2 * c2 + w3 * b)
    assert residual(m, c1, c2, b, w1, w2, w3) == pytest.approx(0.0, abs=1e-6)
    assert residual(m, c1, c2, b2, w1, w2, w3) > 0.01


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_rank_ties() -> None:
    assert rank_of_true([0.5, 0.5, 0.2]) == 1
    assert rank_of_true([0.5, 0.9, 0.2]) == 2
    assert rank_of_true([0.5, 0.9, 0.9]) == 3
    assert rank_of_true([0.5, 0.5, 0.5]) == 1
    assert rank_of_true([]) == 0


def test_rank_metrics_tie_aware() -> None:
    m = rank_metrics([0.5, 0.5, 0.2])
    assert m["rank"] == 1
    assert m["top1"] == 1.0
    assert m["top1_strict"] == 0.0
    assert m["tie_rate"] == pytest.approx(0.5)
    assert m["mrr_tie"] == pytest.approx(1.0 / 1.5)

    m = rank_metrics([0.5, 0.9, 0.5])
    assert m["rank"] == 2
    assert m["top1"] == 0.0
    assert m["top1_strict"] == 0.0
    assert m["mrr_tie"] == pytest.approx(1.0 / 2.5)

    m = rank_metrics([0.9, 0.5, 0.2])
    assert m["rank"] == 1
    assert m["top1_strict"] == 1.0
    assert m["tie_rate"] == 0.0
    assert m["mrr_tie"] == pytest.approx(1.0)

    empty = rank_metrics([])
    assert empty["rank"] == 0
    assert empty["top1"] == 0.0


def test_degenerate_flag() -> None:
    assert is_degenerate({"a", "b"}, {"a", "b", "c"}) is True  # subset
    assert is_degenerate({"a", "b"}, {"a", "c"}) is False  # Jaccard 1/3
    assert is_degenerate({"a", "b"}, {"b", "c"}) is False  # Jaccard 1/3
    assert is_degenerate({"a", "b"}, {"a", "b", "c", "d"}) is True  # 2/4 = 0.5
    assert is_degenerate({"a"}, {"b", "c"}) is False
    assert is_degenerate(set(), {"a"}) is True  # empty set is a subset


def test_auc_from_pairs() -> None:
    assert auc_from_pairs([(0.1, 0.2), (0.0, 0.5)]) == 1.0
    assert auc_from_pairs([(0.2, 0.1), (0.5, 0.0)]) == 0.0
    assert auc_from_pairs([(0.1, 0.1)]) == 0.5
    assert auc_from_pairs([(0.1, 0.2), (0.3, 0.3), (0.5, 0.4)]) == pytest.approx(0.5)
    assert auc_from_pairs([]) is None


def test_band_tolerances_are_monotone_and_include_inf() -> None:
    for base in (0.05, 0.15):
        tol = band_tolerances(base)
        assert tol[0] == base
        assert tol == sorted(tol)
        assert math.isinf(tol[-1])
