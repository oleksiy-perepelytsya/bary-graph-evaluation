"""Unit tests for the light probe trainer's pure helpers."""

from __future__ import annotations

import json

import numpy as np

from scripts.train_probe import (
    bootstrap_delta,
    flatten_cases,
    group_bucket,
    group_key,
    make_features,
    make_grouped_split,
    make_random_split,
    public_metrics,
    rank_scores,
    shuffled_case_labels,
)


def _synthetic(n: int = 5, k: int = 3, d: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {
        "m": rng.normal(size=(n, d)).astype(np.float32),
        "centroid": rng.normal(size=(n, d)).astype(np.float32),
        "b_true": rng.normal(size=(n, d)).astype(np.float32),
        "b_negs": rng.normal(size=(n, k, d)).astype(np.float32),
        "level": rng.choice([9, 7], size=n).astype(np.int16),
        "bridge_n": rng.integers(4, 9, size=n).astype(np.float32),
        "child_n": rng.integers(8, 16, size=n).astype(np.float32),
        "neg_n": rng.integers(4, 9, size=(n, k)).astype(np.float32),
        "weight_delta_true": np.zeros(n, dtype=np.float32),
        "weight_delta_negs": rng.random((n, k)).astype(np.float32) + 0.01,
    }
    for name in ("cos_m", "cos_cent", "cos_b_c1", "cos_b_c2", "jaccard"):
        data[f"{name}_true"] = rng.random(n).astype(np.float32)
        data[f"{name}_negs"] = rng.random((n, k)).astype(np.float32)
    for name in ("overlap",):
        data[f"{name}_true"] = rng.integers(0, 3, size=n).astype(np.float32)
        data[f"{name}_negs"] = rng.integers(0, 3, size=(n, k)).astype(np.float32)
    return data


def test_group_key_is_order_insensitive() -> None:
    a = json.dumps(["beta", "alpha", "gamma"])
    b = json.dumps(["gamma", "alpha", "beta"])
    assert group_key(a) == group_key(b)
    assert group_bucket(a, seed=3) == group_bucket(b, seed=3)


def test_grouped_split_has_no_exact_group_leakage() -> None:
    words = np.array([json.dumps([f"word-{i}"]) for i in range(500)])
    split = make_grouped_split(words, seed=7)
    train = {group_key(w) for w in words[split == "train"]}
    for name in ("val", "test"):
        assert {group_key(w) for w in words[split == name]} & train == set()
    assert (split == "train").any()
    assert (split == "val").any()
    assert (split == "test").any()


def test_random_split_is_reproducible_and_stratified() -> None:
    levels = np.array([9] * 100 + [7] * 50)
    a = make_random_split(levels, seed=11)
    b = make_random_split(levels, seed=11)
    assert np.array_equal(a, b)
    for level in (9, 7):
        for name in ("train", "val", "test"):
            assert ((a == name) & (levels == level)).any()


def test_rank_scores_tie_handling() -> None:
    # True ties one negative: competition top-1, not strict top-1.
    m = rank_scores(np.array([1.0]), np.array([[1.0, 0.0]]))
    assert m["top1"] == 1.0
    assert m["top1_strict"] == 0.0
    assert abs(m["mrr_tie"] - 2.0 / 3.0) < 1e-12

    # A strictly higher negative wins.
    m = rank_scores(np.array([1.0]), np.array([[0.0, 2.0]]))
    assert m["top1"] == 0.0
    assert m["top1_strict"] == 0.0
    assert m["mrr"] == 0.5

    # Strict win.
    m = rank_scores(np.array([2.0]), np.array([[1.0, 1.0]]))
    assert m["top1"] == 1.0
    assert m["top1_strict"] == 1.0
    assert m["mrr_tie"] == 1.0


def test_bootstrap_delta_of_identical_scores_is_zero() -> None:
    a = np.array([1.0, 0.0, 1.0, 1.0])
    out = bootstrap_delta(a, a, n_boot=200, seed=2)
    assert out["delta"] == 0.0
    assert out["ci_low"] == 0.0
    assert out["ci_high"] == 0.0


def test_flatten_cases_shapes_and_labels() -> None:
    ft = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    fn = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    mask = np.array([True, False, True, False, True])
    x, y = flatten_cases(ft, fn, mask)
    assert x.shape == (3 * 3, 3)
    assert y.shape == (3 * 3,)
    assert list(y) == [1, 0, 0, 1, 0, 0, 1, 0, 0]


def test_shuffled_case_labels_has_one_positive_per_case() -> None:
    rng = np.random.default_rng(0)
    y = shuffled_case_labels(7, 9, rng)
    assert y.shape == (7 * 10,)
    rows = y.reshape(7, 10)
    assert np.array_equal(rows.sum(axis=1), np.ones(7))


def test_make_features_shapes_and_no_weight_delta() -> None:
    data = _synthetic()
    expected = {
        "vector": 2 * 4,
        "cos": 5,
        "text_only": 5,
        "text_geom": 10,
        "vector_geom": 2 * 4 + 5,
        "all": 2 * 4 + 5 + 5,
    }
    for kind, f in expected.items():
        ft, fn = make_features(data, kind)
        assert ft.shape == (5, f), kind
        assert fn.shape == (5, 3, f), kind

        # weight_delta is an oracle artifact and must not influence features.
        mutated = dict(data)
        mutated["weight_delta_true"] = data["weight_delta_true"] + 99.0
        mutated["weight_delta_negs"] = data["weight_delta_negs"] + 99.0
        ft2, fn2 = make_features(mutated, kind)
        assert np.array_equal(ft, ft2), kind
        assert np.array_equal(fn, fn2), kind

    ft, fn = make_features(data, "vector_geom", include_level=True)
    assert ft.shape == (5, 2 * 4 + 5 + 1)
    assert fn.shape == (5, 3, 2 * 4 + 5 + 1)


def test_public_metrics_strips_per_case_arrays() -> None:
    m = rank_scores(np.array([1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0]]))
    pub = public_metrics(m)
    assert "per_case_top1_strict" not in pub
    assert "per_case_mrr_tie" not in pub
    assert pub["n"] == 2
