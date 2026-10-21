from __future__ import annotations

import numpy as np

from scripts.train_word_model import (
    DIM,
    bootstrap_mean_delta,
    evaluate,
    grouped_split,
    make_features,
    mean_prediction,
    normalize_rows,
    uniform_prediction,
)


def test_grouped_split_keeps_groups_disjoint_and_levels() -> None:
    levels = np.array([9, 9, 9, 9, 8, 8, 8], dtype=np.int16)
    groups = np.array(["a", "a", "b", "c", "d", "d", "e"], dtype=object)
    split = grouped_split(levels, groups, seed=1, val_frac=0.2, test_frac=0.2)

    # All members of a group stay together.
    by_group: dict[str, set[str]] = {}
    for g, s in zip(groups, split):
        by_group.setdefault(g, set()).add(s)
    assert all(len(v) == 1 for v in by_group.values())

    # Every level contributes training rows, and split labels are known.
    assert set(split) <= {"train", "val", "test"}
    for level in (8, 9):
        assert "train" in set(split[levels == level])


def _fake_data() -> dict:
    n = 4
    mean1 = np.arange(n * DIM, dtype=np.float32).reshape(n, DIM) / 1000.0
    mean2 = mean1 + 0.01
    return {
        "level": np.array([9, 8, 7, 2], dtype=np.int16),
        "mean1": mean1,
        "mean2": mean2,
        "n1": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "n2": np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float32),
        "mb_vec": np.zeros((n, DIM), dtype=np.float32),
        "bridge_vec": np.zeros((n, DIM), dtype=np.float32),
        "bridge_mean": np.zeros((n, DIM), dtype=np.float32),
    }


def test_make_features_shapes_onehot_and_role_swap() -> None:
    data = _fake_data()
    x = make_features(data)
    assert x.shape == (4, 2060)
    assert x[0, 0] == data["mean1"][0, 0]
    assert x[0, DIM] == data["mean2"][0, 0]
    # L9 -> first level column, L2 -> last level column.
    assert x[0, 2 * DIM] == 1.0
    assert x[0, 2 * DIM + 7] == 0.0
    assert x[3, 2 * DIM + 7] == 1.0

    swapped = make_features(data, swap_roles=True)
    assert np.allclose(swapped[:, :DIM], data["mean2"])
    assert np.allclose(swapped[:, DIM:2 * DIM], data["mean1"])
    counts_start = 2 * DIM + 8
    assert np.allclose(swapped[:, counts_start], np.log1p(data["n2"]))


def test_normalize_rows_and_evaluate() -> None:
    y = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    p = y.copy()
    metrics = evaluate(y, p, np.array([9, 9], dtype=np.int16))
    assert metrics["overall"]["cos"]["median"] == 1.0
    assert metrics["overall"]["relres"]["median"] == 0.0
    assert metrics["per_level"]["9"]["cos"]["n"] == 2


def test_uniform_prediction_is_count_weighted_mean() -> None:
    data = _fake_data()
    idx = np.array([0, 1])
    p = uniform_prediction(data, idx)
    expected = (data["n1"][idx, None] * data["mean1"][idx]
                + data["n2"][idx, None] * data["mean2"][idx]) / (
        data["n1"][idx] + data["n2"][idx]
    )[:, None]
    assert np.allclose(p, expected)


def test_mean_prediction_uses_fit_rows_per_level() -> None:
    data = _fake_data()
    data["mb_vec"][0] = normalize_rows(np.array([[1.0] + [0.0] * (DIM - 1)], dtype=np.float32))[0]
    data["mb_vec"][1] = normalize_rows(np.array([[0.0, 1.0] + [0.0] * (DIM - 2)], dtype=np.float32))[0]
    fit = np.array([True, True, False, False])
    idx = np.array([0, 1, 2, 3])
    p = mean_prediction(data, fit, idx, per_level=False)
    expected = normalize_rows(data["mb_vec"][:2].mean(axis=0, keepdims=True))[0]
    assert np.allclose(p[0], expected)
    assert np.allclose(p[3], expected)


def test_bootstrap_delta_identical_vectors() -> None:
    a = np.linspace(-0.2, 0.8, 50)
    out = bootstrap_mean_delta(a, a, n_boot=200, seed=0)
    assert out["delta_mean"] == 0.0
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
