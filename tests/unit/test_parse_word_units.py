from __future__ import annotations

import numpy as np

from scripts.parse_word_units import DIM, _mix_word, fixed_vec_array, merge_agg, parse_levels, vec_of


def _v(*xs: float) -> np.ndarray:
    return np.asarray(xs, dtype=np.float32)


def test_parse_levels_handles_descending_range() -> None:
    assert parse_levels("9-2") == [9, 8, 7, 6, 5, 4, 3, 2]
    assert parse_levels("2-4") == [4, 3, 2]
    assert parse_levels("9,7,5") == [9, 7, 5]


def test_mix_word_prefers_l14_and_keeps_both_counts() -> None:
    agg: dict = {}
    key = ("x", "noun")
    a, b, c = _v(1, 0, 0, 0), _v(0, 1, 0, 0), _v(0, 0, 1, 0)

    _mix_word(agg, key, a, 15, "en")
    _mix_word(agg, key, b, 15, "en")
    assert np.allclose(agg[key]["sum"], a + b)
    assert agg[key]["n_l15"] == 2

    # An L14 vector replaces the L15 mean but keeps the L15 occurrence count.
    _mix_word(agg, key, c, 14, "en")
    assert agg[key]["level"] == 14
    assert np.allclose(agg[key]["sum"], c)
    assert agg[key]["n"] == 1
    assert agg[key]["n_l14"] == 1
    assert agg[key]["n_l15"] == 2

    # Later L15 vectors are counted but do not change the L14 sum.
    _mix_word(agg, key, a, 15, "en")
    assert np.allclose(agg[key]["sum"], c)
    assert agg[key]["n_l15"] == 3


def test_mix_word_averages_multiple_l14_vectors() -> None:
    agg: dict = {}
    key = ("y", "verb")
    a, b = _v(1, 0, 0, 0), _v(0, 1, 0, 0)
    _mix_word(agg, key, a, 14, "en")
    _mix_word(agg, key, b, 14, "en")
    assert np.allclose(agg[key]["sum"], a + b)
    assert agg[key]["n"] == 2
    assert agg[key]["n_l14"] == 2


def test_merge_agg_prefers_l14_across_chunks() -> None:
    key = ("z", "adj")
    a, b, c = _v(1, 0, 0, 0), _v(0, 1, 0, 0), _v(0, 0, 1, 0)

    left: dict = {}
    _mix_word(left, key, a, 15, "en")
    _mix_word(left, key, b, 15, "en")
    right: dict = {}
    _mix_word(right, key, c, 14, "en")

    merge_agg(left, right)
    assert left[key]["level"] == 14
    assert np.allclose(left[key]["sum"], c)
    assert left[key]["n"] == 1
    assert left[key]["n_l15"] == 2  # preserved from the L15-only chunk


def test_merge_agg_sums_same_level_chunks() -> None:
    key = ("w", "noun")
    a, b = _v(1, 0, 0, 0), _v(0, 1, 0, 0)
    left: dict = {}
    _mix_word(left, key, a, 14, "en")
    right: dict = {}
    _mix_word(right, key, b, 14, "en")
    merge_agg(left, right)
    assert np.allclose(left[key]["sum"], a + b)
    assert left[key]["n"] == 2
    assert left[key]["n_l14"] == 2


def test_vec_of_rejects_missing_wrong_dim_and_nonfinite() -> None:
    assert vec_of(None) is None
    assert vec_of({}) is None
    assert vec_of({"vector": np.zeros(DIM - 1, dtype=np.float32)}) is None
    assert vec_of({"vector": np.full(DIM, np.nan, dtype=np.float32)}) is None
    ok = vec_of({"vector": np.ones(DIM, dtype=np.float32)})
    assert ok is not None and ok.shape == (DIM,)


def test_fixed_vec_array_round_trip_shape() -> None:
    arr = fixed_vec_array([np.zeros(DIM, dtype=np.float32), np.ones(DIM, dtype=np.float32)])
    assert len(arr) == 2
    assert arr.type.list_size == DIM
    assert arr[1].as_py()[:3] == [1.0, 1.0, 1.0]
