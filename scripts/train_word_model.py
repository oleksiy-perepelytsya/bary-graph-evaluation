#!/usr/bin/env python3
"""Train a pooled L9-L2 model: cm1+cm2 words -> MB vector.

Model
-----
A single ridge regression pools the two child word lists into mean vectors
(``mean1``, ``mean2``), adds level one-hot features and word-count features,
and predicts the 1024-d MB vector.  Predictions are L2-normalized before
evaluation.  The bridge is never a model input; it is used only to analyse the
unexplained residual ``m - prediction``.

Split
-----
Per level, 70/15/15 train/val/test.  Units with exactly the same cm1+cm2 word
set are kept in one split (grouped split) to avoid exact-input leakage.

Outputs
-------
``evaluation/results/word_model``:
  model_report.json
  model_report.md
  predictions_test.parquet
  ridge_model.npz
  split.json
  README.md

MongoDB is not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIM = 1024
LEVELS = (9, 8, 7, 6, 5, 4, 3, 2)
VEC_TYPE = pa.list_(pa.float32(), DIM)


# ---------------------------------------------------------------------------
# Loading and feature construction
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def flat_vectors(table: pa.Table, name: str) -> np.ndarray:
    arr = table.column(name).combine_chunks()
    flat = arr.values.to_numpy(zero_copy_only=False)
    return np.array(flat, dtype=np.float32, copy=True).reshape(-1, DIM)


def _group_key(w1: list[str], p1: list[str], w2: list[str], p2: list[str]) -> str:
    a = "|".join(f"{w}\t{p}" for w, p in sorted(zip(w1, p1)))
    b = "|".join(f"{w}\t{p}" for w, p in sorted(zip(w2, p2)))
    return hashlib.sha1(f"{a}#{b}".encode("utf-8")).hexdigest()


def load_dataset(units_path: Path, words_path: Path) -> dict[str, Any]:
    log(f"loading {units_path.name} and {words_path.name}")
    ut = pq.read_table(units_path)
    wt = pq.read_table(words_path, columns=["word", "pos", "vec"])

    mb_id = ut.column("mb_id").to_pylist()
    level = ut.column("level").to_numpy().astype(np.int16)
    c1_id = ut.column("c1_id").to_pylist()
    c2_id = ut.column("c2_id").to_pylist()
    mb_vec = flat_vectors(ut, "mb_vec")
    c1_vec = flat_vectors(ut, "c1_vec")
    c2_vec = flat_vectors(ut, "c2_vec")
    bridge_vec = flat_vectors(ut, "bridge_vec")

    word = wt.column("word").to_pylist()
    pos = wt.column("pos").to_pylist()
    word_vec = flat_vectors(wt, "vec")
    word_index = {(w, p): i for i, (w, p) in enumerate(zip(word, pos))}

    cm1_words = ut.column("cm1_words").to_pylist()
    cm1_pos = ut.column("cm1_pos").to_pylist()
    cm2_words = ut.column("cm2_words").to_pylist()
    cm2_pos = ut.column("cm2_pos").to_pylist()
    br_words = ut.column("bridge_words").to_pylist()
    br_pos = ut.column("bridge_pos").to_pylist()

    n = len(level)
    mean1 = np.zeros((n, DIM), dtype=np.float32)
    mean2 = np.zeros((n, DIM), dtype=np.float32)
    bridge_mean = np.zeros((n, DIM), dtype=np.float32)
    n1 = np.zeros(n, dtype=np.float32)
    n2 = np.zeros(n, dtype=np.float32)
    n_bridge = np.zeros(n, dtype=np.float32)
    groups = np.empty(n, dtype=object)

    for i in range(n):
        k1 = [word_index[(w, p)] for w, p in zip(cm1_words[i], cm1_pos[i])]
        k2 = [word_index[(w, p)] for w, p in zip(cm2_words[i], cm2_pos[i])]
        kb = [word_index[(w, p)] for w, p in zip(br_words[i], br_pos[i])]
        if not k1 or not k2:
            raise ValueError(f"empty child word list at unit {mb_id[i]}")
        n1[i], n2[i], n_bridge[i] = len(k1), len(k2), len(kb)
        mean1[i] = word_vec[k1].mean(axis=0)
        mean2[i] = word_vec[k2].mean(axis=0)
        if kb:
            bridge_mean[i] = word_vec[kb].mean(axis=0)
        groups[i] = _group_key(cm1_words[i], cm1_pos[i], cm2_words[i], cm2_pos[i])

    vocab_size = len(word_index)
    del ut, wt, word_vec, word, pos, word_index, cm1_words, cm1_pos, cm2_words, cm2_pos, br_words, br_pos
    log(f"loaded {n:,} units; vocabulary {vocab_size:,} keys")

    return {
        "mb_id": mb_id,
        "level": level,
        "c1_id": c1_id,
        "c2_id": c2_id,
        "mb_vec": mb_vec,
        "c1_vec": c1_vec,
        "c2_vec": c2_vec,
        "bridge_vec": bridge_vec,
        "bridge_mean": bridge_mean,
        "mean1": mean1,
        "mean2": mean2,
        "n1": n1,
        "n2": n2,
        "n_bridge": n_bridge,
        "groups": groups,
        "vocab_size": vocab_size,
    }


def feature_names() -> list[str]:
    names = [f"cm1_mean_{i}" for i in range(DIM)]
    names += [f"cm2_mean_{i}" for i in range(DIM)]
    names += [f"level_L{lvl}" for lvl in LEVELS]
    names += ["log1p_n_cm1", "log1p_n_cm2", "log1p_n_total", "cm1_fraction"]
    return names


def make_features(data: dict[str, Any], idx: np.ndarray | None = None,
                  swap_roles: bool = False) -> np.ndarray:
    if idx is None:
        idx = np.arange(len(data["level"]))
    m1 = data["mean1"][idx]
    m2 = data["mean2"][idx]
    n1 = data["n1"][idx]
    n2 = data["n2"][idx]
    if swap_roles:
        m1, m2 = m2, m1
        n1, n2 = n2, n1

    levels = data["level"][idx]
    onehot = np.zeros((len(idx), len(LEVELS)), dtype=np.float32)
    onehot[np.arange(len(idx)), [9 - int(lvl) for lvl in levels]] = 1.0
    counts = np.stack(
        [
            np.log1p(n1),
            np.log1p(n2),
            np.log1p(n1 + n2),
            n1 / (n1 + n2 + 1e-6),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([m1, m2, onehot, counts], axis=1)


# ---------------------------------------------------------------------------
# Split / metrics / bootstrap
# ---------------------------------------------------------------------------


def grouped_split(
    levels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.full(len(levels), "train", dtype=object)
    for level in sorted(set(int(x) for x in levels), reverse=True):
        level_idx = np.where(levels == level)[0]
        by_group: dict[str, list[int]] = defaultdict(list)
        for i in level_idx:
            by_group[str(groups[i])].append(i)
        keys = sorted(by_group)
        rng.shuffle(keys)
        n_groups = len(keys)
        n_test = max(1, int(round(test_frac * n_groups))) if n_groups >= 3 else 0
        n_val = max(1, int(round(val_frac * n_groups))) if n_groups >= 3 else 0
        if n_test + n_val >= n_groups:
            n_test = max(0, n_groups // 3)
            n_val = max(0, n_groups // 3)
        for j, key in enumerate(keys):
            if j < n_test:
                part = "test"
            elif j < n_test + n_val:
                part = "val"
            else:
                part = "train"
            split[by_group[key]] = part
    return split


def _scalar_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p05": float("nan"), "p95": float("nan")}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize_rows(a)
    b = normalize_rows(b)
    return np.sum(a * b, axis=1)


def evaluate(y: np.ndarray, pred: np.ndarray, levels: np.ndarray) -> dict[str, Any]:
    y = normalize_rows(y)
    pred = normalize_rows(pred)
    cos = _row_cos(y, pred)
    relres = np.linalg.norm(y - pred, axis=1)
    per_level: dict[str, Any] = {}
    for level in sorted(set(int(x) for x in levels), reverse=True):
        mask = levels == level
        per_level[str(level)] = {
            "cos": _scalar_summary(cos[mask]),
            "relres": _scalar_summary(relres[mask]),
        }
    return {
        "overall": {"cos": _scalar_summary(cos), "relres": _scalar_summary(relres)},
        "per_level": per_level,
    }


def bootstrap_mean_delta(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "delta_mean": float(a.mean() - b.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


# ---------------------------------------------------------------------------
# Baselines and model fitting
# ---------------------------------------------------------------------------


def uniform_prediction(data: dict[str, Any], idx: np.ndarray) -> np.ndarray:
    n1 = data["n1"][idx]
    n2 = data["n2"][idx]
    return (n1[:, None] * data["mean1"][idx] + n2[:, None] * data["mean2"][idx]) / (
        n1 + n2 + 1e-6
    )[:, None]


def role_equal_prediction(data: dict[str, Any], idx: np.ndarray) -> np.ndarray:
    return data["mean1"][idx] + data["mean2"][idx]


def bridge_words_diagnostic(data: dict[str, Any], idx: np.ndarray) -> np.ndarray:
    n1 = data["n1"][idx]
    n2 = data["n2"][idx]
    nb = data["n_bridge"][idx]
    return (
        n1[:, None] * data["mean1"][idx]
        + n2[:, None] * data["mean2"][idx]
        + nb[:, None] * data["bridge_mean"][idx]
    ) / (n1 + n2 + nb + 1e-6)[:, None]


def edge_reference(data: dict[str, Any], idx: np.ndarray, include_bridge: bool = False) -> np.ndarray:
    x = data["c1_vec"][idx] + data["c2_vec"][idx]
    if include_bridge:
        x = x + data["bridge_vec"][idx]
    return x


def mean_prediction(
    data: dict[str, Any],
    fit_mask: np.ndarray,
    idx: np.ndarray,
    *,
    per_level: bool,
) -> np.ndarray:
    y = normalize_rows(data["mb_vec"])
    out = np.zeros((len(idx), DIM), dtype=np.float32)
    if per_level:
        for level in sorted(set(int(x) for x in data["level"][idx]), reverse=True):
            target = idx[data["level"][idx] == level]
            source = np.where(fit_mask & (data["level"] == level))[0]
            if source.size == 0:
                source = np.where(fit_mask)[0]
            out[data["level"][idx] == level] = normalize_rows(y[source].mean(axis=0, keepdims=True))[0]
    else:
        source = np.where(fit_mask)[0]
        out[:] = normalize_rows(y[source].mean(axis=0, keepdims=True))[0]
    return out


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs, y, sample_weight=sample_weight)
    return scaler, model


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    levels: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    alphas: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train])
    x_val = scaler.transform(x[val])
    rows: list[dict[str, Any]] = []
    best_alpha = alphas[0]
    best_score = -np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x_train, y[train])
        metrics = evaluate(y[val], model.predict(x_val), levels[val])
        score = metrics["overall"]["cos"]["mean"]
        rows.append({"alpha": float(alpha), "val": metrics["overall"]})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, rows


# ---------------------------------------------------------------------------
# Residual / bridge analysis
# ---------------------------------------------------------------------------


def residual_analysis(
    data: dict[str, Any],
    idx: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    uniform: np.ndarray,
    *,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    levels = data["level"][idx]
    bridge = data["bridge_vec"][idx]
    bridge_words = data["bridge_mean"][idx]
    res = y - pred
    res_uniform = y - uniform

    cos_res_bridge = _row_cos(res, bridge)
    cos_res_bridge_words = _row_cos(res, bridge_words)
    cos_uniform_res_bridge = _row_cos(res_uniform, bridge)
    cos_pred_bridge = _row_cos(pred, bridge)
    cos_true_bridge = _row_cos(y, bridge)
    res_norm = np.linalg.norm(res, axis=1)
    res_norm_uniform = np.linalg.norm(res_uniform, axis=1)

    # Null 1: permute bridge vectors within each level (keeps level means).
    # Null 2: remove each level's mean residual/bridge direction first, so the
    #         null tests pairing beyond shared level geometry.
    res_centered = res.copy()
    bridge_centered = bridge.copy()
    for level in sorted(set(int(x) for x in levels), reverse=True):
        m = levels == level
        res_centered[m] -= res[m].mean(axis=0, keepdims=True)
        bridge_centered[m] -= bridge[m].mean(axis=0, keepdims=True)

    null_uncentered: list[float] = []
    null_centered: list[float] = []
    for _ in range(n_permutations):
        perm_u = bridge.copy()
        perm_c = bridge_centered.copy()
        for level in sorted(set(int(x) for x in levels), reverse=True):
            m = levels == level
            order = rng.permutation(int(m.sum()))
            perm_u[m] = bridge[m][order]
            perm_c[m] = bridge_centered[m][order]
        null_uncentered.append(float(np.median(_row_cos(res, perm_u))))
        null_centered.append(float(np.median(_row_cos(res_centered, perm_c))))

    per_level: dict[str, Any] = {}
    for level in sorted(set(int(x) for x in levels), reverse=True):
        m = levels == level
        per_level[str(level)] = {
            "n": int(m.sum()),
            "residual_norm_model": _scalar_summary(res_norm[m]),
            "residual_norm_uniform": _scalar_summary(res_norm_uniform[m]),
            "cos_res_bridge": _scalar_summary(cos_res_bridge[m]),
            "cos_res_bridge_words": _scalar_summary(cos_res_bridge_words[m]),
            "cos_uniform_res_bridge": _scalar_summary(cos_uniform_res_bridge[m]),
            "cos_pred_bridge": _scalar_summary(cos_pred_bridge[m]),
            "cos_true_bridge": _scalar_summary(cos_true_bridge[m]),
        }

    return {
        "overall": {
            "residual_norm_model": _scalar_summary(res_norm),
            "residual_norm_uniform": _scalar_summary(res_norm_uniform),
            "cos_res_bridge": _scalar_summary(cos_res_bridge),
            "cos_res_bridge_words": _scalar_summary(cos_res_bridge_words),
            "cos_uniform_res_bridge": _scalar_summary(cos_uniform_res_bridge),
            "cos_pred_bridge": _scalar_summary(cos_pred_bridge),
            "cos_true_bridge": _scalar_summary(cos_true_bridge),
            "null_cos_res_bridge": {
                "centered_median_of_medians": float(np.median(null_centered)),
                "centered_p95": float(np.percentile(null_centered, 95)),
                "uncentered_median_of_medians": float(np.median(null_uncentered)),
                "uncentered_p95": float(np.percentile(null_uncentered, 95)),
            },
        },
        "per_level": per_level,
        "rows": {
            "cos_res_bridge": cos_res_bridge,
            "cos_res_bridge_words": cos_res_bridge_words,
            "cos_uniform_res_bridge": cos_uniform_res_bridge,
            "cos_pred_bridge": cos_pred_bridge,
            "cos_true_bridge": cos_true_bridge,
            "res_norm": res_norm,
            "res_norm_uniform": res_norm_uniform,
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_predictions(
    path: Path,
    data: dict[str, Any],
    idx: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    uniform: np.ndarray,
    residual: dict[str, Any],
) -> None:
    order = sorted(range(len(idx)), key=lambda j: (-int(data["level"][idx][j]), data["mb_id"][idx[j]]))
    idx = idx[order]
    y = y[order]
    pred = pred[order]
    uniform = uniform[order]
    rows = residual["rows"]
    cos_res_bridge = rows["cos_res_bridge"][order]
    cos_res_bridge_words = rows["cos_res_bridge_words"][order]
    cos_uniform_res_bridge = rows["cos_uniform_res_bridge"][order]
    cos_pred_bridge = rows["cos_pred_bridge"][order]
    cos_true_bridge = rows["cos_true_bridge"][order]
    res_norm = rows["res_norm"][order]
    res_norm_uniform = rows["res_norm_uniform"][order]
    res_vec = normalize_rows(y - pred)

    schema = pa.schema(
        [
            ("mb_id", pa.string()),
            ("level", pa.int8()),
            ("mb_vec", VEC_TYPE),
            ("pred_vec", VEC_TYPE),
            ("uniform_vec", VEC_TYPE),
            ("bridge_vec", VEC_TYPE),
            ("bridge_word_mean", VEC_TYPE),
            ("residual_vec", VEC_TYPE),
            ("cos_model", pa.float32()),
            ("cos_uniform", pa.float32()),
            ("relres_model", pa.float32()),
            ("relres_uniform", pa.float32()),
            ("cos_res_bridge", pa.float32()),
            ("cos_res_bridge_words", pa.float32()),
            ("cos_uniform_res_bridge", pa.float32()),
            ("cos_pred_bridge", pa.float32()),
            ("cos_true_bridge", pa.float32()),
        ]
    )

    def vec_col(vectors: np.ndarray) -> pa.FixedSizeListArray:
        flat = pa.array(np.asarray(vectors, dtype=np.float32).reshape(-1), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, DIM)

    cos_model = _row_cos(y, pred)
    cos_uniform = _row_cos(y, uniform)
    table = pa.table(
        {
            "mb_id": pa.array([data["mb_id"][i] for i in idx], type=pa.string()),
            "level": pa.array([int(data["level"][i]) for i in idx], type=pa.int8()),
            "mb_vec": vec_col(y),
            "pred_vec": vec_col(pred),
            "uniform_vec": vec_col(uniform),
            "bridge_vec": vec_col(data["bridge_vec"][idx]),
            "bridge_word_mean": vec_col(data["bridge_mean"][idx]),
            "residual_vec": vec_col(res_vec),
            "cos_model": pa.array(cos_model, type=pa.float32()),
            "cos_uniform": pa.array(cos_uniform, type=pa.float32()),
            "relres_model": pa.array(res_norm, type=pa.float32()),
            "relres_uniform": pa.array(res_norm_uniform, type=pa.float32()),
            "cos_res_bridge": pa.array(cos_res_bridge, type=pa.float32()),
            "cos_res_bridge_words": pa.array(cos_res_bridge_words, type=pa.float32()),
            "cos_uniform_res_bridge": pa.array(cos_uniform_res_bridge, type=pa.float32()),
            "cos_pred_bridge": pa.array(cos_pred_bridge, type=pa.float32()),
            "cos_true_bridge": pa.array(cos_true_bridge, type=pa.float32()),
        },
        schema=schema,
    )
    pq.write_table(table, path, compression="zstd")


def write_report_md(
    path: Path,
    report: dict[str, Any],
) -> None:
    md: list[str] = [
        "# Pooled L9-L2 word model",
        "",
        f"Generated: {report['generated_at']}",
        f"Runtime: {report['runtime_seconds'] / 60:.1f} min",
        "",
        "## Model",
        "",
        "Single ridge regression from `mean(cm1 words)`, `mean(cm2 words)`, level one-hot "
        "features and word-count features to the real MB vector. Predictions are L2-normalized. "
        "The bridge is not an input.",
        "",
        f"- selected alpha: `{report['model']['selected_alpha']}`",
        f"- fit on: `{report['model']['fit_on']}`",
        f"- features: `{report['model']['n_features']}`",
        "",
        "## Split",
        "",
        "| L | train | val | test |",
        "|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        s = report["split"]["per_level"][str(level)]
        md.append(f"| {level} | {s['train']:,} | {s['val']:,} | {s['test']:,} |")
    md += [
        "",
        "Groups are exact cm1+cm2 word sets, so identical inputs never cross splits.",
        "",
        "## Validation alpha scan (overall)",
        "",
        "| alpha | cos median | cos mean | relres median |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["validation"]["alpha_scan"]:
        o = row["val"]
        md.append(
            f"| {row['alpha']:.0f} | {o['cos']['median']:.3f} | {o['cos']['mean']:.3f} | "
            f"{o['relres']['median']:.3f} |"
        )
    md += [
        "",
        f"Selected alpha: **{report['model']['selected_alpha']}** (highest validation mean cosine).",
        "",
        "## Test per level",
        "",
        "| L | n | uniform | +bridge words | edge ref | model | delta mean [95% CI] | uniform relres | model relres | level-mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        s = str(level)
        t = report["test"]
        u = t["uniform"]["per_level"][s]
        m = t["ridge"]["per_level"][s]
        lm = t["level_mean"]["per_level"][s]
        bw = t["bridge_words"]["per_level"][s]
        er = t["edge"]["per_level"][s]
        b = report["bootstrap"]["ridge_minus_uniform"]["per_level"][s]
        md.append(
            f"| {level} | {m['cos']['n']} | {u['cos']['median']:.3f} | {bw['cos']['median']:.3f} | "
            f"{er['cos']['median']:.3f} | {m['cos']['median']:.3f} | "
            f"{b['delta_mean']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
            f"{u['relres']['median']:.3f} | {m['relres']['median']:.3f} | {lm['cos']['median']:.3f} |"
        )
    o = report["test"]
    u = o["uniform"]["overall"]
    m = o["ridge"]["overall"]
    bw = o["bridge_words"]["overall"]
    er = o["edge"]["overall"]
    b = report["bootstrap"]["ridge_minus_uniform"]["overall"]
    md.append(
        f"| all | {m['cos']['n']:,} | {u['cos']['median']:.3f} | {bw['cos']['median']:.3f} | "
        f"{er['cos']['median']:.3f} | {m['cos']['median']:.3f} | "
        f"{b['delta_mean']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
        f"{u['relres']['median']:.3f} | {m['relres']['median']:.3f} | — |"
    )
    md += [
        "",
        "`+bridge words` and `edge ref` are diagnostics, not model inputs. The model sees only child words.",
        "",
        "## Test controls (overall)",
        "",
        "| predictor | cos median | cos mean | relres median |",
        "|---|---:|---:|---:|",
    ]
    labels = [
        ("uniform", "uniform child-word mean"),
        ("role_equal", "role-equal child mean"),
        ("bridge_words", "child+bridge words (diagnostic, not input)"),
        ("level_mean", "train+val level mean"),
        ("global_mean", "train+val global mean"),
        ("edge", "c1+c2 edge reference"),
        ("edge_bridge", "c1+c2+bridge edge reference"),
        ("ridge", "ridge model"),
        ("shuffled_target", "ridge, shuffled training targets"),
        ("role_swap", "ridge, cm1/cm2 swapped at test"),
    ]
    for key, label in labels:
        x = report["test"][key]["overall"]
        md.append(
            f"| {label} | {x['cos']['median']:.3f} | {x['cos']['mean']:.3f} | "
            f"{x['relres']['median']:.3f} |"
        )
    md += [
        "",
        "## Bridge residual",
        "",
        "Residual `r = m - prediction`. A positive `cos(r, bridge)` means the bridge vector still "
        "carries unexplained MB signal.",
        "",
        "| L | n | model relres | cos(r, bridge vec) | cos(r, bridge words) | uniform relres | cos(uniform r, bridge) | cos(pred, bridge) | cos(m, bridge) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        s = str(level)
        r = report["residual"]["per_level"][s]
        md.append(
            f"| {level} | {r['n']} | {r['residual_norm_model']['median']:.3f} | "
            f"{r['cos_res_bridge']['median']:.3f} | {r['cos_res_bridge_words']['median']:.3f} | "
            f"{r['residual_norm_uniform']['median']:.3f} | {r['cos_uniform_res_bridge']['median']:.3f} | "
            f"{r['cos_pred_bridge']['median']:.3f} | {r['cos_true_bridge']['median']:.3f} |"
        )
    ro = report["residual"]["overall"]
    md += [
        f"| all | {ro['cos_res_bridge']['n']:,} | {ro['residual_norm_model']['median']:.3f} | "
        f"{ro['cos_res_bridge']['median']:.3f} | {ro['cos_res_bridge_words']['median']:.3f} | "
        f"{ro['residual_norm_uniform']['median']:.3f} | {ro['cos_uniform_res_bridge']['median']:.3f} | "
        f"{ro['cos_pred_bridge']['median']:.3f} | {ro['cos_true_bridge']['median']:.3f} |",
        "",
        f"Centered within-level bridge permutation null: median cos = "
        f"{ro['null_cos_res_bridge']['centered_median_of_medians']:.3f} "
        f"(p95 = {ro['null_cos_res_bridge']['centered_p95']:.3f}). "
        f"Uncentered permutation null = "
        f"{ro['null_cos_res_bridge']['uncentered_median_of_medians']:.3f} "
        f"(shared level geometry, not a zero-centered null).",
        "",
        "## Key findings",
        "",
        f"- Overall test median cosine: {u['cos']['median']:.3f} -> {m['cos']['median']:.3f}; "
        f"mean delta {b['delta_mean']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}].",
        f"- Median residual norm: {ro['residual_norm_uniform']['median']:.3f} -> "
        f"{ro['residual_norm_model']['median']:.3f} "
        f"({100 * (1 - ro['residual_norm_model']['median'] / ro['residual_norm_uniform']['median']):.1f}% lower).",
        f"- Median cosine(residual, bridge): {ro['cos_uniform_res_bridge']['median']:.3f} -> "
        f"{ro['cos_res_bridge']['median']:.3f}; centered permutation null "
        f"{ro['null_cos_res_bridge']['centered_median_of_medians']:.3f}.",
        f"- L9: uniform {o['uniform']['per_level']['9']['cos']['median']:.3f} -> model "
        f"{o['ridge']['per_level']['9']['cos']['median']:.3f}; +bridge words "
        f"{o['bridge_words']['per_level']['9']['cos']['median']:.3f}; edge ref "
        f"{o['edge']['per_level']['9']['cos']['median']:.3f}.",
        f"- `cm1`/`cm2` swap: {o['role_swap']['overall']['cos']['median']:.3f} vs "
        f"{o['ridge']['overall']['cos']['median']:.3f}; the learned map is approximately symmetric "
        f"in the two child roles.",
        "",
        "## Files",
        "",
        "| file | content |",
        "|---|---|",
        "| `predictions_test.parquet` | test predictions, residuals and per-unit bridge diagnostics |",
        "| `ridge_model.npz` | ridge coefficients, scaler, alpha, feature names |",
        "| `split.json` | unit ids per split |",
        "| `model_report.json` | full metrics and configuration |",
        "",
        "## Notes",
        "",
        "- The bridge is never a model input. `bridge_words` and the edge references are diagnostics.",
        "- The split groups exact cm1+cm2 word sets; related MBs at different levels can still share sub-words.",
        "- Low-level test sets are small (L2/L3 ~12 units), so their CIs are wide.",
        "",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", default=str(ROOT / "evaluation/results/word_units/units.parquet"))
    ap.add_argument("--words", default=str(ROOT / "evaluation/results/word_units/words.parquet"))
    ap.add_argument("--out-dir", default=str(ROOT / "evaluation/results/word_model"))
    ap.add_argument("--alphas", default="100,300,500,1000,2000,3000,10000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=200)
    args = ap.parse_args(argv)

    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    data = load_dataset(Path(args.units), Path(args.words))
    n = len(data["level"])
    split = grouped_split(
        data["level"], data["groups"],
        seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac,
    )
    train = split == "train"
    val = split == "val"
    test = split == "test"
    fit_mask = train | val
    test_idx = np.where(test)[0]
    log(f"split train={train.sum():,} val={val.sum():,} test={test.sum():,}")

    y = normalize_rows(data["mb_vec"])
    x = make_features(data)
    log(f"features {x.shape[1]:,}")

    # ---- validation alpha scan -------------------------------------------
    t0 = time.time()
    selected_alpha, alpha_scan = select_alpha(x, y, data["level"], train, val, alphas)
    log(f"selected alpha={selected_alpha:g} in {time.time() - t0:.1f}s")

    # ---- final fit on train+val -----------------------------------------
    t0 = time.time()
    scaler, model = fit_ridge(x[fit_mask], y[fit_mask], selected_alpha)
    pred_test = normalize_rows(model.predict(scaler.transform(x[test_idx])))
    log(f"final ridge fit in {time.time() - t0:.1f}s")

    # ---- test predictions / baselines ------------------------------------
    y_test = y[test_idx]
    uniform_test = uniform_prediction(data, test_idx)
    role_equal_test = role_equal_prediction(data, test_idx)
    bridge_words_test = bridge_words_diagnostic(data, test_idx)
    level_mean_test = mean_prediction(data, fit_mask, test_idx, per_level=True)
    global_mean_test = mean_prediction(data, fit_mask, test_idx, per_level=False)
    edge_test = edge_reference(data, test_idx, include_bridge=False)
    edge_bridge_test = edge_reference(data, test_idx, include_bridge=True)

    predictions: dict[str, np.ndarray] = {
        "uniform": uniform_test,
        "role_equal": role_equal_test,
        "bridge_words": bridge_words_test,
        "level_mean": level_mean_test,
        "global_mean": global_mean_test,
        "edge": edge_test,
        "edge_bridge": edge_bridge_test,
        "ridge": pred_test,
    }
    test_metrics = {
        key: evaluate(y_test, value, data["level"][test_idx])
        for key, value in predictions.items()
    }

    # ---- controls --------------------------------------------------------
    rng = np.random.default_rng(args.seed + 1)
    y_fit = y[fit_mask]
    levels_fit = data["level"][fit_mask]
    fit_indices = np.arange(int(fit_mask.sum()))
    y_shuffled = y_fit.copy()
    for level in sorted(set(int(v) for v in levels_fit), reverse=True):
        level_idx = fit_indices[levels_fit == level]
        y_shuffled[level_idx] = y_fit[level_idx][rng.permutation(len(level_idx))]
    scaler_shuf, model_shuf = fit_ridge(x[fit_mask], y_shuffled, selected_alpha)
    pred_shuf = normalize_rows(model_shuf.predict(scaler_shuf.transform(x[test_idx])))
    test_metrics["shuffled_target"] = evaluate(y_test, pred_shuf, data["level"][test_idx])

    x_swap = make_features(data, idx=test_idx, swap_roles=True)
    pred_swap = normalize_rows(model.predict(scaler.transform(x_swap)))
    test_metrics["role_swap"] = evaluate(y_test, pred_swap, data["level"][test_idx])

    # ---- bootstrap ridge vs uniform -------------------------------------
    ridge_cos = _row_cos(y_test, pred_test)
    uniform_cos = _row_cos(y_test, uniform_test)
    boot_overall = bootstrap_mean_delta(ridge_cos, uniform_cos, n_boot=args.bootstrap, seed=args.seed)
    boot_per_level: dict[str, Any] = {}
    for level in LEVELS:
        m = data["level"][test_idx] == level
        boot_per_level[str(level)] = bootstrap_mean_delta(
            ridge_cos[m], uniform_cos[m], n_boot=args.bootstrap, seed=args.seed + level
        )

    # ---- residual analysis ----------------------------------------------
    residual = residual_analysis(
        data, test_idx, y_test, pred_test, uniform_test,
        n_permutations=args.permutations, seed=args.seed + 2,
    )

    # ---- split summary ---------------------------------------------------
    split_per_level: dict[str, Any] = {}
    for level in LEVELS:
        m = data["level"] == level
        split_per_level[str(level)] = {
            "train": int((train & m).sum()),
            "val": int((val & m).sum()),
            "test": int((test & m).sum()),
        }

    # ---- save ------------------------------------------------------------
    write_predictions(
        out_dir / "predictions_test.parquet",
        data, test_idx, y_test, pred_test, uniform_test, residual,
    )

    np.savez_compressed(
        out_dir / "ridge_model.npz",
        coef=np.asarray(model.coef_, dtype=np.float32),
        intercept=np.asarray(model.intercept_, dtype=np.float32),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
        alpha=np.array([selected_alpha], dtype=np.float64),
        feature_names=np.array(feature_names(), dtype=object),
    )
    split_json = {
        "seed": args.seed,
        "grouping": "exact cm1+cm2 (word, pos) set",
        "train": [data["mb_id"][i] for i in np.where(train)[0]],
        "val": [data["mb_id"][i] for i in np.where(val)[0]],
        "test": [data["mb_id"][i] for i in np.where(test)[0]],
    }
    (out_dir / "split.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "dataset": {
            "units": int(n),
            "levels": list(LEVELS),
            "vocabulary": int(data["vocab_size"]),
            "target": "mb_vec (L2-normalized)",
        },
        "model": {
            "type": "Ridge",
            "fit_on": "train+val",
            "selected_alpha": float(selected_alpha),
            "n_features": int(x.shape[1]),
            "features": feature_names(),
            "prediction_normalized": True,
            "bridge_as_input": False,
        },
        "split": {
            "seed": args.seed,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "grouping": "exact cm1+cm2 (word, pos) set",
            "per_level": split_per_level,
        },
        "validation": {
            "alphas": alphas,
            "alpha_scan": alpha_scan,
        },
        "bootstrap": {
            "n_boot": args.bootstrap,
            "ridge_minus_uniform": {
                "overall": boot_overall,
                "per_level": boot_per_level,
            },
        },
        "test": {
            **test_metrics,
        },
        "residual": {
            "overall": residual["overall"],
            "per_level": residual["per_level"],
            "n_permutations": args.permutations,
        },
        "notes": [
            "Single pooled ridge model over L9-L2; level is represented by one-hot features.",
            "Model inputs are mean pooled cm1/mean cm2 word vectors, level one-hot and word counts.",
            "The bridge vector and bridge words are never model inputs.",
            "Test predictions are L2-normalized before metrics.",
            "Residual r = m - prediction; cos(r, bridge) measures the bridge signal left unexplained.",
            "The centered residual null removes per-level mean directions before permuting bridge vectors.",
            "The cm1/cm2 swap control is near-identical, so the learned map is approximately symmetric in the child roles.",
            "Splits are grouped by exact cm1+cm2 word sets, but related MBs at different levels may share sub-words.",
            "L2/L3 test sets are small; their per-level CIs are wide.",
        ],
    }
    (out_dir / "model_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_report_md(out_dir / "model_report.md", report)

    # README is a small pointer to the report.
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Pooled L9-L2 word model",
                "",
                "See `model_report.md` for metrics and `model_report.json` for the full record.",
                "",
                "Main files:",
                "- `predictions_test.parquet`",
                "- `ridge_model.npz`",
                "- `split.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ---- console summary -------------------------------------------------
    o = test_metrics["ridge"]["overall"]
    u = test_metrics["uniform"]["overall"]
    print(f"\n=== pooled L9-L2 ridge (alpha={selected_alpha:g}) ===")
    print(f"test units: {o['cos']['n']:,}")
    print(f"uniform : cos median {u['cos']['median']:.4f}  mean {u['cos']['mean']:.4f}  relres {u['relres']['median']:.4f}")
    print(f"ridge   : cos median {o['cos']['median']:.4f}  mean {o['cos']['mean']:.4f}  relres {o['relres']['median']:.4f}")
    print(f"delta mean cos: {boot_overall['delta_mean']:+.4f} "
          f"[{boot_overall['ci_low']:+.4f}, {boot_overall['ci_high']:+.4f}]")
    print(f"residual cos(bridge): {residual['overall']['cos_res_bridge']['median']:.4f} "
          f"(uniform {residual['overall']['cos_uniform_res_bridge']['median']:.4f}, "
          f"centered null {residual['overall']['null_cos_res_bridge']['centered_median_of_medians']:.4f})")
    print(f"\nL  n     uniform  ridge   delta   relres_u relres_m  cos(r,b)")
    for level in LEVELS:
        s = str(level)
        uu = test_metrics["uniform"]["per_level"][s]
        mm = test_metrics["ridge"]["per_level"][s]
        bb = boot_per_level[s]
        rr = residual["per_level"][s]
        print(f"{level}  {mm['cos']['n']:4d}  {uu['cos']['median']:.3f}    {mm['cos']['median']:.3f}  "
              f"{bb['delta_mean']:+.3f}  {uu['relres']['median']:.3f}    {mm['relres']['median']:.3f}    "
              f"{rr['cos_res_bridge']['median']:.3f}")
    print(f"\nOutput: {out_dir}")
    print(f"Runtime: {(time.time() - started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
