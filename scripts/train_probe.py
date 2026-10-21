#!/usr/bin/env python3
"""Train a light learned metric / re-ranker on cached MetaBary vectors.

The probe is deliberately small and CPU-only:

  * vector-only logistic metric on ``m * b`` and ``|m - b|``;
  * vector + geometry logistic metric;
  * gradient-boosted re-ranker on cosine/Jaccard/weight features.

The primary split is grouped by bridge-word hash, so the same bridge word set
never appears in both train and test. A random split is also reported to expose
word leakage. Models are selected on a validation split and evaluated once on
the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exp_bridge import log  # noqa: E402

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def group_key(words_json: str) -> str:
    words = json.loads(words_json)
    return "|".join(sorted(str(w) for w in words))


def group_bucket(words_json: str, seed: int = 0) -> int:
    key = f"{seed}:{group_key(words_json)}"
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def make_grouped_split(bridge_words: np.ndarray, seed: int = 0) -> np.ndarray:
    buckets = np.array([group_bucket(w, seed) for w in bridge_words], dtype=np.int16)
    split = np.full(len(bridge_words), "train", dtype=object)
    split[(buckets >= 70) & (buckets < 85)] = "val"
    split[buckets >= 85] = "test"
    return split


def make_random_split(levels: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.full(len(levels), "train", dtype=object)
    for level in sorted(set(int(x) for x in levels)):
        idx = np.where(levels == level)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(0.15 * n)))
        n_val = max(1, int(round(0.15 * n)))
        if n < 3:
            split[idx] = "train"
            continue
        split[idx[:n_test]] = "test"
        split[idx[n_test:n_test + n_val]] = "val"
    return split


def rank_scores(true: np.ndarray, negs: np.ndarray) -> dict[str, Any]:
    """Rank metrics for true [N] vs negatives [N, K] (higher score is better)."""
    true = np.asarray(true, dtype=np.float64)
    negs = np.asarray(negs, dtype=np.float64)
    if negs.ndim == 1:
        negs = negs[:, None]
    n, k = negs.shape
    higher = (negs > true[:, None]).sum(axis=1)
    ties = (negs == true[:, None]).sum(axis=1)
    rank = 1 + higher
    avg_rank = rank + ties / 2.0
    top1 = (rank == 1).astype(np.float64)
    top1_strict = ((higher == 0) & (ties == 0)).astype(np.float64)
    mrr = 1.0 / rank
    mrr_tie = 1.0 / avg_rank
    margin = true - negs.max(axis=1) if k else np.zeros(n)
    tie_rate = ties / k if k else np.zeros(n)
    return {
        "n": int(n),
        "k": int(k),
        "top1": float(top1.mean()),
        "top1_strict": float(top1_strict.mean()),
        "mrr": float(mrr.mean()),
        "mrr_tie": float(mrr_tie.mean()),
        "tie_rate": float(tie_rate.mean()),
        "margin": float(margin.mean()),
        "per_case_top1_strict": top1_strict,
        "per_case_mrr_tie": mrr_tie,
    }


def public_metrics(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if not k.startswith("per_case_")}


def bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of mean(a) - mean(b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if n == 0 or len(b) != n:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "delta": float(a.mean() - b.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


def flatten_cases(features_true: np.ndarray, features_negs: np.ndarray,
                  mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ft = features_true[mask]
    fn = features_negs[mask]
    n = ft.shape[0]
    k = fn.shape[1]
    x = np.concatenate([ft[:, None, :], fn], axis=1).reshape(n * (k + 1), -1)
    y = np.tile(np.concatenate([[1.0], np.zeros(k, dtype=np.float64)]), n)
    return x, y


def shuffled_case_labels(n_cases: int, k: int, rng: np.random.Generator) -> np.ndarray:
    y = np.zeros((n_cases, k + 1), dtype=np.float64)
    y[np.arange(n_cases), rng.integers(0, k + 1, size=n_cases)] = 1.0
    return y.reshape(-1)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def make_features(data: dict[str, np.ndarray], kind: str,
                  include_level: bool = False) -> tuple[np.ndarray, np.ndarray]:
    m = data["m"].astype(np.float32, copy=False)
    cent = data["centroid"].astype(np.float32, copy=False)
    b_true = data["b_true"].astype(np.float32, copy=False)
    b_negs = data["b_negs"].astype(np.float32, copy=False)

    def cos_pair(name: str) -> tuple[np.ndarray, np.ndarray]:
        return data[f"{name}_true"][:, None], data[f"{name}_negs"][:, :, None]

    ft: list[np.ndarray] = []
    fn: list[np.ndarray] = []

    if kind in ("vector", "vector_geom", "all"):
        ft.append(m * b_true)
        fn.append(m[:, None, :] * b_negs)
        ft.append(np.abs(m - b_true))
        fn.append(np.abs(m[:, None, :] - b_negs))

    if kind in ("cos", "text_geom", "vector_geom", "all"):
        for name in ("cos_m", "cos_cent", "cos_b_c1", "cos_b_c2", "jaccard"):
            a, b = cos_pair(name)
            ft.append(a.astype(np.float32))
            fn.append(b.astype(np.float32))

    if kind in ("text_only", "text_geom", "all"):
        n, k = data["neg_n"].shape
        # Jaccard / overlap are text-only signals; candidate word count is also
        # available without looking at the true bridge weight.
        ft.append(data["jaccard_true"][:, None].astype(np.float32))
        fn.append(data["jaccard_negs"][:, :, None].astype(np.float32))
        ft.append(data["overlap_true"][:, None].astype(np.float32))
        fn.append(data["overlap_negs"][:, :, None].astype(np.float32))
        ft.append(data["bridge_n"][:, None].astype(np.float32))
        fn.append(np.broadcast_to(data["bridge_n"][:, None], (n, k))[:, :, None]
                  .astype(np.float32))
        ft.append(data["child_n"][:, None].astype(np.float32))
        fn.append(np.broadcast_to(data["child_n"][:, None], (n, k))[:, :, None]
                  .astype(np.float32))
        # Candidate word count: for the true bridge this is bridge_n.
        ft.append(data["bridge_n"][:, None].astype(np.float32))
        fn.append(data["neg_n"][:, :, None].astype(np.float32))

    if include_level:
        n, k = data["neg_n"].shape
        ft.append(data["level"][:, None].astype(np.float32))
        fn.append(np.broadcast_to(data["level"][:, None], (n, k))[:, :, None]
                  .astype(np.float32))

    return np.concatenate(ft, axis=1), np.concatenate(fn, axis=2)


FEATURE_SETS = {
    "cos": "cosine/Jaccard (no weight delta)",
    "text_only": "Jaccard/overlap/word-count text only",
    "text_geom": "cosine/Jaccard + overlap/counts (no weight delta)",
    "vector": "m*b and |m-b| (vector-only learned metric)",
    "vector_geom": "vector features + cosine/Jaccard",
    "all": "vector_geom + overlap/counts",
}




# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def _val_key(m: dict[str, Any]) -> tuple[float, float]:
    return (m["top1_strict"], m["mrr_tie"])


def fit_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_val: int,
    k: int,
    *,
    cs: list[float],
    seed: int,
) -> tuple[Any, float, dict[str, Any]]:
    best: tuple[tuple[float, float], float, Any, dict[str, Any]] | None = None
    for c in cs:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            ),
        )
        model.fit(x_train, y_train)
        scores = model.decision_function(x_val)
        metrics = rank_scores(scores[0::(k + 1)], scores.reshape(n_val, k + 1)[:, 1:])
        key = _val_key(metrics)
        if best is None or key > best[0]:
            best = (key, c, model, metrics)
    assert best is not None
    return best[2], best[1], best[3]


def predict_logreg(model: Any, x: np.ndarray) -> np.ndarray:
    return model.decision_function(x)


def fit_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    n_val: int,
    k: int,
    *,
    max_iters: list[int],
    seed: int,
) -> tuple[Any, int, dict[str, Any]]:
    best: tuple[tuple[float, float], int, Any, dict[str, Any]] | None = None
    for n_iter in max_iters:
        model = HistGradientBoostingClassifier(
            max_iter=n_iter,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_val)[:, 1]
        metrics = rank_scores(proba.reshape(n_val, k + 1)[:, 0],
                              proba.reshape(n_val, k + 1)[:, 1:])
        key = _val_key(metrics)
        if best is None or key > best[0]:
            best = (key, n_iter, model, metrics)
    assert best is not None
    return best[2], best[1], best[3]


def predict_hgb(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def evaluate_scores(true: np.ndarray, negs: np.ndarray,
                    mask: np.ndarray, no_overlap: np.ndarray,
                    subset: str,
                    band_tight: np.ndarray | None = None) -> dict[str, Any] | None:
    idx = mask.copy()
    if subset == "no_overlap":
        idx &= no_overlap
    elif subset == "tight":
        # Cases whose negatives were all drawn inside a narrow weight band.
        # Larger negative pools often have to widen the band, so this subset
        # keeps the weight-matched comparison honest.
        if band_tight is None:
            return None
        idx &= band_tight
    if not idx.any():
        return None
    m = rank_scores(true[idx], negs[idx])
    return m


def run_models(data: dict[str, np.ndarray], split: np.ndarray, split_name: str,
               levels: list[int], seed: int,
               only: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}
    n = len(data["level"])
    k = data["b_negs"].shape[1]
    no_overlap = data["no_overlap"].astype(bool)
    band_tight = (data["band_tol"] <= 0.15) if "band_tol" in data else None

    # Baselines are scored on every split/level/subset.
    for lvl in levels + ["all"]:
        base_mask = (split == "test") & ((data["level"] == lvl) if lvl != "all"
                                         else np.ones(n, dtype=bool))
        true_a3 = data["cos_m_true"]
        negs_a3 = data["cos_m_negs"]
        true_a2 = data["cos_cent_true"]
        negs_a2 = data["cos_cent_negs"]
        true_a1 = data["jaccard_true"]
        negs_a1 = data["jaccard_negs"]
        rng = np.random.default_rng(seed)
        true_rand = rng.random(n)
        negs_rand = rng.random((n, k))
        true_w = np.zeros(n)
        negs_w = -data["weight_delta_negs"]
        for name, (t, ng) in {
            "A0_random": (true_rand, negs_rand),
            "A1_jaccard": (true_a1, negs_a1),
            "A2_centroid": (true_a2, negs_a2),
            "A3_m_cosine": (true_a3, negs_a3),
            # Oracle leak: the true bridge is the only candidate with exactly
            # the observed bridge weight. Not a usable selector, but it explains
            # why a weight_delta feature scores 100%.
            "L0_weight_identity_oracle": (true_w, negs_w),
        }.items():
            for subset in ("all", "no_overlap", "tight"):
                m = evaluate_scores(t, ng, base_mask, no_overlap, subset,
                                    band_tight)
                if m is None:
                    continue
                baselines[f"{split_name}/{lvl}/{subset}/{name}"] = m

    configs = [
        # (label, feature kind, scope, model type, include_level)
        ("L9 vector-only logreg", "vector", "9", "logreg", False),
        ("L9 vector+geom logreg", "vector_geom", "9", "logreg", False),
        ("L9 text-only HGB", "text_only", "9", "hgb", False),
        ("L9 text+geom HGB", "text_geom", "9", "hgb", False),
        ("L7 vector-only logreg", "vector", "7", "logreg", False),
        ("L7 vector+geom logreg", "vector_geom", "7", "logreg", False),
        ("L7 text+geom HGB", "text_geom", "7", "hgb", False),
        ("L9+L7 vector+geom logreg", "vector_geom", "combined", "logreg", True),
        ("L9+L7 text+geom HGB", "text_geom", "combined", "hgb", True),
    ]

    feature_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]] = {}

    for label, kind, scope, model_type, include_level in configs:
        if only is not None and label not in only:
            continue
        if split_name == "random" and scope == "7":
            # Random-split comparison is for the primary L9 / combined models only.
            continue
        if split_name == "random" and kind == "text_only":
            continue
        key = (kind, include_level)
        if key not in feature_cache:
            log(f"building features: {kind} include_level={include_level}")
            feature_cache[key] = make_features(data, kind, include_level=include_level)
        ft, fn = feature_cache[key]

        if scope == "combined":
            train_mask = split == "train"
            val_mask = split == "val"
            test_levels = levels
        else:
            lvl = int(scope)
            train_mask = (split == "train") & (data["level"] == lvl)
            val_mask = (split == "val") & (data["level"] == lvl)
            test_levels = [lvl]

        if not train_mask.any() or not val_mask.any():
            log(f"SKIP {label}: empty train/val")
            continue

        x_train, y_train = flatten_cases(ft, fn, train_mask)
        x_val, _ = flatten_cases(ft, fn, val_mask)
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        log(f"fit {label}: train={n_train} val={n_val} features={x_train.shape[1]}")

        if model_type == "logreg":
            model, best_param, val_metrics = fit_logreg(
                x_train, y_train, x_val, n_val, k,
                # The fine sweep showed an interior optimum around 1e-4; the
                # old grid bottomed out at 1e-3.
                cs=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], seed=seed,
            )
            # Refit on train+val with the selected C.
            fit_mask = train_mask | val_mask
            x_fit, y_fit = flatten_cases(ft, fn, fit_mask)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=best_param, class_weight="balanced", solver="liblinear",
                    max_iter=2000, random_state=seed,
                ),
            )
            model.fit(x_fit, y_fit)
            predict = predict_logreg
            param_name = "C"
        else:
            model, best_param, val_metrics = fit_hgb(
                x_train, y_train, x_val, n_val, k,
                max_iters=[50, 100, 200, 400], seed=seed,
            )
            fit_mask = train_mask | val_mask
            x_fit, y_fit = flatten_cases(ft, fn, fit_mask)
            model = HistGradientBoostingClassifier(
                max_iter=best_param, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, early_stopping=False, random_state=seed,
            )
            model.fit(x_fit, y_fit)
            predict = predict_hgb
            param_name = "max_iter"

        result: dict[str, Any] = {
            "label": label,
            "features": kind,
            "feature_description": FEATURE_SETS[kind],
            "scope": scope,
            "model_type": model_type,
            "include_level": include_level,
            "train_cases": n_train,
            "val_cases": n_val,
            "best_param": {param_name: best_param},
            "val": public_metrics(val_metrics),
            "test": {},
        }
        eval_levels = test_levels + (["all"] if scope == "combined" else [])
        for lvl in eval_levels:
            if lvl == "all":
                test_mask = split == "test"
            else:
                test_mask = (split == "test") & (data["level"] == int(lvl))
            if not test_mask.any():
                continue
            x_test, _ = flatten_cases(ft, fn, test_mask)
            scores = predict(model, x_test).reshape(int(test_mask.sum()), k + 1)
            local_mask = np.ones(int(test_mask.sum()), dtype=bool)
            local_no_overlap = no_overlap[test_mask]
            local_band_tight = band_tight[test_mask] if band_tight is not None else None
            for subset in ("all", "no_overlap", "tight"):
                metrics = evaluate_scores(scores[:, 0], scores[:, 1:],
                                          local_mask, local_no_overlap, subset,
                                          local_band_tight)
                if metrics is None:
                    continue
                pub = public_metrics(metrics)
                base = baselines.get(f"{split_name}/{lvl}/{subset}/A3_m_cosine")
                if base is not None:
                    pub["delta_vs_A3"] = bootstrap_delta(
                        metrics["per_case_top1_strict"],
                        base["per_case_top1_strict"],
                        seed=seed,
                    )
                result["test"][f"{lvl}/{subset}"] = pub
        results.append(result)
        summary_key = f"{test_levels[0]}/all"
        log(f"done {label}: {result['test'].get(summary_key, {})}")

    return results, baselines


def control_label_shuffle(data: dict[str, np.ndarray], split: np.ndarray,
                          seed: int) -> dict[str, Any]:
    ft, fn = make_features(data, "vector_geom", include_level=False)
    k = data["b_negs"].shape[1]
    n = len(data["level"])
    train_mask = (split == "train") & (data["level"] == 9)
    val_mask = (split == "val") & (data["level"] == 9)
    test_mask = (split == "test") & (data["level"] == 9)
    if not train_mask.any() or not test_mask.any():
        return {}
    x_train, _ = flatten_cases(ft, fn, train_mask)
    rng = np.random.default_rng(seed)
    y_train = shuffled_case_labels(int(train_mask.sum()), k, rng)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear",
                           max_iter=2000, random_state=seed),
    )
    model.fit(x_train, y_train)
    x_test, _ = flatten_cases(ft, fn, test_mask)
    scores = model.decision_function(x_test).reshape(int(test_mask.sum()), k + 1)
    m = rank_scores(scores[:, 0], scores[:, 1:])
    return {"label_shuffle_L9": public_metrics(m)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "evaluation/results/bridge_probe/dataset.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "evaluation/results/bridge_probe"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--controls", action="store_true", help="run label-shuffle control")
    ap.add_argument("--only", default="",
                    help="comma-separated exact model labels to run (default: all)")
    ap.add_argument("--skip-random", action="store_true",
                    help="skip the random-split leakage comparison")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    levels = [9, 7]
    log(f"loading {args.dataset}")
    with np.load(args.dataset) as z:
        data = {k: z[k] for k in z.files}
    n = len(data["level"])
    k = data["b_negs"].shape[1]
    n_fallback_band = int(np.isnan(data["band_tol"]).sum()) if "band_tol" in data else 0
    log(f"{n} cases, {k} negatives, levels={dict(Counter(int(x) for x in data['level']))}")

    grouped = make_grouped_split(data["bridge_words"], seed=args.seed)
    random_split = make_random_split(data["level"], seed=args.seed)

    def split_stats(split: np.ndarray) -> dict[str, Any]:
        def words_for(idx: np.ndarray) -> set[str]:
            return {w for s in data["bridge_words"][idx] for w in json.loads(s)}

        train_idx = np.where(split == "train")[0]
        train_words = words_for(train_idx)
        out: dict[str, Any] = {}
        for name in ("train", "val", "test"):
            idx = np.where(split == name)[0]
            words = words_for(idx)
            out[name] = {
                "cases": int(len(idx)),
                "per_level": dict(Counter(int(x) for x in data["level"][idx])),
                # Exact bridge-word-set identity overlap: grouped split is 0.
                "bridge_group_overlap_with_train": int(len(
                    set(data["bridge_words"][idx]) & set(data["bridge_words"][train_idx])
                )) if name != "train" else 0,
                # Token-level overlap can be nonzero even for disjoint word sets;
                # random split should be higher than grouped.
                "bridge_token_overlap_with_train": (
                    len(words & train_words) / len(words) if words and name != "train" else 0.0
                ),
            }
        return out

    log("grouped split: " + json.dumps(split_stats(grouped)))
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    grouped_results, grouped_baselines = run_models(
        data, grouped, "grouped", levels, args.seed, only=only,
    )

    random_results: list[dict[str, Any]] = []
    random_baselines: dict[str, Any] = {}
    if not args.skip_random:
        log("random split: " + json.dumps(split_stats(random_split)))
        random_results, random_baselines = run_models(
            data, random_split, "random", levels, args.seed, only=only,
        )

    controls: dict[str, Any] = {}
    if args.controls:
        log("label-shuffle control")
        controls.update(control_label_shuffle(data, grouped, args.seed))

    # Prune per-case arrays before writing JSON.
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.time() - started,
            "dataset": args.dataset,
            "cases": n,
            "negatives": k,
            "levels": levels,
            "seed": args.seed,
            "fallback_band_cases": n_fallback_band,
            "grouped_split": split_stats(grouped),
            "random_split": split_stats(random_split) if not args.skip_random else None,
            "feature_sets": FEATURE_SETS,
        },
        "grouped": {
            "baselines": {key: public_metrics(v) for key, v in grouped_baselines.items()},
            "models": grouped_results,
        },
        "random": {
            "baselines": {key: public_metrics(v) for key, v in random_baselines.items()},
            "models": random_results,
        },
        "controls": controls,
        "notes": [
            "Grouped split hashes the sorted bridge-word set; exact bridge-word "
            "sets are disjoint between train and test. Token-level overlap is "
            "reported because different bridge-word sets can still share tokens. "
            "Random split is a secondary stability check; bridge-word sets are "
            "nearly unique here, so it is not a strong leakage stress test.",
            "All cases are clean (non-degenerate), weight-band matched, and exclude "
            "near-exact duplicate bridges (cos > 0.9999 to the true bridge).",
            "weight_delta is excluded from all model features. The true bridge is "
            "the only candidate with delta exactly 0, so a weight_delta feature is "
            "an oracle identifier (L0_weight_identity_oracle); it is retained only "
            "as a leakage diagnostic, not as a selector.",
            "A1 Jaccard is tie-dominated at L9/L7, so strict top-1 and MRR_tie are "
            "the relevant columns.",
            "The vector-only model uses m*b and |m-b|. It does not see words, "
            "child vectors, weights, or bridge ids.",
            "text_geom is the stronger text/geometry re-ranker baseline; text_only "
            "isolates word overlap/counts without any vector or child geometry.",
            f"{n_fallback_band}/{n} cases had no candidate inside the initial "
            "weight band and used the final fallback; weight features are excluded.",
            "Models are selected on validation cases and refit on train+val; test "
            "is evaluated once.",
        ],
    }

    json_path = out_dir / "probe_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"wrote {json_path}")

    # Console tables.
    def print_split(title: str, split_name: str, baselines: dict[str, Any],
                    models: list[dict[str, Any]]):
        for lvl in (9, 7):
            probe_key = f"{split_name}/{lvl}/all/A3_m_cosine"
            if probe_key not in baselines:
                continue
            print(f"\n=== {title}: L{lvl} test ===")
            print(f"{'model':<32} {'n':>4} {'top1':>7} {'strict':>7} {'MRRt':>7} {'dA3':>8}")
            for name in ("A0_random", "A1_jaccard", "A2_centroid", "A3_m_cosine"):
                m = baselines.get(f"{split_name}/{lvl}/all/{name}")
                if m:
                    print(f"{name:<32} {m['n']:>4} {m['top1']:>6.1%} "
                          f"{m['top1_strict']:>6.1%} {m['mrr_tie']:>7.3f} {'':>8}")
            for res in models:
                if res["scope"] not in (str(lvl), "combined"):
                    continue
                m = res["test"].get(f"{lvl}/all")
                if not m:
                    continue
                delta = m.get("delta_vs_A3", {}).get("delta")
                delta_s = f"{delta:+.3f}" if delta is not None else ""
                print(f"{res['label']:<32} {m['n']:>4} {m['top1']:>6.1%} "
                      f"{m['top1_strict']:>6.1%} {m['mrr_tie']:>7.3f} {delta_s:>8}")

    print_split("Grouped split", "grouped", grouped_baselines, grouped_results)
    if random_results:
        print_split("Random split", "random", random_baselines, random_results)
    if controls:
        print("\n=== Controls ===")
        for name, m in controls.items():
            print(f"{name}: top1={m['top1']:.1%} strict={m['top1_strict']:.1%} "
                  f"MRRt={m['mrr_tie']:.3f} n={m['n']}")
    print(f"\nReport: {json_path}")
    print(f"Runtime: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
