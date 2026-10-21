#!/usr/bin/env python3
"""Fine L9 regularization sweep for the light vector probe models.

The main probe grid bottoms out at ``C=0.001`` on every L9 model, so the
selected value sits on the grid boundary. This script trains the two L9
vector models across a wider, denser ``C`` grid and reports validation and
test rank metrics for every value.

Model selection in the main probe uses validation cases only. This sweep is
diagnostic: it shows the shape of the validation/test curve so we can tell
whether the boundary choice is an artefact of the coarse grid. The test
columns are reported for transparency, not used to pick the production model.

CPU only; reads the cached dataset and writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exp_bridge import log  # noqa: E402
from scripts.train_probe import (  # noqa: E402
    bootstrap_delta,
    flatten_cases,
    make_features,
    make_grouped_split,
    public_metrics,
    rank_scores,
)

DEFAULT_CS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

KINDS = {
    "vector": "m*b and |m-b| (vector-only learned metric)",
    "vector_geom": "vector features + cosine/Jaccard",
}


def fit_model(x: np.ndarray, y: np.ndarray, c: float, seed: int):
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
    model.fit(x, y)
    return model


def score(model, x: np.ndarray, n_cases: int, k: int) -> dict[str, Any]:
    scores = model.decision_function(x).reshape(n_cases, k + 1)
    return rank_scores(scores[:, 0], scores[:, 1:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "evaluation/results/bridge_probe/dataset.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "evaluation/results/bridge_probe/c_sweep"))
    ap.add_argument("--levels", default="9")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cs", default=",".join(str(c) for c in DEFAULT_CS))
    ap.add_argument("--kinds", default="vector,vector_geom")
    args = ap.parse_args(argv)

    cs = [float(x) for x in args.cs.split(",") if x.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    kinds = [x.strip() for x in args.kinds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with np.load(args.dataset) as z:
        data = {key: z[key] for key in z.files}
    k = data["b_negs"].shape[1]
    split = make_grouped_split(data["bridge_words"], seed=args.seed)
    log(f"{len(data['level'])} cases, k={k}, seed={args.seed}, cs={cs}")

    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "seed": args.seed,
            "levels": levels,
            "negatives": k,
            "cs": cs,
            "kinds": {kind: KINDS[kind] for kind in kinds},
        },
        "levels": {},
    }

    for level in levels:
        train_mask = (split == "train") & (data["level"] == level)
        val_mask = (split == "val") & (data["level"] == level)
        test_mask = (split == "test") & (data["level"] == level)
        fit_mask = train_mask | val_mask
        n_train, n_val, n_test = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
        a3 = rank_scores(data["cos_m_true"][test_mask], data["cos_m_negs"][test_mask])
        log(f"L{level}: train={n_train} val={n_val} test={n_test}")

        level_out: dict[str, Any] = {
            "cases": {"train": n_train, "val": n_val, "test": n_test},
            "baseline_A3": public_metrics(a3),
            "kinds": {},
        }

        for kind in kinds:
            ft, fn = make_features(data, kind)
            x_train, y_train = flatten_cases(ft, fn, train_mask)
            x_val, _ = flatten_cases(ft, fn, val_mask)
            x_fit, y_fit = flatten_cases(ft, fn, fit_mask)
            x_test, _ = flatten_cases(ft, fn, test_mask)

            rows = []
            for c in cs:
                t0 = time.time()
                # Validation curve: train-only fit, matching the main probe's
                # model-selection step.
                m_val = fit_model(x_train, y_train, c, args.seed)
                val_metrics = score(m_val, x_val, n_val, k)
                # Test curve: refit on train+val, matching the final model.
                m_fit = fit_model(x_fit, y_fit, c, args.seed)
                test_metrics = score(m_fit, x_test, n_test, k)
                row = {
                    "C": c,
                    "val": public_metrics(val_metrics),
                    "test": public_metrics(test_metrics),
                    "test_delta_vs_A3": bootstrap_delta(
                        test_metrics["per_case_top1_strict"],
                        a3["per_case_top1_strict"],
                        seed=args.seed,
                    ),
                    "fit_seconds": round(time.time() - t0, 1),
                }
                rows.append(row)
                log(f"L{level} {kind} C={c:g}: val strict={row['val']['top1_strict']:.3f} "
                    f"test strict={row['test']['top1_strict']:.3f}")

            best_val = max(rows, key=lambda r: (r["val"]["top1_strict"],
                                               r["val"]["mrr_tie"]))
            best_test = max(rows, key=lambda r: (r["test"]["top1_strict"],
                                                r["test"]["mrr_tie"]))
            level_out["kinds"][kind] = {
                "description": KINDS[kind],
                "rows": rows,
                "best_by_val": {"C": best_val["C"],
                                "val": best_val["val"],
                                "test": best_val["test"],
                                "test_delta_vs_A3": best_val["test_delta_vs_A3"]},
                "best_by_test_oracle": {"C": best_test["C"],
                                        "test": best_test["test"]},
            }
            log(f"L{level} {kind}: best by val C={best_val['C']:g} "
                f"(test strict={best_val['test']['top1_strict']:.3f}); "
                f"best on test C={best_test['C']:g} "
                f"(strict={best_test['test']['top1_strict']:.3f})")

        report["levels"][str(level)] = level_out

    report["meta"]["runtime_seconds"] = time.time() - started
    path = out_dir / f"c_sweep_seed{args.seed}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {path}")

    for level in levels:
        for kind in kinds:
            out = report["levels"][str(level)]["kinds"][kind]
            print(f"\n=== L{level} {kind} ===")
            print(f"{'C':>10} {'val strict':>10} {'val MRRt':>9} "
                  f"{'test strict':>11} {'test MRRt':>10} {'dA3':>7}")
            for row in out["rows"]:
                d = row["test_delta_vs_A3"]["delta"]
                print(f"{row['C']:>10g} {row['val']['top1_strict']:>10.3f} "
                      f"{row['val']['mrr_tie']:>9.3f} {row['test']['top1_strict']:>11.3f} "
                      f"{row['test']['mrr_tie']:>10.3f} {d:>+7.3f}")
    print(f"\nReport: {path}")
    print(f"Runtime: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
