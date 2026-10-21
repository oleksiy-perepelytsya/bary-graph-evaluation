#!/usr/bin/env python3
"""Build a cached 10-way bridge-selection dataset for probe training.

For each clean MetaBary triad we store:

  * m, the child centroid, the true bridge, and 9 hard near-neighbour
    candidate bridges (all weight-band matched, no exact duplicate of the
    true bridge);
  * precomputed cosine / Jaccard / overlap features for baselines and
    re-rankers;
  * bridge words and child words so future text baselines can be added
    without re-querying Mongo.

The output is a single ``dataset.npz`` plus ``dataset_meta.json``. Mongo is
read-only; this script writes only under ``evaluation/results/bridge_probe``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.db import get_collection  # noqa: E402
from lib.vector import cosine, normalize  # noqa: E402
from scripts.exp_bridge import (  # noqa: E402
    WordCache,
    build_near_pool,
    build_triads,
    jaccard,
    log,
    select_negatives,
)


def _json_words(words) -> str:
    return json.dumps(sorted(str(w) for w in words), ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="9,7",
                    help="MetaBary levels to cache (default 9,7)")
    ap.add_argument("--n", type=int, default=2000,
                    help="max MBs per level; n >= count means all (default 2000)")
    ap.add_argument("--negatives", type=int, default=9,
                    help="hard negatives per case (default 9)")
    ap.add_argument("--near-limit", type=int, default=100,
                    help="ANN candidates fetched per triad (default 100)")
    ap.add_argument("--max-words", type=int, default=8)
    ap.add_argument("--scan-cap", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "evaluation/results/bridge_probe"))
    args = ap.parse_args(argv)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    started = time.time()

    log(f"connecting; levels={levels} n={args.n} negatives={args.negatives}")
    coll = get_collection()
    try:
        coll.database.client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: MongoDB unreachable: {exc}")
        return 2

    wc = WordCache(coll, max_words=args.max_words)
    cases: list[dict[str, Any]] = []
    data_stats: dict[str, Any] = {}

    for level in levels:
        level_start = len(cases)
        log(f"L{level}: extracting up to {args.n} MBs")
        triads, stats = build_triads(
            coll, level, args.n, rng, wc, scan_cap=args.scan_cap,
        )
        clean = [t for t in triads if t.clean]
        log(f"L{level}: {len(triads)} triads, {len(clean)} clean")

        skipped = Counter()
        tolerated: Counter = Counter()
        for idx, triad in enumerate(clean, start=1):
            near = build_near_pool(coll, triad, limit=args.near_limit)
            if not near.candidates:
                skipped["empty_near_pool"] += 1
                continue
            negs, tol = select_negatives(
                near, triad, args.negatives, rng,
                word_cache=wc,
                fill_words=True,
                exclude_duplicate_bridge=True,
            )
            if len(negs) < args.negatives:
                skipped[f"only_{len(negs)}_negatives"] += 1
                continue
            tolerated[str(tol)] += 1

            centroid = normalize(triad.c1 + triad.c2)
            child_words = triad.child_words
            bridge_words = triad.bridge_words
            neg_words = [set(n.words) for n in negs]

            cases.append({
                "m": triad.m.astype(np.float32),
                "centroid": centroid.astype(np.float32),
                "b_true": triad.b.astype(np.float32),
                "b_negs": np.stack([n.vec for n in negs]).astype(np.float32),
                "w_true": float(triad.w3),
                "w_negs": np.array([n.weight for n in negs], dtype=np.float32),
                "level": int(level),
                "bridge_id": str(triad.b_id),
                "triad_id": str(triad.mb_id),
                "neg_ids": [str(n.id) for n in negs],
                "edge_type": str(triad.bridge_edge_type),
                "no_overlap": bool(triad.no_overlap),
                "recon_error": float(triad.recon_error),
                "band_tol": float(tol) if np.isfinite(tol) else None,
                "bridge_words": bridge_words,
                "child_words": child_words,
                "neg_words": neg_words,
                "jaccard_true": jaccard(bridge_words, child_words),
                "jaccard_negs": np.array(
                    [jaccard(w, child_words) for w in neg_words], dtype=np.float32
                ),
                "overlap_true": len(set(bridge_words) & set(child_words)),
                "overlap_negs": np.array(
                    [len(set(w) & set(child_words)) for w in neg_words], dtype=np.float32
                ),
                "bridge_n": len(bridge_words),
                "child_n": len(child_words),
                "neg_n": np.array([len(w) for w in neg_words], dtype=np.float32),
                "cos_m_true": cosine(triad.m, triad.b),
                "cos_m_negs": np.array(
                    [cosine(triad.m, n.vec) for n in negs], dtype=np.float32
                ),
                "cos_cent_true": cosine(centroid, triad.b),
                "cos_cent_negs": np.array(
                    [cosine(centroid, n.vec) for n in negs], dtype=np.float32
                ),
                "cos_b_c1_true": cosine(triad.b, triad.c1),
                "cos_b_c1_negs": np.array(
                    [cosine(n.vec, triad.c1) for n in negs], dtype=np.float32
                ),
                "cos_b_c2_true": cosine(triad.b, triad.c2),
                "cos_b_c2_negs": np.array(
                    [cosine(n.vec, triad.c2) for n in negs], dtype=np.float32
                ),
                "weight_delta_true": 0.0,
                "weight_delta_negs": np.array(
                    [abs(n.weight - triad.w3) for n in negs], dtype=np.float32
                ),
            })
            if idx % 50 == 0:
                log(f"L{level}: cached {len(cases)} cases ({idx}/{len(clean)}) "
                    f"word-cache {wc.hits} hits/{wc.misses} misses")

        stats["clean"] = len(clean)
        stats["cached_cases"] = len(cases) - level_start
        stats["skipped"] = dict(skipped)
        stats["band_tolerances"] = dict(tolerated)
        data_stats[str(level)] = stats
        log(f"L{level}: cached={len(cases) - level_start} skipped={dict(skipped)}")

    if not cases:
        log("ERROR: no cases built")
        return 1

    N = len(cases)
    K = args.negatives
    D = int(cases[0]["m"].size)
    log(f"stacking {N} cases x (1+{K}) candidates x {D} dims")

    def stack(key: str) -> np.ndarray:
        return np.stack([c[key] for c in cases]).astype(np.float32)

    arr = {
        "m": stack("m"),
        "centroid": stack("centroid"),
        "b_true": stack("b_true"),
        "b_negs": stack("b_negs"),
        "w_true": np.array([c["w_true"] for c in cases], dtype=np.float32),
        "w_negs": stack("w_negs"),
        "level": np.array([c["level"] for c in cases], dtype=np.int16),
        "no_overlap": np.array([c["no_overlap"] for c in cases], dtype=bool),
        "recon_error": np.array([c["recon_error"] for c in cases], dtype=np.float32),
        "band_tol": np.array(
            [np.nan if c["band_tol"] is None else c["band_tol"] for c in cases],
            dtype=np.float32,
        ),
        "jaccard_true": np.array([c["jaccard_true"] for c in cases], dtype=np.float32),
        "jaccard_negs": stack("jaccard_negs"),
        "overlap_true": np.array([c["overlap_true"] for c in cases], dtype=np.float32),
        "overlap_negs": stack("overlap_negs"),
        "bridge_n": np.array([c["bridge_n"] for c in cases], dtype=np.float32),
        "child_n": np.array([c["child_n"] for c in cases], dtype=np.float32),
        "neg_n": stack("neg_n"),
        "cos_m_true": np.array([c["cos_m_true"] for c in cases], dtype=np.float32),
        "cos_m_negs": stack("cos_m_negs"),
        "cos_cent_true": np.array([c["cos_cent_true"] for c in cases], dtype=np.float32),
        "cos_cent_negs": stack("cos_cent_negs"),
        "cos_b_c1_true": np.array([c["cos_b_c1_true"] for c in cases], dtype=np.float32),
        "cos_b_c1_negs": stack("cos_b_c1_negs"),
        "cos_b_c2_true": np.array([c["cos_b_c2_true"] for c in cases], dtype=np.float32),
        "cos_b_c2_negs": stack("cos_b_c2_negs"),
        "weight_delta_true": np.array(
            [c["weight_delta_true"] for c in cases], dtype=np.float32
        ),
        "weight_delta_negs": stack("weight_delta_negs"),
        "bridge_id": np.array([c["bridge_id"] for c in cases], dtype=str),
        "triad_id": np.array([c["triad_id"] for c in cases], dtype=str),
        "neg_ids": np.array([c["neg_ids"] for c in cases], dtype=str),
        "edge_type": np.array([c["edge_type"] for c in cases], dtype=str),
        "bridge_words": np.array([_json_words(c["bridge_words"]) for c in cases], dtype=str),
        "child_words": np.array([_json_words(c["child_words"]) for c in cases], dtype=str),
        "neg_words": np.array(
            [json.dumps([sorted(w) for w in c["neg_words"]], ensure_ascii=False)
             for c in cases],
            dtype=str,
        ),
    }

    npz_path = out_dir / "dataset.npz"
    np.savez(npz_path, **arr)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "levels": levels,
        "n_requested": args.n,
        "negatives": K,
        "near_limit": args.near_limit,
        "seed": args.seed,
        "max_words": args.max_words,
        "cases": N,
        "cases_per_level": dict(Counter(int(x) for x in arr["level"])),
        "no_overlap_cases": int(arr["no_overlap"].sum()),
        "recon_error_max": float(arr["recon_error"].max()),
        "word_cache_hits": wc.hits,
        "word_cache_misses": wc.misses,
        "data": data_stats,
        "mongo_db": coll.database.name,
        "mongo_collection": coll.name,
    }
    meta_path = out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log(f"wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    log(f"wrote {meta_path}")
    log(f"cases={N} per_level={meta['cases_per_level']} "
        f"no_overlap={meta['no_overlap_cases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
