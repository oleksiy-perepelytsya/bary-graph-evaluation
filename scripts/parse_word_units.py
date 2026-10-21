#!/usr/bin/env python3
"""Phase 1 — parse L2–L9 MetaBary units into a word-level dataset.

For every non-structural baryedge at the requested levels we store:

  * the real MB vector ``mb_vec``;
  * the two child edge vectors ``c1_vec`` / ``c2_vec``;
  * the bridge edge vector ``bridge_vec`` (stored for residual/oracle analysis,
    never as a model input);
  * the unique ``(word, pos)`` keys reachable through cm1, cm2 and the bridge.

Word vectors are kept in a separate vocabulary table.  The canonical vector for
a ``(word, pos)`` key is the mean of its L14 word-node vectors when at least one
L14 node is reachable; otherwise it is the mean of its L15 sense vectors.

MongoDB is read-only.  Outputs go to ``evaluation/results/word_units``:

  units.parquet   one row per MB
  words.parquet   one row per (word, pos)
  parse_report.json
  README.md

Usage:
    python scripts/parse_word_units.py --levels 9-2 --workers 8
    python scripts/parse_word_units.py --levels 9 --limit 200   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pymongo.errors import PyMongoError
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.db import chunks, get_collection  # noqa: E402
from lib.vector import cosine, normalize, unpack_vec  # noqa: E402

DIM = 1024
VEC_TYPE = pa.list_(pa.float32(), DIM)

MB_PROJ = {"_id": 1, "level": 1, "cm1_id": 1, "cm2_id": 1, "vector": 1}
BRIDGE_PROJ = {"_id": 1, "level": 1, "vector": 1, "parent_edge_id": 1}
CHILD_PROJ = {"_id": 1, "level": 1, "vector": 1}
TERM_PROJ = {
    "_id": 1,
    "doc_type": 1,
    "level": 1,
    "cm1_id": 1,
    "cm2_id": 1,
    "properties": 1,
    "vector": 1,
}

UNIT_SCHEMA = pa.schema(
    [
        ("mb_id", pa.string()),
        ("level", pa.int8()),
        ("mb_vec", VEC_TYPE),
        ("c1_id", pa.string()),
        ("c2_id", pa.string()),
        ("c1_vec", VEC_TYPE),
        ("c2_vec", VEC_TYPE),
        ("bridge_vec", VEC_TYPE),
        ("cm1_words", pa.list_(pa.string())),
        ("cm1_pos", pa.list_(pa.string())),
        ("cm2_words", pa.list_(pa.string())),
        ("cm2_pos", pa.list_(pa.string())),
        ("bridge_words", pa.list_(pa.string())),
        ("bridge_pos", pa.list_(pa.string())),
        ("n_cm1", pa.int32()),
        ("n_cm2", pa.int32()),
        ("n_bridge", pa.int32()),
        ("n_words_total", pa.int32()),
    ]
)

WORD_SCHEMA = pa.schema(
    [
        ("word", pa.string()),
        ("pos", pa.string()),
        ("lang", pa.string()),
        ("vec", VEC_TYPE),
        ("source_level", pa.int8()),
        ("n_nodes", pa.int32()),
        ("n_l14", pa.int32()),
        ("n_l15", pa.int32()),
    ]
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def parse_levels(spec: str) -> list[int]:
    levels: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            levels.update(range(min(lo_i, hi_i), max(lo_i, hi_i) + 1))
        else:
            levels.add(int(part))
    return sorted(levels, reverse=True)


def vec_of(doc: dict[str, Any] | None) -> np.ndarray | None:
    if not doc:
        return None
    blob = doc.get("vector")
    if blob is None:
        return None
    try:
        vec = unpack_vec(blob)
    except Exception:  # noqa: BLE001
        return None
    if vec.ndim != 1 or vec.size != DIM or not np.all(np.isfinite(vec)):
        return None
    return vec.astype(np.float32, copy=False)


def fetch_map(coll, ids: list[Any], projection: dict[str, int], chunk: int = 500) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for part in chunks(ids, chunk):
        for doc in coll.find({"_id": {"$in": part}}, projection):
            out[doc["_id"]] = doc
    return out


def bfs_terms(coll, root: Any, max_depth: int = 20) -> list[dict[str, Any]]:
    """Terminal node documents reachable from a baryedge root."""
    frontier: set[Any] = {root}
    visited: set[Any] = set()
    terms: dict[Any, dict[str, Any]] = {}
    for _ in range(max_depth):
        to_fetch = [oid for oid in frontier if oid not in visited]
        if not to_fetch:
            break
        visited.update(to_fetch)
        next_frontier: set[Any] = set()
        for part in chunks(to_fetch, 500):
            for doc in coll.find({"_id": {"$in": part}}, TERM_PROJ):
                if doc.get("doc_type") == "node":
                    terms[doc["_id"]] = doc
                else:
                    if doc.get("cm1_id") is not None:
                        next_frontier.add(doc["cm1_id"])
                    if doc.get("cm2_id") is not None:
                        next_frontier.add(doc["cm2_id"])
        frontier = next_frontier
    return list(terms.values())


def _mix_word(
    agg: dict[tuple[str, str], dict[str, Any]],
    key: tuple[str, str],
    vec: np.ndarray,
    level: int,
    lang: str,
) -> None:
    """Accumulate the preferred vector for one (word, pos) key.

    L14 word vectors win over L15 sense vectors.  Only one sum is kept: when an
    L14 vector appears after L15 vectors, the sum is replaced.  Counts of both
    node kinds are retained for the report.
    """
    rec = agg.get(key)
    if rec is None:
        agg[key] = {
            "level": level,
            "n": 1,
            "sum": vec.astype(np.float32, copy=True),
            "n_l14": 1 if level == 14 else 0,
            "n_l15": 1 if level == 15 else 0,
            "lang": lang,
        }
        return

    if level == rec["level"]:
        rec["n"] += 1
        rec["sum"] += vec
        if level == 14:
            rec["n_l14"] += 1
        else:
            rec["n_l15"] += 1
    elif level == 14:
        # L14 wins: drop the L15 sum, keep its occurrence count.
        rec["level"] = 14
        rec["n"] = 1
        rec["sum"] = vec.astype(np.float32, copy=True)
        rec["n_l14"] = 1
        if not rec.get("lang"):
            rec["lang"] = lang
    else:
        # L14 already present; keep the L15 occurrence count only.
        rec["n_l15"] += 1


def role_words(
    terms: list[dict[str, Any]],
    agg: dict[tuple[str, str], dict[str, Any]],
    stats: Counter,
) -> tuple[list[str], list[str], set[tuple[str, str]]]:
    """Unique (word, pos) keys for one MB role, sorted deterministically."""
    seen: set[tuple[str, str]] = set()
    for doc in terms:
        props = doc.get("properties") or {}
        word = props.get("word")
        if not word:
            stats["missing_word"] += 1
            continue
        word = str(word)
        pos = str(props.get("pos") or "")
        key = (word, pos)
        vec = vec_of(doc)
        if vec is None:
            stats["missing_word_vec"] += 1
            continue
        _mix_word(agg, key, vec, int(doc.get("level") or 0), str(props.get("lang") or ""))
        seen.add(key)
    pairs = sorted(seen)
    return [w for w, _ in pairs], [p for _, p in pairs], seen


def merge_agg(
    target: dict[tuple[str, str], dict[str, Any]],
    local: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for key, rec in local.items():
        cur = target.get(key)
        if cur is None:
            target[key] = rec
            continue
        if rec["level"] == 14 and cur["level"] == 15:
            rec["n_l15"] = rec.get("n_l15", 0) + cur.get("n_l15", 0)
            target[key] = rec
        elif rec["level"] == 15 and cur["level"] == 14:
            cur["n_l15"] += rec.get("n_l15", 0)
        elif rec["level"] == cur["level"]:
            cur["n"] += rec["n"]
            cur["sum"] += rec["sum"]
            cur["n_l14"] += rec.get("n_l14", 0)
            cur["n_l15"] += rec.get("n_l15", 0)
            if not cur.get("lang"):
                cur["lang"] = rec.get("lang", "")


def process_chunk(
    mbs: list[dict[str, Any]],
    bridge_by_mb: dict[Any, dict[str, Any]],
    child_by_id: dict[Any, dict[str, Any]],
    coll,
    max_depth: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], Counter]:
    units: list[dict[str, Any]] = []
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    drops: Counter = Counter()

    for mb in mbs:
        level = int(mb["level"])
        bridge = bridge_by_mb.get(mb["_id"])
        c1 = child_by_id.get(mb.get("cm1_id"))
        c2 = child_by_id.get(mb.get("cm2_id"))
        if bridge is None:
            drops["missing_bridge"] += 1
            continue
        if c1 is None or c2 is None:
            drops["missing_child"] += 1
            continue

        m_vec = vec_of(mb)
        c1_vec = vec_of(c1)
        c2_vec = vec_of(c2)
        b_vec = vec_of(bridge)
        if m_vec is None or c1_vec is None or c2_vec is None or b_vec is None:
            drops["missing_or_bad_vector"] += 1
            continue

        cm1_words, cm1_pos, keys1 = role_words(bfs_terms(coll, c1["_id"], max_depth), agg, drops)
        cm2_words, cm2_pos, keys2 = role_words(bfs_terms(coll, c2["_id"], max_depth), agg, drops)
        br_words, br_pos, keysb = role_words(bfs_terms(coll, bridge["_id"], max_depth), agg, drops)

        if not cm1_words:
            drops["empty_cm1"] += 1
            continue
        if not cm2_words:
            drops["empty_cm2"] += 1
            continue
        if not br_words:
            drops["empty_bridge"] += 1
            continue

        units.append(
            {
                "mb_id": str(mb["_id"]),
                "level": level,
                "mb_vec": m_vec,
                "c1_id": str(c1["_id"]),
                "c2_id": str(c2["_id"]),
                "c1_vec": c1_vec,
                "c2_vec": c2_vec,
                "bridge_vec": b_vec,
                "cm1_words": cm1_words,
                "cm1_pos": cm1_pos,
                "cm2_words": cm2_words,
                "cm2_pos": cm2_pos,
                "bridge_words": br_words,
                "bridge_pos": br_pos,
                "n_cm1": len(cm1_words),
                "n_cm2": len(cm2_words),
                "n_bridge": len(br_words),
                "n_words_total": len(keys1 | keys2 | keysb),
            }
        )

    return units, agg, drops


# ---------------------------------------------------------------------------
# Parquet output
# ---------------------------------------------------------------------------


def fixed_vec_array(vectors: list[np.ndarray]) -> pa.FixedSizeListArray:
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in vectors]).astype(np.float32, copy=False)
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, DIM)


def _unit_table(batch: list[dict[str, Any]]) -> pa.Table:
    cols = {
        "mb_id": pa.array([r["mb_id"] for r in batch], type=pa.string()),
        "level": pa.array([r["level"] for r in batch], type=pa.int8()),
        "mb_vec": fixed_vec_array([r["mb_vec"] for r in batch]),
        "c1_id": pa.array([r["c1_id"] for r in batch], type=pa.string()),
        "c2_id": pa.array([r["c2_id"] for r in batch], type=pa.string()),
        "c1_vec": fixed_vec_array([r["c1_vec"] for r in batch]),
        "c2_vec": fixed_vec_array([r["c2_vec"] for r in batch]),
        "bridge_vec": fixed_vec_array([r["bridge_vec"] for r in batch]),
        "cm1_words": pa.array([r["cm1_words"] for r in batch], type=pa.list_(pa.string())),
        "cm1_pos": pa.array([r["cm1_pos"] for r in batch], type=pa.list_(pa.string())),
        "cm2_words": pa.array([r["cm2_words"] for r in batch], type=pa.list_(pa.string())),
        "cm2_pos": pa.array([r["cm2_pos"] for r in batch], type=pa.list_(pa.string())),
        "bridge_words": pa.array([r["bridge_words"] for r in batch], type=pa.list_(pa.string())),
        "bridge_pos": pa.array([r["bridge_pos"] for r in batch], type=pa.list_(pa.string())),
        "n_cm1": pa.array([r["n_cm1"] for r in batch], type=pa.int32()),
        "n_cm2": pa.array([r["n_cm2"] for r in batch], type=pa.int32()),
        "n_bridge": pa.array([r["n_bridge"] for r in batch], type=pa.int32()),
        "n_words_total": pa.array([r["n_words_total"] for r in batch], type=pa.int32()),
    }
    return pa.table(cols, schema=UNIT_SCHEMA)


def _word_table(batch: list[dict[str, Any]]) -> pa.Table:
    cols = {
        "word": pa.array([r["word"] for r in batch], type=pa.string()),
        "pos": pa.array([r["pos"] for r in batch], type=pa.string()),
        "lang": pa.array([r["lang"] for r in batch], type=pa.string()),
        "vec": fixed_vec_array([r["vec"] for r in batch]),
        "source_level": pa.array([r["source_level"] for r in batch], type=pa.int8()),
        "n_nodes": pa.array([r["n_nodes"] for r in batch], type=pa.int32()),
        "n_l14": pa.array([r["n_l14"] for r in batch], type=pa.int32()),
        "n_l15": pa.array([r["n_l15"] for r in batch], type=pa.int32()),
    }
    return pa.table(cols, schema=WORD_SCHEMA)


def write_batches(path: Path, schema: pa.Schema, rows: list[dict[str, Any]], make_table, batch_size: int) -> None:
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for start in range(0, len(rows), batch_size):
            writer.write_table(make_table(rows[start:start + batch_size]))


# ---------------------------------------------------------------------------
# Diagnostics / report
# ---------------------------------------------------------------------------


def dist(values: list[float] | list[int]) -> dict[str, Any] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def baseline_stats(units: list[dict[str, Any]], word_map: dict[tuple[str, str], np.ndarray]) -> list[dict[str, Any]]:
    """Sanity check: reconstruct mb_vec from child words / child edge vectors."""
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for u in units:
        by_level[u["level"]].append(u)

    out: list[dict[str, Any]] = []
    for level in sorted(by_level, reverse=True):
        us = by_level[level]
        n_level = len(us)
        mb_total = np.sum(np.stack([u["mb_vec"] for u in us], axis=0), axis=0)
        cos_child, cos_all, cos_edge, cos_edge_all, cos_global = [], [], [], [], []
        rel_child, rel_all = [], []
        skipped = 0
        for u in us:
            keys1 = list(zip(u["cm1_words"], u["cm1_pos"]))
            keys2 = list(zip(u["cm2_words"], u["cm2_pos"]))
            keysb = list(zip(u["bridge_words"], u["bridge_pos"]))
            v1 = [word_map.get(k) for k in keys1]
            v2 = [word_map.get(k) for k in keys2]
            vb = [word_map.get(k) for k in keysb]
            if any(v is None for v in v1 + v2):
                skipped += 1
                continue
            m = normalize(u["mb_vec"])
            child_vecs = np.stack(v1 + v2)
            p_child = normalize(np.mean(child_vecs, axis=0))
            cos_child.append(cosine(m, p_child))
            rel_child.append(float(np.linalg.norm(m - p_child)))
            if vb and all(v is not None for v in vb):
                p_all = normalize(np.mean(np.stack(v1 + v2 + vb), axis=0))
                cos_all.append(cosine(m, p_all))
                rel_all.append(float(np.linalg.norm(m - p_all)))
            p_edge = normalize(u["c1_vec"] + u["c2_vec"])
            cos_edge.append(cosine(m, p_edge))
            p_edge_all = normalize(u["c1_vec"] + u["c2_vec"] + u["bridge_vec"])
            cos_edge_all.append(cosine(m, p_edge_all))
            if n_level > 1:
                loo_global = normalize((mb_total - u["mb_vec"]) / (n_level - 1))
                cos_global.append(cosine(m, loo_global))

        out.append(
            {
                "level": level,
                "n": len(us),
                "skipped_missing_word_vec": skipped,
                "cos_children_words": dist(cos_child),
                "relres_children_words": dist(rel_child),
                "cos_children_plus_bridge_words": dist(cos_all),
                "relres_children_plus_bridge_words": dist(rel_all),
                "cos_child_edge_oracle": dist(cos_edge),
                "cos_child_plus_bridge_edge_oracle": dist(cos_edge_all),
                "cos_global_mean": dist(cos_global),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="9-2", help="levels, e.g. 9-2 or 9,7,5 (default 9-2)")
    ap.add_argument("--workers", type=int, default=8, help="BFS worker threads (default 8)")
    ap.add_argument("--chunk", type=int, default=100, help="MBs per worker task (default 100)")
    ap.add_argument("--max-depth", type=int, default=20, help="BFS depth cap (default 20)")
    ap.add_argument("--limit", type=int, default=0, help="optional cap on MBs per level for smoke tests")
    ap.add_argument("--out-dir", default=str(ROOT / "evaluation/results/word_units"))
    args = ap.parse_args(argv)

    levels = parse_levels(args.levels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    log(f"connecting; levels={levels} workers={args.workers}")
    coll = get_collection()
    try:
        coll.database.client.admin.command("ping")
    except PyMongoError as exc:
        log(f"ERROR: MongoDB unreachable: {exc}")
        return 2

    # ---- 1. MB documents -------------------------------------------------
    t0 = time.time()
    mb_docs: list[dict[str, Any]] = []
    candidates: Counter = Counter()
    for level in levels:
        query = {"doc_type": "baryedge", "level": level, "source": {"$ne": "structural"}}
        docs = list(coll.find(query, MB_PROJ).sort("_id", 1))
        if args.limit:
            docs = docs[: args.limit]
        candidates[str(level)] = len(docs)
        mb_docs.extend(docs)
        log(f"L{level}: {len(docs):,} MB candidates")
    log(f"fetched {len(mb_docs):,} MB docs in {time.time() - t0:.1f}s")

    if not mb_docs:
        log("ERROR: no MB candidates")
        return 1

    # ---- 2. bridges and children ----------------------------------------
    t0 = time.time()
    bridge_by_mb: dict[Any, dict[str, Any]] = {}
    duplicate_bridges = 0
    for level in levels:
        ids = [d["_id"] for d in mb_docs if int(d["level"]) == level]
        for part in chunks(ids, 500):
            cursor = coll.find(
                {"doc_type": "baryedge", "level": level + 1, "parent_edge_id": {"$in": part}},
                BRIDGE_PROJ,
            )
            for doc in cursor:
                parent = doc.get("parent_edge_id")
                if parent is None:
                    continue
                cur = bridge_by_mb.get(parent)
                if cur is None:
                    bridge_by_mb[parent] = doc
                else:
                    duplicate_bridges += 1
                    if str(doc["_id"]) < str(cur["_id"]):
                        bridge_by_mb[parent] = doc
    log(f"bridges: {len(bridge_by_mb):,} for {len(mb_docs):,} MBs "
        f"({duplicate_bridges} duplicate parent links) in {time.time() - t0:.1f}s")

    t0 = time.time()
    child_ids: set[Any] = set()
    for mb in mb_docs:
        if mb.get("cm1_id") is not None:
            child_ids.add(mb["cm1_id"])
        if mb.get("cm2_id") is not None:
            child_ids.add(mb["cm2_id"])
    child_by_id = fetch_map(coll, list(child_ids), CHILD_PROJ)
    log(f"children: {len(child_by_id):,} docs for {len(child_ids):,} ids in {time.time() - t0:.1f}s")

    # ---- 3. leaf traversal + word aggregation ----------------------------
    t0 = time.time()
    chunks_list = [mb_docs[i:i + args.chunk] for i in range(0, len(mb_docs), args.chunk)]
    all_units: list[dict[str, Any]] = []
    global_agg: dict[tuple[str, str], dict[str, Any]] = {}
    drops: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_chunk, chunk, bridge_by_mb, child_by_id, coll, args.max_depth): i
            for i, chunk in enumerate(chunks_list)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="parse", unit="chunk"):
            units, agg, chunk_drops = fut.result()
            all_units.extend(units)
            merge_agg(global_agg, agg)
            drops.update(chunk_drops)
    all_units.sort(key=lambda r: (-r["level"], r["mb_id"]))
    log(f"parsed {len(all_units):,} units in {time.time() - t0:.1f}s; drops={dict(drops)}")

    if not all_units:
        log("ERROR: no units parsed")
        return 1

    # ---- 4. vocabulary ---------------------------------------------------
    t0 = time.time()
    word_rows: list[dict[str, Any]] = []
    bad_word_vectors = 0
    for key in sorted(global_agg):
        rec = global_agg[key]
        n = int(rec["n"])
        if n <= 0:
            continue
        vec = (rec["sum"] / n).astype(np.float32, copy=False)
        if not np.all(np.isfinite(vec)) or float(np.linalg.norm(vec)) < 1e-12:
            bad_word_vectors += 1
            continue
        word_rows.append(
            {
                "word": key[0],
                "pos": key[1],
                "lang": rec.get("lang") or "",
                "vec": vec,
                "source_level": int(rec["level"]),
                "n_nodes": n,
                "n_l14": int(rec.get("n_l14", 0)),
                "n_l15": int(rec.get("n_l15", 0)),
            }
        )
    word_map = {(r["word"], r["pos"]): r["vec"] for r in word_rows}
    l14_words = sum(1 for r in word_rows if r["source_level"] == 14)
    l15_words = sum(1 for r in word_rows if r["source_level"] == 15)
    del global_agg
    log(f"vocabulary: {len(word_rows):,} (word, pos) keys "
        f"(L14={l14_words:,}, L15 fallback={l15_words:,}, bad={bad_word_vectors}) "
        f"in {time.time() - t0:.1f}s")

    # ---- 5. diagnostics --------------------------------------------------
    t0 = time.time()
    baseline = baseline_stats(all_units, word_map)
    log(f"baseline diagnostics in {time.time() - t0:.1f}s")

    # ---- 6. write --------------------------------------------------------
    t0 = time.time()
    units_path = out_dir / "units.parquet"
    words_path = out_dir / "words.parquet"
    write_batches(units_path, UNIT_SCHEMA, all_units, _unit_table, batch_size=1_000)
    write_batches(words_path, WORD_SCHEMA, word_rows, _word_table, batch_size=5_000)
    log(f"wrote {units_path.name} ({units_path.stat().st_size / 1e6:.1f} MB) and "
        f"{words_path.name} ({words_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")

    # ---- 7. report -------------------------------------------------------
    per_level: dict[str, Any] = {}
    for level in levels:
        us = [u for u in all_units if u["level"] == level]
        base = next((b for b in baseline if b["level"] == level), None)
        per_level[str(level)] = {
            "mb_candidates": candidates.get(str(level), 0),
            "parsed": len(us),
            "words_cm1": dist([u["n_cm1"] for u in us]),
            "words_cm2": dist([u["n_cm2"] for u in us]),
            "words_bridge": dist([u["n_bridge"] for u in us]),
            "words_total": dist([u["n_words_total"] for u in us]),
            "baseline": base,
        }

    duration = time.time() - started
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": duration,
        "levels": levels,
        "workers": args.workers,
        "chunk": args.chunk,
        "mb_candidates": sum(candidates.values()),
        "units_parsed": len(all_units),
        "drops": dict(drops),
        "duplicate_bridge_links": duplicate_bridges,
        "per_level": per_level,
        "vocabulary": {
            "keys": len(word_rows),
            "source_l14": l14_words,
            "source_l15_fallback": l15_words,
            "bad_vectors": bad_word_vectors,
            "n_nodes": dist([r["n_nodes"] for r in word_rows]),
            "n_l14": dist([r["n_l14"] for r in word_rows]),
            "n_l15": dist([r["n_l15"] for r in word_rows]),
        },
        "output": {
            "units": str(units_path),
            "words": str(words_path),
            "units_bytes": units_path.stat().st_size,
            "words_bytes": words_path.stat().st_size,
        },
        "schema": {
            "units": [f.name for f in UNIT_SCHEMA],
            "words": [f.name for f in WORD_SCHEMA],
            "vector_dim": DIM,
        },
        "notes": [
            "MB levels are 2-9; every non-structural baryedge at a level is one unit.",
            "cm1/cm2/bridge word lists are unique (word, pos) keys, sorted by key.",
            "Canonical word vector = mean of L14 word-node vectors when any exist, else mean of L15 sense vectors.",
            "bridge_words and bridge_vec are stored for residual/oracle analysis; the planned model input is cm1+cm2 words only.",
            "accumulated_weight and connection_strength are not stored.",
            "bridge_id and edge_type are not stored; the bridge is recoverable from the MB id.",
            "Baseline numbers are a sanity check on the parsed vocabulary, not a trained model.",
            "cos_global_mean is a leave-one-out mean of the level's MB vectors (the target unit is excluded).",
        ],
    }
    report_path = out_dir / "parse_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md = [
        "# Word-level MB units (L2–L9)",
        "",
        f"Generated: {report['generated_at']}",
        f"Units: {len(all_units):,}; vocabulary: {len(word_rows):,} (word, pos) keys; "
        f"runtime: {duration / 60:.1f} min",
        "",
        "## Files",
        "",
        "| file | content |",
        "|---|---|",
        "| `units.parquet` | one row per MB: vector, child/bridge vectors, word keys |",
        "| `words.parquet` | one row per `(word, pos)` with the canonical 1024-d vector |",
        "| `parse_report.json` | counts, drops, per-level stats, sanity baselines |",
        "",
        "## Word vector rule",
        "",
        "For each `(word, pos)`, use the mean of its L14 word-node vectors when at least one is "
        "reachable; otherwise use the mean of its L15 sense vectors. The `words.parquet` row "
        "records `source_level` (14 or 15), `n_nodes`, `n_l14` and `n_l15`.",
        "",
        "## Per-level counts",
        "",
        "| L | candidates | parsed | words/MB (cm1) | words/MB (cm2) | bridge | total |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in levels:
        p = per_level[str(level)]
        def med(d):
            return "—" if not d else f"{d['median']:.0f}"
        md.append(
            f"| {level} | {p['mb_candidates']:,} | {p['parsed']:,} | "
            f"{med(p['words_cm1'])} | {med(p['words_cm2'])} | {med(p['words_bridge'])} | "
            f"{med(p['words_total'])} |"
        )
    md += [
        "",
        "## Sanity baselines (median cosine)",
        "",
        "| L | children words | +bridge words | child-edge oracle | +bridge edge | global mean (LOO) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for b in baseline:
        def m(key):
            d = b.get(key)
            return "—" if not d else f"{d['median']:.3f}"
        md.append(
            f"| {b['level']} | {m('cos_children_words')} | {m('cos_children_plus_bridge_words')} | "
            f"{m('cos_child_edge_oracle')} | {m('cos_child_plus_bridge_edge_oracle')} | "
            f"{m('cos_global_mean')} |"
        )
    md += [
        "",
        "`global mean (LOO)` excludes the target MB from the level mean.",
        "",
        "The bridge is stored, not used as a model input. The gap between children words and "
        "children+bridge words is the residual the bridge must explain.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    # ---- 8. console summary ---------------------------------------------
    print(f"\n=== parsed {len(all_units):,} units, {len(word_rows):,} words ===")
    print(f"{'L':>3} {'MBs':>7} {'cm1':>6} {'cm2':>6} {'bridge':>7} {'total':>6} "
          f"{'cos_child':>10} {'cos+bridge':>11} {'cos_edge':>9} {'cos_global':>11}")
    for b in baseline:
        p = per_level[str(b["level"])]
        def med(d):
            return float("nan") if not d else d["median"]
        print(
            f"{b['level']:>3} {p['parsed']:>7,} "
            f"{med(p['words_cm1']):>6.0f} {med(p['words_cm2']):>6.0f} "
            f"{med(p['words_bridge']):>7.0f} {med(p['words_total']):>6.0f} "
            f"{med(b['cos_children_words']):>10.3f} "
            f"{med(b['cos_children_plus_bridge_words']):>11.3f} "
            f"{med(b['cos_child_edge_oracle']):>9.3f} "
            f"{med(b['cos_global_mean']):>11.3f}"
        )
    print(f"\nOutput: {out_dir}")
    print(f"Report: {report_path}")
    print(f"Runtime: {duration / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
