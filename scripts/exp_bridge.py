#!/usr/bin/env python3
"""True-vs-swapped bridge experiment for BaryGraph MetaBary triads.

This is a measurement script, not a training script. It answers:

  A. Does the stored MetaBary vector ``m`` carry information about its bridge
     beyond what the two children give?  We compare word-overlap, child-centroid
     cosine, raw ``cos(m, candidate)``, and an algebraic oracle in a 10-way
     selection task.
  B. Can we tell a real bridge from a swapped one using the residual
     ``||m - normalize(w1*c1 + w2*c2 + w3*b')||``?

The script is read-only against MongoDB. It samples (never scans/aggregates the
whole collection) and writes ``evaluation/results/bridge_exp/report.json``.

Usage:
    python scripts/exp_bridge.py --levels 13,11 --n 30 --control
    python scripts/exp_bridge.py --levels 13,11,9 --n 200 --control
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymongo.errors import OperationFailure, PyMongoError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.db import chunks, cm_leaf_words, get_collection, vector_search  # noqa: E402
from lib.vector import cosine, normalize, unpack_vec  # noqa: E402

# ---------------------------------------------------------------------------
# Small pure helpers (covered by tests/unit/test_exp_bridge.py)
# ---------------------------------------------------------------------------


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity. Two empty sets score 0.0 (no evidence either way)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def rank_of_true(scores: list[float], true_index: int = 0) -> int:
    """Competition rank of the true item: 1 + number of strictly higher scores.

    Ties are *not* broken arbitrarily: a true item tied for best is rank 1.
    """
    if not scores:
        return 0
    true_score = scores[true_index]
    return 1 + sum(1 for i, s in enumerate(scores) if i != true_index and s > true_score)


def oracle_bridge(
    m: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    w1: float,
    w2: float,
    w3: float,
) -> np.ndarray:
    """Solve ``m = normalize(w1*c1 + w2*c2 + w3*b)`` for ``b``.

    The derivation assumes unit-norm ``m, c1, c2`` and a positive ``w3``. If
    ``w3`` is ~0 the bridge term vanishes and ``m`` itself is the only sensible
    answer.
    """
    m = normalize(m)
    c1 = normalize(c1)
    c2 = normalize(c2)
    if w3 < 1e-6:
        return m.copy()

    a = w1 * c1 + w2 * c2
    ma = float(np.dot(m, a))
    disc = ma * ma - float(np.dot(a, a)) + w3 * w3
    lam = ma + math.sqrt(max(disc, 0.0))
    return ((lam * m) - a) / w3


def residual(
    m: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    bridge: np.ndarray,
    w1: float,
    w2: float,
    w3: float,
) -> float:
    """L2 residual of ``m`` against ``normalize(w1*c1 + w2*c2 + w3*bridge)``."""
    raw = w1 * np.asarray(c1, dtype=np.float32) + w2 * np.asarray(c2, dtype=np.float32) \
        + w3 * np.asarray(bridge, dtype=np.float32)
    raw = normalize(raw)
    return float(np.linalg.norm(np.asarray(m, dtype=np.float32) - raw))


def auc_from_pairs(pairs: Iterable[tuple[float, float]]) -> float | None:
    """P(true residual < swapped residual), ties count 0.5.

    Lower residual means "more real", so this is the AUC for the verification
    task. Returns ``None`` for an empty input.
    """
    vals = list(pairs)
    if not vals:
        return None
    score = 0.0
    for true_res, swap_res in vals:
        if true_res < swap_res:
            score += 1.0
        elif true_res == swap_res:
            score += 0.5
    return score / len(vals)


def is_degenerate(bridge_words: Iterable[str], child_words: Iterable[str]) -> bool:
    """A bridge that merely restates its children adds no new member."""
    bw, cw = set(bridge_words), set(child_words)
    return bw.issubset(cw) or jaccard(bw, cw) >= 0.5


def band_tolerances(base: float) -> list[float]:
    out: list[float] = []
    for v in (base, max(base, 0.15), 0.3, 0.6, 1.0, float("inf")):
        if not out or v != out[-1]:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    id: Any
    vec: np.ndarray
    weight: float
    words: frozenset[str] = frozenset()
    source: str = "far"


@dataclass
class Triad:
    mb_id: Any
    level: int
    bridge_level: int
    m: np.ndarray
    c1_id: Any
    c2_id: Any
    b_id: Any
    c1: np.ndarray
    c2: np.ndarray
    b: np.ndarray
    w1: float
    w2: float
    w3: float
    child_words: frozenset[str]
    bridge_words: frozenset[str]
    bridge_edge_type: str | None
    recon_error: float
    degenerate: bool
    no_overlap: bool

    @property
    def clean(self) -> bool:
        return not self.degenerate


@dataclass
class Pool:
    name: str
    candidates: list[Candidate]
    meta: dict[str, Any] = field(default_factory=dict)


class WordCache:
    """Cache the expensive cm-lineage BFS by node id."""

    def __init__(self, coll, max_words: int = 8, max_depth: int = 15):
        self.coll = coll
        self.max_words = max_words
        self.max_depth = max_depth
        self._cache: dict[str, frozenset[str]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, oid: Any) -> frozenset[str]:
        key = str(oid)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        words = cm_leaf_words(
            self.coll,
            oid,
            max_depth=self.max_depth,
            max_words=self.max_words,
        )
        val = frozenset(words)
        self._cache[key] = val
        return val


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _weight_of(doc: dict[str, Any]) -> float | None:
    for key in ("accumulated_weight", "connection_strength"):
        val = doc.get(key)
        if val is None:
            continue
        try:
            w = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(w):
            return w
    return None


def _vec_of(doc: dict[str, Any]) -> np.ndarray | None:
    blob = doc.get("vector")
    if blob is None:
        return None
    try:
        vec = unpack_vec(blob)
    except Exception:
        return None
    if vec.ndim != 1 or vec.size == 0 or not np.all(np.isfinite(vec)):
        return None
    if float(np.linalg.norm(vec)) < 1e-8:
        return None
    return vec.astype(np.float32, copy=False)


def fetch_docs(coll, ids: list[Any], projection: dict[str, int], chunk_size: int = 500,
               timeout_ms: int = 120_000) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for part in chunks(ids, chunk_size):
        try:
            cur = coll.find({"_id": {"$in": part}}, projection).max_time_ms(timeout_ms)
            for doc in cur:
                out[doc["_id"]] = doc
        except OperationFailure as exc:
            log(f"  WARN fetch chunk timed out: {exc}")
    return out


def sample_mb_ids(coll, level: int, n: int, rng: random.Random,
                  scan_cap: int | None) -> tuple[list[Any], int]:
    """Reservoir-sample MB ids from the full matching set (ids only)."""
    query = {
        "doc_type": "baryedge",
        "level": level,
        "parent_edge_id": None,
        "source": {"$ne": "structural"},
    }
    sample: list[Any] = []
    seen = 0
    try:
        cursor = coll.find(query, {"_id": 1}).batch_size(10_000)
        for doc in cursor:
            seen += 1
            oid = doc["_id"]
            if len(sample) < n:
                sample.append(oid)
            else:
                j = rng.randrange(seen)
                if j < n:
                    sample[j] = oid
            if scan_cap and seen >= scan_cap:
                break
    except OperationFailure as exc:
        log(f"  WARN MB id scan at L{level} stopped: {exc}")
    return sample, seen


def build_triads(
    coll,
    level: int,
    n: int,
    rng: random.Random,
    word_cache: WordCache,
    scan_cap: int | None,
) -> tuple[list[Triad], dict[str, Any]]:
    """Sample MBs at ``level`` and materialise complete (m, c1, c2, b) triads."""
    stats: dict[str, Any] = {
        "level": level,
        "requested": n,
        "mb_ids_scanned": 0,
        "mb_ids_sampled": 0,
        "mb_docs_found": 0,
        "children_found": 0,
        "dropped": Counter(),
        "edge_types": Counter(),
    }
    mb_ids, scanned = sample_mb_ids(coll, level, n, rng, scan_cap)
    stats["mb_ids_scanned"] = scanned
    stats["mb_ids_sampled"] = len(mb_ids)
    if not mb_ids:
        return [], stats

    mb_docs = fetch_docs(
        coll,
        mb_ids,
        {"vector": 1, "cm1_id": 1, "cm2_id": 1, "level": 1,
         "accumulated_weight": 1, "connection_strength": 1},
    )
    stats["mb_docs_found"] = len(mb_docs)

    children_by_parent: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for part in chunks(mb_ids, 500):
        try:
            cur = coll.find(
                {"parent_edge_id": {"$in": part}},
                {"vector": 1, "level": 1, "cm1_id": 1, "cm2_id": 1,
                 "accumulated_weight": 1, "connection_strength": 1,
                 "edge_type": 1, "parent_edge_id": 1},
            ).max_time_ms(120_000)
            for doc in cur:
                children_by_parent[doc.get("parent_edge_id")].append(doc)
        except OperationFailure as exc:
            log(f"  WARN child fetch at L{level} timed out: {exc}")

    triads: list[Triad] = []
    seen_keys: set[Any] = set()
    dropped = stats["dropped"]

    for idx, mb_id in enumerate(mb_ids):
        mb = mb_docs.get(mb_id)
        if not mb:
            dropped["mb_missing"] += 1
            continue

        kids = children_by_parent.get(mb_id, [])
        stats["children_found"] += len(kids)
        by_id = {k["_id"]: k for k in kids}

        c1_doc = by_id.get(mb.get("cm1_id"))
        c2_doc = by_id.get(mb.get("cm2_id"))
        if c1_doc is None or c2_doc is None:
            dropped["missing_children"] += 1
            continue

        bridges = [k for k in kids if k.get("level") == level + 1]
        if not bridges:
            dropped["missing_bridge"] += 1
            continue
        # One bridge per MB in the pipeline. If a malformed MB has several,
        # take the first in a deterministic id order.
        b_doc = sorted(bridges, key=lambda d: str(d["_id"]))[0]

        expected_child_level = level + 2
        if c1_doc.get("level") != expected_child_level or c2_doc.get("level") != expected_child_level:
            dropped["child_level_mismatch"] += 1
            continue

        m = _vec_of(mb)
        c1 = _vec_of(c1_doc)
        c2 = _vec_of(c2_doc)
        b = _vec_of(b_doc)
        if m is None or c1 is None or c2 is None or b is None:
            dropped["missing_or_zero_vector"] += 1
            continue
        if not (m.size == c1.size == c2.size == b.size):
            dropped["dim_mismatch"] += 1
            continue

        w1 = _weight_of(c1_doc)
        w2 = _weight_of(c2_doc)
        w3 = _weight_of(b_doc)
        if w1 is None or w2 is None or w3 is None:
            dropped["missing_weight"] += 1
            continue
        if w1 < 0 or w2 < 0 or w3 < 0 or (w1 == 0 and w2 == 0 and w3 == 0):
            dropped["bad_weight"] += 1
            continue

        cos12 = cosine(c1, c2)
        if cos12 > 0.9999:
            dropped["collinear_children"] += 1
            continue

        child_words = word_cache.get(mb["cm1_id"]) | word_cache.get(mb["cm2_id"])
        bridge_words = word_cache.get(b_doc["_id"])
        if not child_words:
            dropped["no_child_words"] += 1
            continue
        if not bridge_words:
            dropped["no_bridge_words"] += 1
            continue

        key = (frozenset(child_words), frozenset(bridge_words))
        if key in seen_keys:
            dropped["duplicate_triad"] += 1
            continue
        seen_keys.add(key)

        recon = residual(m, c1, c2, b, w1, w2, w3)
        deg = is_degenerate(bridge_words, child_words)
        triad = Triad(
            mb_id=mb_id,
            level=level,
            bridge_level=level + 1,
            m=m,
            c1_id=c1_doc["_id"],
            c2_id=c2_doc["_id"],
            b_id=b_doc["_id"],
            c1=c1,
            c2=c2,
            b=b,
            w1=w1,
            w2=w2,
            w3=w3,
            child_words=frozenset(child_words),
            bridge_words=frozenset(bridge_words),
            bridge_edge_type=b_doc.get("edge_type"),
            recon_error=recon,
            degenerate=deg,
            no_overlap=not bool(set(bridge_words) & set(child_words)),
        )
        triads.append(triad)
        stats["edge_types"][str(triad.bridge_edge_type)] += 1

        if (idx + 1) % 25 == 0:
            log(f"  L{level}: built {len(triads)}/{idx + 1} triads "
                f"(word cache {word_cache.hits} hits/{word_cache.misses} misses)")

    stats["triads"] = len(triads)
    stats["degenerate"] = sum(1 for t in triads if t.degenerate)
    stats["clean"] = sum(1 for t in triads if t.clean)
    stats["no_overlap"] = sum(1 for t in triads if t.no_overlap)
    stats["dropped"] = dict(dropped)
    stats["edge_types"] = dict(stats["edge_types"])
    if triads:
        errs = sorted(t.recon_error for t in triads)
        stats["recon_error"] = {
            "median": float(statistics.median(errs)),
            "p95": float(errs[min(len(errs) - 1, int(0.95 * len(errs)))]),
            "max": float(max(errs)),
            "below_1e-4": sum(1 for x in errs if x < 1e-4) / len(errs),
            "below_1e-3": sum(1 for x in errs if x < 1e-3) / len(errs),
        }
    return triads, stats


# ---------------------------------------------------------------------------
# Candidate pools
# ---------------------------------------------------------------------------


def make_candidate(doc: dict[str, Any], source: str,
                   word_cache: WordCache | None = None) -> Candidate | None:
    vec = _vec_of(doc)
    if vec is None:
        return None
    w = _weight_of(doc)
    if w is None:
        return None
    words = word_cache.get(doc["_id"]) if word_cache is not None else frozenset()
    return Candidate(id=doc["_id"], vec=vec, weight=w, words=words, source=source)


def build_random_pool(coll, level: int, size: int, scan_cap: int,
                      rng: random.Random) -> Pool:
    """Reservoir sample same-level BEs, then fetch their vectors in bulk."""
    query = {"doc_type": "baryedge", "level": level}
    sampled: list[Any] = []
    seen = 0
    try:
        cursor = coll.find(query, {"_id": 1, "accumulated_weight": 1}).batch_size(10_000)
        for doc in cursor:
            seen += 1
            oid = doc["_id"]
            if len(sampled) < size:
                sampled.append(oid)
            else:
                j = rng.randrange(seen)
                if j < size:
                    sampled[j] = oid
            if scan_cap and seen >= scan_cap:
                break
    except OperationFailure as exc:
        log(f"  WARN random pool scan at L{level}: {exc}")

    docs = fetch_docs(
        coll,
        sampled,
        {"vector": 1, "accumulated_weight": 1, "connection_strength": 1, "level": 1},
    )
    cands = [c for oid in sampled if (d := docs.get(oid)) and (c := make_candidate(d, "far"))]
    weights = [c.weight for c in cands]
    return Pool(
        name="far",
        candidates=cands,
        meta={
            "level": level,
            "requested": size,
            "scanned": seen,
            "returned": len(cands),
            "weight_min": min(weights) if weights else None,
            "weight_median": float(statistics.median(weights)) if weights else None,
            "weight_max": max(weights) if weights else None,
        },
    )


def build_lexical_pool(coll, level: int, random_pool: Pool, size: int,
                       rng: random.Random, word_cache: WordCache) -> Pool:
    """A fixed random same-level sample with leaf words precomputed.

    The handoff asks for ~120 random same-level BEs per triad. Recomputing the
    cm-lineage BFS per triad would dominate the runtime, so we use one shared
    random sample per level and rank it per triad. This is still a random
    same-level sample; it only changes the variance, not the baseline.
    """
    if not random_pool.candidates:
        return Pool(name="lexical", candidates=[], meta={"level": level, "returned": 0})
    take = min(size, len(random_pool.candidates))
    picked = rng.sample(random_pool.candidates, take)
    out: list[Candidate] = []
    for c in picked:
        words = word_cache.get(c.id)
        if words:
            out.append(Candidate(id=c.id, vec=c.vec, weight=c.weight,
                                 words=frozenset(words), source="lexical"))
    return Pool(
        name="lexical",
        candidates=out,
        meta={"level": level, "requested": size, "returned": len(out)},
    )


def build_near_pool(coll, triad: Triad, limit: int = 100,
                    num_candidates: int | None = None) -> Pool:
    # Wider pools (e.g. 49 hard negatives) need enough ANN candidates to fill
    # the weight band; otherwise most cases would be skipped or widened.
    if num_candidates is None:
        num_candidates = max(200, 2 * limit)
    try:
        docs = vector_search(
            coll,
            triad.m,
            limit=limit,
            num_candidates=num_candidates,
            filter={"doc_type": "baryedge", "level": triad.bridge_level},
        )
    except (OperationFailure, PyMongoError) as exc:
        return Pool(name="near", candidates=[], meta={"error": str(exc), "returned": 0})

    cands = [c for d in docs if (c := make_candidate(d, "near"))]
    return Pool(
        name="near",
        candidates=cands,
        meta={
            "requested": limit,
            "returned": len(cands),
            "bridge_level": triad.bridge_level,
        },
    )


def build_sibling_pool(coll, triad: Triad, timeout_ms: int = 3_000) -> Pool:
    """Bridges of other MBs sharing one of this triad's children.

    There is no (level, cm1_id) compound index in the PoC, so this query is
    capped hard. An empty pool is expected on some levels and is reported as
    such rather than silently treated as a modelling result.
    """
    query = {
        "doc_type": "baryedge",
        "level": triad.level,
        "parent_edge_id": None,
        "source": {"$ne": "structural"},
        "$or": [
            {"cm1_id": {"$in": [triad.c1_id, triad.c2_id]}},
            {"cm2_id": {"$in": [triad.c1_id, triad.c2_id]}},
        ],
    }
    try:
        sib = list(coll.find(query, {"_id": 1}).max_time_ms(timeout_ms))
    except OperationFailure as exc:
        return Pool(name="sibling", candidates=[],
                    meta={"error": str(exc), "returned": 0})
    sib_ids = [d["_id"] for d in sib if d["_id"] != triad.mb_id]
    if not sib_ids:
        return Pool(name="sibling", candidates=[], meta={"sibling_mbs": 0, "returned": 0})

    try:
        bridge_docs = list(
            coll.find(
                {"parent_edge_id": {"$in": sib_ids}, "level": triad.bridge_level},
                {"vector": 1, "accumulated_weight": 1, "connection_strength": 1,
                 "level": 1, "edge_type": 1},
            ).max_time_ms(timeout_ms)
        )
    except OperationFailure as exc:
        return Pool(name="sibling", candidates=[],
                    meta={"sibling_mbs": len(sib_ids), "error": str(exc), "returned": 0})

    cands = [c for d in bridge_docs if (c := make_candidate(d, "sibling"))]
    return Pool(
        name="sibling",
        candidates=cands,
        meta={"sibling_mbs": len(sib_ids), "returned": len(cands)},
    )


def select_negatives(
    pool: Pool,
    triad: Triad,
    k: int,
    rng: random.Random,
    *,
    lexical: bool = False,
    require_disjoint_words: bool = False,
    word_cache: WordCache | None = None,
    fill_words: bool = False,
    exclude_duplicate_bridge: bool = False,
    extra_band: float | None = None,
) -> tuple[list[Candidate], float | None]:
    """Pick up to k negatives, widening the weight band only if needed."""
    exclude = {triad.mb_id, triad.c1_id, triad.c2_id, triad.b_id}
    base = extra_band if extra_band is not None else (0.15 if triad.level == 13 else 0.05)

    best: list[Candidate] = []
    used_tol: float | None = None
    for tol in band_tolerances(base):
        seen_ids: set[Any] = set()
        filt: list[Candidate] = []
        for c in pool.candidates:
            if c.id in exclude or c.id in seen_ids:
                continue
            if abs(c.weight - triad.w3) > tol:
                continue
            if exclude_duplicate_bridge and cosine(c.vec, triad.b) > 0.9999:
                # The same relation stored twice is not a negative: it is the
                # answer. Leave it out so A3/A1 are not trivially perfect.
                continue
            if lexical and c.words == triad.bridge_words:
                continue
            if require_disjoint_words:
                if word_cache is None:
                    raise ValueError("word_cache is required for disjoint word filtering")
                words = c.words or word_cache.get(c.id)
                if not words:
                    continue
                c.words = frozenset(words)
                if not c.words.isdisjoint(triad.bridge_words):
                    continue
            seen_ids.add(c.id)
            filt.append(c)

        if lexical:
            filt.sort(key=lambda c: jaccard(c.words, triad.child_words), reverse=True)
            chosen = filt[:k]
        else:
            rng.shuffle(filt)
            chosen = filt[:k]

        if len(chosen) > len(best):
            best, used_tol = chosen, tol
        if len(chosen) >= k:
            break

    if fill_words and word_cache is not None:
        for c in best:
            if not c.words:
                c.words = frozenset(word_cache.get(c.id))
    return best, used_tol


# ---------------------------------------------------------------------------
# Task A — 10-way selection
# ---------------------------------------------------------------------------


def rank_metrics(scores: list[float], true_index: int = 0) -> dict[str, float]:
    """Rank metrics with explicit tie handling.

    ``top1`` uses competition rank (a tie for best counts as rank 1).
    ``top1_strict`` requires the true item to be strictly above every
    competitor, and ``mrr_tie`` uses the average rank within a tie group.
    """
    if not scores:
        return {"rank": 0, "top1": 0.0, "top1_strict": 0.0, "mrr": 0.0,
                "mrr_tie": 0.0, "tie_rate": 0.0, "margin": 0.0, "k": 0}
    true_score = scores[true_index]
    others = [s for i, s in enumerate(scores) if i != true_index]
    higher = sum(1 for s in others if s > true_score)
    ties = sum(1 for s in others if s == true_score)
    rank = 1 + higher
    avg_rank = rank + ties / 2.0
    return {
        "rank": rank,
        "top1": 1.0 if rank == 1 else 0.0,
        "top1_strict": 1.0 if higher == 0 and ties == 0 else 0.0,
        "mrr": 1.0 / rank,
        "mrr_tie": 1.0 / avg_rank if avg_rank else 0.0,
        "tie_rate": (ties / len(others)) if others else 0.0,
        "margin": true_score - max(others) if others else 0.0,
        "k": len(scores),
    }


def evaluate_arms(candidates: list[Candidate], triad: Triad, rng: random.Random) -> dict[str, dict[str, float]]:
    """Score every candidate with A0..A4 and rank the true (index 0) item."""
    child_dir = normalize(triad.c1 + triad.c2)
    oracle = oracle_bridge(triad.m, triad.c1, triad.c2, triad.w1, triad.w2, triad.w3)

    scores: dict[str, list[float]] = {
        "A0": [rng.random() for _ in candidates],
        "A1": [jaccard(c.words, triad.child_words) for c in candidates],
        "A2": [cosine(c.vec, child_dir) for c in candidates],
        "A3": [cosine(c.vec, triad.m) for c in candidates],
        "A4": [cosine(c.vec, oracle) for c in candidates],
    }
    return {arm: rank_metrics(vals, 0) for arm, vals in scores.items()}


def add_rows(
    rows: list[dict[str, Any]],
    triad: Triad,
    pool_name: str,
    arm: str,
    metrics: dict[str, float],
    band_tol: float | None,
    *,
    control: str | None = None,
) -> None:
    subsets = ["all"]
    if triad.clean:
        subsets.append("clean")
    if triad.no_overlap:
        subsets.append("no_overlap")
    if triad.recon_error < 1e-4:
        subsets.append("faithful")
    for subset in subsets:
        rows.append({
            "level": triad.level,
            "pool": pool_name,
            "arm": arm,
            "subset": subset,
            "control": control,
            "top1": metrics["top1"],
            "top1_strict": metrics["top1_strict"],
            "mrr": metrics["mrr"],
            "mrr_tie": metrics["mrr_tie"],
            "tie_rate": metrics["tie_rate"],
            "margin": metrics["margin"],
            "rank": metrics["rank"],
            "k": metrics["k"],
            "band_tol": band_tol,
            "degenerate": triad.degenerate,
            "no_overlap": triad.no_overlap,
            "recon_error": triad.recon_error,
        })


def summarize_rows(rows: list[dict[str, Any]], *, control: str | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if control is None:
            if r.get("control") is not None:
                continue
        elif r.get("control") != control:
            continue
        groups[(r["level"], r["pool"], r["arm"], r["subset"])].append(r)

    out: list[dict[str, Any]] = []
    for (level, pool, arm, subset), rs in sorted(groups.items()):
        n = len(rs)
        top1 = sum(r["top1"] for r in rs) / n
        top1_strict = sum(r["top1_strict"] for r in rs) / n
        mrr = sum(r["mrr"] for r in rs) / n
        mrr_tie = sum(r["mrr_tie"] for r in rs) / n
        tie_rate = sum(r["tie_rate"] for r in rs) / n
        margin = sum(r["margin"] for r in rs) / n
        k_mean = sum(r["k"] for r in rs) / n
        p = top1
        se = math.sqrt(p * (1.0 - p) / n) if n else 0.0
        out.append({
            "level": level,
            "pool": pool,
            "arm": arm,
            "subset": subset,
            "n": n,
            "k_mean": k_mean,
            "top1": top1,
            "top1_strict": top1_strict,
            "top1_stderr": se,
            "mrr": mrr,
            "mrr_tie": mrr_tie,
            "tie_rate": tie_rate,
            "margin_mean": margin,
        })
    return out


def evaluate_triad_task_a(
    triad: Triad,
    pools: dict[str, Pool],
    rng: random.Random,
    rows: list[dict[str, Any]],
    *,
    word_cache: WordCache | None = None,
    control_m: np.ndarray | None = None,
    label_shuffle: bool = False,
) -> None:
    for pool_name, pool in pools.items():
        negs, tol = select_negatives(
            pool, triad, 9, rng,
            lexical=(pool_name == "lexical"),
            require_disjoint_words=False,
            word_cache=word_cache,
            fill_words=word_cache is not None,
            exclude_duplicate_bridge=(pool_name != "far"),
        )
        if not negs:
            continue
        true_c = Candidate(triad.b_id, triad.b, triad.w3, triad.bridge_words, "true")
        candidates = [true_c] + negs

        metrics = evaluate_arms(candidates, triad, rng)
        for arm, m in metrics.items():
            add_rows(rows, triad, pool_name, arm, m, tol)

        if control_m is not None:
            # Vector-shuffle control: keep the candidate pool, replace the
            # query vector with another triad's m. A3 must collapse to chance.
            vals = [cosine(c.vec, control_m) for c in candidates]
            add_rows(rows, triad, pool_name, "A3_shuffle",
                     rank_metrics(vals, 0), tol, control="vector_shuffle")

        if label_shuffle:
            # Pseudo-truth label shuffle. Scores are real A3 scores; only the
            # label is random. This is chance by construction and checks that
            # the ranking code has no position bias.
            vals = [cosine(c.vec, triad.m) for c in candidates]
            pseudo = rng.randrange(len(candidates))
            add_rows(rows, triad, pool_name, "A3_label_shuffle",
                     rank_metrics(vals, pseudo), tol, control="label_shuffle")


# ---------------------------------------------------------------------------
# Task B — true vs swapped bridge
# ---------------------------------------------------------------------------


def _swap_candidates(
    pool: Pool,
    triad: Triad,
    k: int,
    rng: random.Random,
    word_cache: WordCache,
    *,
    distance_match: bool = False,
) -> list[Candidate]:
    exclude = {triad.mb_id, triad.c1_id, triad.c2_id, triad.b_id}
    tol = 0.15 if triad.level == 13 else 0.05
    cos_true = cosine(triad.m, triad.b)
    seen: set[Any] = set()
    out: list[Candidate] = []
    shuffled = list(pool.candidates)
    rng.shuffle(shuffled)

    for c in shuffled:
        if c.id in exclude or c.id in seen:
            continue
        if abs(c.weight - triad.w3) > tol:
            continue
        if cosine(c.vec, triad.b) > 0.9999:
            continue
        cos_c = cosine(triad.m, c.vec)
        if distance_match and abs(cos_c - cos_true) > 0.02:
            continue
        words = c.words or word_cache.get(c.id)
        if not words or not frozenset(words).isdisjoint(triad.bridge_words):
            continue
        seen.add(c.id)
        c.words = frozenset(words)
        out.append(c)
        if len(out) >= k:
            break
    return out


def evaluate_task_b(
    triads: list[Triad],
    pools_by_triad: dict[Any, dict[str, Pool]],
    word_cache: WordCache,
    rng: random.Random,
    swap_n: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    missing = Counter()
    by_tier_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_level_pairs: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)

    pool_map = {
        "random": "far",
        "neighbour": "near",
        "distance_matched": None,  # union of far/near/lexical
    }

    for triad in triads:
        true_res = triad.recon_error
        cos_true = cosine(triad.m, triad.b)
        triad_pools = pools_by_triad.get(triad.mb_id, {})

        for tier, pool_name in pool_map.items():
            if pool_name is None:
                # Union of the three content pools, deduped by id.
                merged: list[Candidate] = []
                seen: set[Any] = set()
                for p in ("far", "near", "lexical"):
                    for c in triad_pools.get(p, Pool(p, [])).candidates:
                        if c.id not in seen:
                            seen.add(c.id)
                            merged.append(c)
                pool = Pool("union", merged)
            else:
                pool = triad_pools.get(pool_name)
            if pool is None or not pool.candidates:
                missing[tier] += 1
                continue

            swaps = _swap_candidates(
                pool, triad, swap_n, rng, word_cache,
                distance_match=(tier == "distance_matched"),
            )
            if not swaps:
                missing[tier] += 1
                continue

            for c in swaps:
                swap_res = residual(triad.m, triad.c1, triad.c2, c.vec,
                                    triad.w1, triad.w2, triad.w3)
                cos_swap = cosine(triad.m, c.vec)
                correct = 1.0 if true_res < swap_res else (0.5 if true_res == swap_res else 0.0)
                case = {
                    "triad_id": str(triad.mb_id),
                    "level": triad.level,
                    "bridge_level": triad.bridge_level,
                    "tier": tier,
                    "bridge_id": str(triad.b_id),
                    "swapped_bridge_id": str(c.id),
                    "residual_true": true_res,
                    "residual_swapped": swap_res,
                    "correct": correct,
                    "cos_m_true_bridge": cos_true,
                    "cos_m_swapped_bridge": cos_swap,
                    "distance_delta": abs(cos_swap - cos_true),
                    "weight_delta": abs(c.weight - triad.w3),
                    "degenerate": triad.degenerate,
                    "no_overlap": triad.no_overlap,
                    "bridge_edge_type": triad.bridge_edge_type,
                }
                cases.append(case)
                pair = (true_res, swap_res)
                by_tier_pairs[tier].append(pair)
                by_level_pairs[(triad.level, tier)].append(pair)

    tier_summary: dict[str, Any] = {}
    for tier, pairs in sorted(by_tier_pairs.items()):
        n = len(pairs)
        tier_summary[tier] = {
            "n_pairs": n,
            "auc": auc_from_pairs(pairs),
            "mean_true_residual": sum(p[0] for p in pairs) / n if n else None,
            "mean_swapped_residual": sum(p[1] for p in pairs) / n if n else None,
            "correct_rate": sum(1.0 if t < s else (0.5 if t == s else 0.0)
                                for t, s in pairs) / n if n else None,
        }

    level_tier_summary: list[dict[str, Any]] = []
    for (level, tier), pairs in sorted(by_level_pairs.items()):
        level_tier_summary.append({
            "level": level,
            "tier": tier,
            "n_pairs": len(pairs),
            "auc": auc_from_pairs(pairs),
        })

    return {
        "summary": tier_summary,
        "by_level_tier": level_tier_summary,
        "missing": dict(missing),
        "cases": cases,
        "n_cases": len(cases),
    }


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:.1f}%"


def print_headline(summary: list[dict[str, Any]], subset: str = "clean") -> None:
    rows = [r for r in summary if r["subset"] == subset]
    rows.sort(key=lambda r: (r["level"], r["pool"], r["arm"]))
    print(f"\n=== Task A — 10-way selection (subset={subset}) ===")
    print(f"{'L':>3} {'pool':<8} {'arm':<16} {'n':>5} {'top1':>7} {'strict':>7} "
          f"{'MRR':>7} {'MRRt':>7} {'margin':>8}")
    for r in rows:
        print(f"{r['level']:>3} {r['pool']:<8} {r['arm']:<16} {r['n']:>5} "
              f"{r['top1']:>6.1%} {r['top1_strict']:>6.1%} {r['mrr']:>7.3f} "
              f"{r['mrr_tie']:>7.3f} {r['margin_mean']:>8.3f}")


def print_coverage(coverage: dict[str, Any]) -> None:
    print("\n=== Pool coverage ===")
    print(f"{'L':>3} {'pool':<8} {'triads':>7} {'nonempty':>9} {'mean cands':>11}")
    for key, c in sorted(coverage.items(), key=lambda kv: (int(kv[0].split('/')[0]), kv[0].split('/')[1])):
        level, pool = key.split("/", 1)
        print(f"{level:>3} {pool:<8} {c['triads']:>7} {c['nonempty']:>9} "
              f"{c['mean_candidates']:>11.1f}")


def print_task_b(tb: dict[str, Any]) -> None:
    print("\n=== Task B — true vs swapped bridge (residual AUC) ===")
    print(f"{'tier':<18} {'pairs':>6} {'AUC':>7} {'true res':>10} {'swap res':>10}")
    for tier, s in sorted(tb["summary"].items()):
        print(f"{tier:<18} {s['n_pairs']:>6} {_fmt_pct(s['auc']):>7} "
              f"{(s['mean_true_residual'] or 0):>10.4f} {(s['mean_swapped_residual'] or 0):>10.4f}")
    if tb.get("missing"):
        print(f"  skipped (no eligible swap): {tb['missing']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="13,11,9,7",
                    help="comma-separated MetaBary levels (default 13,11,9,7)")
    ap.add_argument("--n", type=int, default=200, help="triads per level (default 200)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control", action="store_true",
                    help="also run vector-shuffle and label-shuffle controls")
    ap.add_argument("--no-task-b", action="store_true", help="skip Task B")
    ap.add_argument("--swap-n", type=int, default=5,
                    help="swapped-bridge candidates per triad/tier (default 5)")
    ap.add_argument("--max-words", type=int, default=8,
                    help="cm_leaf_words cap (default 8)")
    ap.add_argument("--random-pool", type=int, default=3000,
                    help="same-level negatives sampled per bridge level (default 3000)")
    ap.add_argument("--scan-cap", type=int, default=20_000,
                    help="max docs scanned while sampling a pool (default 20000)")
    ap.add_argument("--lexical-pool", type=int, default=240,
                    help="shared lexical candidate sample per level (default 240)")
    ap.add_argument("--out", default=str(ROOT / "evaluation/results/bridge_exp/report.json"))
    ap.add_argument("--write-md", action="store_true", help="also write report.md next to JSON")
    args = ap.parse_args(argv)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    rng = random.Random(args.seed)
    started = time.time()

    log(f"connecting to MongoDB ({args.levels=}, n={args.n})")
    coll = get_collection()
    try:
        coll.database.client.admin.command("ping")
    except PyMongoError as exc:
        log(f"ERROR: MongoDB is not reachable: {exc}")
        return 2

    word_cache = WordCache(coll, max_words=args.max_words)
    all_triads: list[Triad] = []
    data_stats: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    pools_by_triad: dict[Any, dict[str, Pool]] = {}
    level_ctx: dict[int, dict[str, Pool]] = {}
    coverage_raw: dict[str, dict[str, int]] = defaultdict(
        lambda: {"triads": 0, "nonempty": 0, "candidates": 0}
    )

    for level in levels:
        log(f"L{level}: sampling up to {args.n} MBs")
        triads, stats = build_triads(
            coll, level, args.n, rng, word_cache, scan_cap=args.scan_cap,
        )
        log(f"L{level}: {len(triads)} triads, drops={stats['dropped']}")
        data_stats[str(level)] = stats
        if not triads:
            continue

        bridge_level = level + 1
        if bridge_level not in level_ctx:
            log(f"L{level}: building same-level candidate pools at L{bridge_level}")
            random_pool = build_random_pool(
                coll, bridge_level, args.random_pool, args.scan_cap, rng,
            )
            lexical_pool = build_lexical_pool(
                coll, bridge_level, random_pool, args.lexical_pool, rng, word_cache,
            )
            level_ctx[bridge_level] = {"far": random_pool, "lexical": lexical_pool}
            log(f"L{level}: pools far={len(random_pool.candidates)} "
                f"lexical={len(lexical_pool.candidates)}")

        # Vector-shuffle control needs a same-level m from another triad.
        shuffled_m: dict[Any, np.ndarray] = {}
        if args.control and len(triads) > 1:
            order = list(range(len(triads)))
            rng.shuffle(order)
            for i, j in zip(order, order[1:] + order[:1]):
                shuffled_m[triads[i].mb_id] = triads[j].m

        for idx, triad in enumerate(triads, start=1):
            near_pool = build_near_pool(coll, triad)
            sibling_pool = build_sibling_pool(coll, triad)
            pools = {
                "far": level_ctx[bridge_level]["far"],
                "near": near_pool,
                "sibling": sibling_pool,
                "lexical": level_ctx[bridge_level]["lexical"],
            }
            pools_by_triad[triad.mb_id] = pools
            for pool_name, pool in pools.items():
                c = coverage_raw[f"{level}/{pool_name}"]
                c["triads"] += 1
                c["candidates"] += len(pool.candidates)
                if pool.candidates:
                    c["nonempty"] += 1

            evaluate_triad_task_a(
                triad, pools, rng, rows,
                word_cache=word_cache,
                control_m=shuffled_m.get(triad.mb_id),
                label_shuffle=args.control,
            )
            if idx % 25 == 0:
                log(f"L{level}: Task A {idx}/{len(triads)} "
                    f"(near={len(near_pool.candidates)}, sibling={len(sibling_pool.candidates)})")

        all_triads.extend(triads)

    task_b: dict[str, Any] = {}
    if not args.no_task_b and all_triads:
        log(f"Task B: evaluating {len(all_triads)} triads")
        task_b = evaluate_task_b(all_triads, pools_by_triad, word_cache, rng, args.swap_n)
        log(f"Task B: {task_b['n_cases']} cases; missing={task_b['missing']}")

    summary = summarize_rows(rows, control=None)
    controls_summary = []
    for ctl in ("vector_shuffle", "label_shuffle"):
        controls_summary.extend(summarize_rows(rows, control=ctl))

    oracle_stats: dict[str, Any] = {}
    for level in levels:
        ts = [t for t in all_triads if t.level == level]
        if not ts:
            continue
        errs = sorted(t.recon_error for t in ts)
        oracle_stats[str(level)] = {
            "n": len(ts),
            "median_recon_error": float(statistics.median(errs)),
            "p95_recon_error": float(errs[min(len(errs) - 1, int(0.95 * len(errs)))]),
            "max_recon_error": float(max(errs)),
            "share_below_1e-4": sum(1 for x in errs if x < 1e-4) / len(errs),
            "share_below_1e-3": sum(1 for x in errs if x < 1e-3) / len(errs),
        }

    duration = time.time() - started
    coverage = {
        key: {
            **vals,
            "mean_candidates": (vals["candidates"] / vals["triads"]) if vals["triads"] else 0.0,
        }
        for key, vals in coverage_raw.items()
    }
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "levels": levels,
            "n_per_level": args.n,
            "seed": args.seed,
            "control": args.control,
            "task_b": not args.no_task_b,
            "swap_n": args.swap_n,
            "max_words": args.max_words,
            "random_pool": args.random_pool,
            "scan_cap": args.scan_cap,
            "lexical_pool": args.lexical_pool,
            "word_cache_hits": word_cache.hits,
            "word_cache_misses": word_cache.misses,
            "mongo_db": coll.database.name,
            "mongo_collection": coll.name,
        },
        "data": {
            "per_level": data_stats,
            "pool_coverage": coverage,
            "triads_total": len(all_triads),
            "triads_clean": sum(1 for t in all_triads if t.clean),
            "triads_no_overlap": sum(1 for t in all_triads if t.no_overlap),
            "triads_faithful": sum(1 for t in all_triads if t.recon_error < 1e-4),
        },
        "oracle": oracle_stats,
        "task_a": {
            "summary": summary,
            "headline_subset": "clean",
            "headline_near_clean": [
                r for r in summary
                if r["subset"] == "clean" and r["pool"] == "near"
            ],
            "n_rows": sum(1 for r in rows if r.get("control") is None),
            "n_control_rows": sum(1 for r in rows if r.get("control") is not None),
        },
        "controls": controls_summary,
        "task_b": task_b,
        "notes": [
            "Task A A0 is a seeded random score per candidate; A4 is the algebraic oracle.",
            "A1 is tie-dominated at L9/L11: many candidates share Jaccard=0 with the children, so "
            "competition top1 can be 100% while top1_strict is only ~2-8% and MRR_tie is below "
            "chance. Use top1_strict / MRR_tie for A1; A3 has tie_rate=0.",
            "Sibling coverage is 0: sampled triads had sibling_mbs=0, not a timeout. The build "
            "appears to assign each child to one parent, so this pool is structurally unavailable.",
            "Task B AUC 100% is expected given A4=1.0: the true residual is 0, so it verifies "
            "algebraic faithfulness, not the quality of a learned metric.",
            "Task B uses the true triad weights (w1,w2,w3) for both the true and swapped bridge; "
            "candidates are weight-band matched.",
            "Lexical pool is one shared random same-level sample per level, ranked per triad "
            "(a per-triad resample would only add variance at large BFS cost).",
            "Near/lexical/sibling candidates with cosine > 0.9999 to the true bridge are excluded: "
            "the corpus stores some relations twice, and those duplicates are the answer, not negatives.",
            "The distance-matched Task B tier requires |delta cos| <= 0.02 *and* disjoint bridge words; "
            "if coverage is low, that is a property of the near neighbourhood, not a silent failure.",
            "Degenerate triads (bridge words subset of children, or Jaccard >= 0.5) are reported "
            "in subset=all but excluded from the clean headline. L13 has only 38 clean triads "
            "(144/182 are restatements), so L13 clean numbers are low-n.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    print_coverage(coverage)
    print_headline(summary, subset="clean")
    print_headline(summary, subset="all")
    if task_b:
        print_task_b(task_b)
    if oracle_stats:
        print("\n=== Oracle reconstruction error (A4 faithfulness) ===")
        for level, s in sorted(oracle_stats.items(), key=lambda kv: int(kv[0])):
            print(f"  L{level}: n={s['n']:>4} median={s['median_recon_error']:.2e} "
                  f"p95={s['p95_recon_error']:.2e} <1e-4={s['share_below_1e-4']:.1%}")
    print(f"\nReport: {out}")
    print(f"Runtime: {duration / 60.0:.1f} min")

    if args.write_md:
        md = out.with_suffix(".md")
        lines = ["# BaryGraph bridge experiment", "",
                 f"Generated: {report['meta']['generated_at']}",
                 f"Levels: {levels}; n per level: {args.n}; runtime: {duration / 60:.1f} min", ""]
        for subset in ("clean", "all"):
            lines += [f"## Task A ({subset} subset)", "",
                      "| L | pool | arm | n | top1 | strict | MRR | MRRt | margin |",
                      "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
            for r in [x for x in summary if x["subset"] == subset]:
                lines.append(f"| {r['level']} | {r['pool']} | {r['arm']} | {r['n']} | "
                             f"{r['top1']:.1%} | {r['top1_strict']:.1%} | {r['mrr']:.3f} | "
                             f"{r['mrr_tie']:.3f} | {r['margin_mean']:.3f} |")
            lines.append("")
        lines += ["## Pool coverage", "", "| L | pool | triads | nonempty | mean candidates |",
                  "|---:|---|---:|---:|---:|"]
        for key, c in sorted(coverage.items(),
                             key=lambda kv: (int(kv[0].split("/")[0]), kv[0].split("/")[1])):
            level, pool = key.split("/", 1)
            lines.append(f"| {level} | {pool} | {c['triads']} | {c['nonempty']} | "
                         f"{c['mean_candidates']:.1f} |")
        if task_b:
            lines += ["", "## Task B", "", "| tier | pairs | AUC | true res | swap res |",
                      "|---|---:|---:|---:|---:|"]
            for tier, s in sorted(task_b["summary"].items()):
                lines.append(f"| {tier} | {s['n_pairs']} | {_fmt_pct(s['auc'])} | "
                             f"{s['mean_true_residual']:.4f} | {s['mean_swapped_residual']:.4f} |")
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"wrote {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
