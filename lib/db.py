"""Read-only MongoDB helpers for the bridge experiment.

Only the functions needed by ``scripts/exp_bridge.py`` live here. They mirror
the reference implementation's semantics (``lib.db`` in the bary-vector repo)
but keep the fresh project self-contained.

Environment overrides:
  MONGO_URI         default mongodb://localhost:27117/?directConnection=true
  MONGO_DB          default barygraph_poc
  MONGO_COLLECTION  default barygraph
  PYMONGO_SOCKET_TIMEOUT_MS default 300000
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, PyMongoError

DEFAULT_URI = "mongodb://localhost:27117/?directConnection=true"
DEFAULT_DB = "barygraph_poc"
DEFAULT_COLLECTION = "barygraph"
VECTOR_INDEX_NAME = "barygraph_vector"


@lru_cache(maxsize=4)
def _cached_client(uri: str) -> MongoClient:
    socket_ms = int(os.environ.get("PYMONGO_SOCKET_TIMEOUT_MS", "300000"))
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=10_000,
        connectTimeoutMS=10_000,
        socketTimeoutMS=socket_ms,
        waitQueueTimeoutMS=5_000,
        maxPoolSize=20,
    )


def get_collection(
    uri: str | None = None,
    db: str | None = None,
    collection: str | None = None,
) -> Collection:
    uri = uri or os.environ.get("MONGO_URI", DEFAULT_URI)
    db = db or os.environ.get("MONGO_DB", DEFAULT_DB)
    collection = collection or os.environ.get("MONGO_COLLECTION", DEFAULT_COLLECTION)
    return _cached_client(uri)[db][collection]


def ping() -> bool:
    try:
        _cached_client(os.environ.get("MONGO_URI", DEFAULT_URI)).admin.command("ping")
        return True
    except PyMongoError:
        return False


def chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    buf: list[Any] = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def vector_search(
    coll: Collection,
    query_vector: list[float] | Any,
    *,
    index: str = VECTOR_INDEX_NAME,
    limit: int = 20,
    num_candidates: int = 200,
    filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """$vectorSearch through mongot; adds ``_score`` to every result."""
    stage: dict[str, Any] = {
        "index": index,
        "path": "vector",
        "queryVector": list(map(float, query_vector)),
        "numCandidates": int(num_candidates),
        "limit": int(limit),
    }
    if filter:
        stage["filter"] = filter
    return list(
        coll.aggregate(
            [
                {"$vectorSearch": stage},
                {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
            ],
            maxTimeMS=120_000,
        )
    )


def cm_leaf_words(
    coll: Collection,
    be_id: Any,
    *,
    max_depth: int = 15,
    max_words: int | None = None,
    timeout_ms: int = 5_000,
) -> set[str]:
    """All words reachable through the cm1/cm2 lineage of one BE/MB.

    BFS stops as soon as ``max_words`` distinct words have been collected (the
    same cap semantics as the reference helper, with an additional early exit
    inside a level). A timeout on one level returns the words found so far.
    """
    if be_id is None:
        return set()

    frontier: set[Any] = {be_id}
    visited: set[Any] = set()
    words: set[str] = set()

    for _ in range(max_depth):
        to_fetch = frontier - visited
        if not to_fetch:
            break
        visited |= to_fetch
        next_frontier: set[Any] = set()
        try:
            docs = list(
                coll.find(
                    {"_id": {"$in": list(to_fetch)}},
                    {"doc_type": 1, "cm1_id": 1, "cm2_id": 1, "properties.word": 1},
                ).max_time_ms(timeout_ms)
            )
        except OperationFailure:
            break

        for doc in docs:
            if doc.get("doc_type") == "node":
                w = (doc.get("properties") or {}).get("word")
                if w:
                    words.add(str(w))
            else:
                if doc.get("cm1_id") is not None:
                    next_frontier.add(doc["cm1_id"])
                if doc.get("cm2_id") is not None:
                    next_frontier.add(doc["cm2_id"])
            if max_words is not None and len(words) >= max_words:
                return words

        if max_words is not None and len(words) >= max_words:
            break
        frontier = next_frontier

    return words
