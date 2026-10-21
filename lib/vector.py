"""Vector helpers for the BaryGraph bridge experiment.

The storage format is the one used by the existing BaryGraph PoC:
  - ``Binary`` subtype 9 (BSON BinaryVector, FLOAT32): 2-byte header + f32 data
  - legacy ``Binary`` subtype 0: raw little-endian f32 data
  - plain list/tuple/ndarray

This module is intentionally small and self-contained so the experiment does not
need to import the (much larger) reference project at runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from bson.binary import Binary, BinaryVectorDtype

_DTYPE = np.float32
VECTOR_SUBTYPE = 9
_DTYPE_F32 = BinaryVectorDtype.FLOAT32.value[0]
_DTYPE_I8 = BinaryVectorDtype.INT8.value[0]


def decode_subtype9(blob: bytes) -> np.ndarray:
    tag = blob[0]
    data = blob[2:]
    if tag == _DTYPE_F32:
        return np.frombuffer(data, dtype=np.float32)
    if tag == _DTYPE_I8:
        return np.frombuffer(data, dtype=np.int8).astype(np.float32)
    raise ValueError(f"unsupported BSON vector dtype 0x{tag:02x}")


def unpack_vec(blob: Any, dim: int | None = None) -> np.ndarray:
    """Decode a stored vector into a 1-D float32 array."""
    if isinstance(blob, np.ndarray):
        arr = np.asarray(blob, dtype=_DTYPE)
    elif isinstance(blob, (list, tuple)):
        arr = np.asarray(blob, dtype=_DTYPE)
    elif isinstance(blob, Binary) and blob.subtype == VECTOR_SUBTYPE:
        arr = decode_subtype9(bytes(blob))
    else:
        arr = np.frombuffer(blob, dtype=_DTYPE)
    if dim is not None:
        arr = arr.reshape(dim)
    return arr


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
