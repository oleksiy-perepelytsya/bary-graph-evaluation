# Word-level MB units (L2–L9)

Generated: 2026-10-21T20:31:52.393728+00:00
Units: 18,401; vocabulary: 213,574 (word, pos) keys; runtime: 6.4 min

## Files

| file | content |
|---|---|
| `units.parquet` | one row per MB: vector, child/bridge vectors, word keys |
| `words.parquet` | one row per `(word, pos)` with the canonical 1024-d vector |
| `parse_report.json` | counts, drops, per-level stats, sanity baselines |

## Word vector rule

For each `(word, pos)`, use the mean of its L14 word-node vectors when at least one is reachable; otherwise use the mean of its L15 sense vectors. The `words.parquet` row records `source_level` (14 or 15), `n_nodes`, `n_l14` and `n_l15`.

## Per-level counts

| L | candidates | parsed | words/MB (cm1) | words/MB (cm2) | bridge | total |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 10,299 | 10,299 | 6 | 6 | 8 | 20 |
| 8 | 4,100 | 4,100 | 8 | 8 | 13 | 28 |
| 7 | 2,258 | 2,258 | 11 | 12 | 16 | 36 |
| 6 | 921 | 921 | 16 | 16 | 24 | 55 |
| 5 | 415 | 415 | 20 | 20 | 31 | 69 |
| 4 | 253 | 253 | 31 | 31 | 37 | 100 |
| 3 | 78 | 78 | 38 | 38 | 62 | 135 |
| 2 | 77 | 77 | 62 | 62 | 74 | 197 |

## Sanity baselines (median cosine)

| L | children words | +bridge words | child-edge oracle | +bridge edge | global mean (LOO) |
|---:|---:|---:|---:|---:|---:|
| 9 | 0.895 | 0.969 | 0.971 | 0.993 | 0.849 |
| 8 | 0.909 | 0.955 | 0.936 | 0.983 | 0.872 |
| 7 | 0.801 | 0.930 | 0.955 | 0.986 | 0.902 |
| 6 | 0.896 | 0.932 | 0.981 | 0.999 | 0.913 |
| 5 | 0.788 | 0.917 | 0.993 | 0.999 | 0.941 |
| 4 | 0.904 | 0.926 | 0.995 | 0.999 | 0.938 |
| 3 | 0.767 | 0.897 | 0.995 | 0.998 | 0.963 |
| 2 | 0.898 | 0.924 | 0.989 | 0.996 | 0.958 |

`global mean (LOO)` excludes the target MB from the level mean.

The bridge is stored, not used as a model input. The gap between children words and children+bridge words is the residual the bridge must explain.
