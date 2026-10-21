# BaryGraph bridge weights by level

Generated: 2026-10-21  
Scope: read-only collection from `barygraph_poc.barygraph` (MongoDB 8.2.6)  
Data: 5,940,987 docs total — 2,681,909 `baryedge`, 3,259,080 `node`

This report measures the **gradient of `accumulated_weight` and `connection_strength`
for bridges by level**. It is the data behind the level-scope decision to train on
**MetaBary (MB) levels L9, L11, L13** and to exclude weights from model features
and evaluation.

Machine-readable versions: [`level_gradient.json`](level_gradient.json),
[`level_gradient.csv`](level_gradient.csv).  
Model direction: [`design_notes.md`](design_notes.md).

## Definitions

- A **bridge** at level `L` is a `baryedge` whose `parent_edge_id` points to a
  baryedge at `L−1`.
- A **child** at level `L` points to a baryedge at `L−2`.
- Every baryedge at level `P` produces one bridge at `P+1` and two children at
  `P+2`. Verified on 200/200 L9 MBs: exactly 1 bridge at L10 + 2 children at L11.

Classification is by parent level, not merely by `parent_edge_id != null`; that
distinction matters because a doc with a parent can be either a bridge or a child.
Means/min/max are exact full-population aggregates. Quantiles are exact
`$percentile` results for L11–L14 and full-population values for L3–L10.

## Bridges by level

`w` = `accumulated_weight`; `cs` = `connection_strength`.

| bridge level | bridges | w mean | w p5 | w p50 | w p95 | w max | cs mean | cs/w | Δw | ×prev |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 77 | 0.563 | 0.004 | 0.437 | 1.408 | 1.423 | 0.395 | 0.703 | — | — |
| 4 | 78 | 0.626 | 0.063 | 0.594 | 1.248 | 1.283 | 0.452 | 0.722 | +0.064 | 1.11 |
| 5 | 253 | 0.625 | 0.105 | 0.578 | 1.327 | 1.346 | 0.464 | 0.743 | −0.002 | 1.00 |
| 6 | 415 | 0.830 | 0.324 | 0.807 | 1.308 | 1.308 | 0.635 | 0.765 | +0.205 | 1.33 |
| 7 | 921 | **1.119** | 0.705 | 1.259 | 1.269 | 1.269 | 0.882 | 0.788 | +0.289 | 1.35 |
| 8 | 2,258 | 0.975 | 0.005 | 1.202 | 1.231 | 1.231 | 0.793 | 0.813 | −0.144 | 0.87 |
| 9 | 4,100 | 0.845 | 0.038 | 1.024 | 1.192 | 1.192 | 0.709 | 0.839 | −0.131 | 0.87 |
| 10 | 10,299 | 0.509 | 0.058 | 0.385 | 1.135 | 1.154 | 0.441 | 0.867 | −0.336 | 0.60 |
| 11 | 18,499 | **0.355** | 0.133 | 0.239 | 0.894 | 1.106 | 0.319 | 0.897 | −0.153 | 0.70 |
| 12 | 94,623 | 0.395 | 0.227 | 0.350 | 0.704 | 0.882 | 0.367 | 0.929 | +0.039 | 1.11 |
| 13 | 131,621 | 0.622 | 0.362 | 0.644 | 0.734 | 0.837 | 0.599 | 0.963 | +0.227 | 1.58 |
| 14 | 509,204 | 0.900 | 0.600 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | +0.278 | 1.45 |
| 15 | **0** | — | — | — | — | — | — | — | — | — |

`Δw` is the change in mean weight from the previous bridge level; `×prev` is the
ratio.

## `connection_strength` is a level rescaling of `accumulated_weight`

The two fields are not independent. For every sampled baryedge — bridges,
children, and roots — the ratio is exactly constant within a level (one unique
value at 6 decimals, zero variance):

```text
connection_strength = min(1, 26 / (40 − L)) × accumulated_weight
```

| level | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| factor | .703 | .722 | .743 | .765 | .788 | .813 | .839 | .867 | .897 | .929 | .963 | 1.000 | 1.000 |

The cap at 1.0 explains why `cs` never exceeds 1 while `w` reaches 1.4. At L7–L10
the observed maximum `w` sits exactly on the implied ceiling `(40−L)/26`; at
L11–L13 the data does not reach that ceiling, so `cs` stays below 1.

**Consequence:** `cs` carries no independent information. Feeding both fields to
a model would be redundant. `cs` is a per-level normalized copy of `w`.

## Shape of the weight gradient

Three regimes:

1. **Buildup, L3→L7:** +0.556 total, peaking at **1.119** at L7. 75% of L7
   bridges are above 1.0.
2. **Collapse, L8→L11:** −0.764 total. The largest step is **L9→L10, −0.336
   (×0.60)**; the trough is **0.355 at L11**.
3. **Recovery, L12→L14:** +0.505 total, returning to **0.900** at L14. 26% of
   L14 bridges sit at the 1.0 ceiling.

The weight is **not monotonic in level**. It is also bimodal at several levels:
L8 has p5 = 0.005 but p50 = 1.202; L3 has p5 = 0.004 but p95 = 1.408. A mean
alone hides a near-zero population plus a strong (~1.2) population.

## Bridge vs sibling children at the same parent level

| parent level P | bridge level | bridge mean w | children level | children mean w | bridge / children |
|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 0.563 | 4 | 0.456 | 1.23 |
| 3 | 4 | 0.626 | 5 | 0.800 | 0.78 |
| 4 | 5 | 0.625 | 6 | 0.831 | 0.75 |
| 5 | 6 | 0.830 | 7 | 0.785 | 1.06 |
| 6 | 7 | 1.119 | 8 | 0.932 | 1.20 |
| 7 | 8 | 0.975 | 9 | 0.488 | **2.00** |
| 8 | 9 | 0.845 | 10 | 0.362 | **2.34** |
| 9 | 10 | 0.509 | 11 | 0.430 | 1.18 |
| 10 | 11 | 0.355 | 12 | 0.412 | 0.86 |
| 11 | 12 | 0.395 | 13 | 0.586 | 0.67 |
| 12 | 13 | 0.622 | 14 | 0.846 | 0.74 |
| 13 | 14 | 0.900 | 15 | 0.879 | 1.02 |

The bridge is the dominant child during the L7–L9 compression regime, but weaker
than the sibling children during the L10–L12 expansion regime.

## Level scope decision

Train and evaluate on **MetaBary (MB) root levels L9, L11, L13** — the MB levels
inside the requested L9–L13 range. MBs are root `baryedge` docs (no parent,
`source != structural`). Each MB at level `L` has its bridge at `L+1`.

| MB level | MB count (root, non-structural) | bridge level | bridges at bridge level |
|---:|---:|---:|---:|
| 9 | 1,683 | 10 | 10,299 |
| 11 | 55,526 | 12 | 94,623 |
| 13 | 188,337 | 14 | 509,204 |

MB levels **10 and 12 contain no root MBs** (L10 has 21 `parent=null` docs, all
`structural`; L12 has 0). So the requested L9–L13 range reduces to the odd MB
levels **9/11/13** by construction. This matches the earlier experiment's MB
convention (`--levels 13,11,9,7`), now with L7 dropped.

- Include MB L9, L11, L13: spans maximum bridge dominance (bridge L10), the
  weight trough (bridge L12), and recovery (bridge L14), with 1,683–188,337 MBs.
- Exclude MB L7: extreme weight regime (75% of bridges above 1.0) and already
  represented by the earlier probe.
- Bridge level 14 (from MB 13) was heavily degenerate in the earlier 200-MB
  sample (144/182 triads). The degeneracy filter is mandatory, and L13 must be
  re-checked on the rebuilt cache.

**Do not use `accumulated_weight` or `connection_strength` as model features or
as evaluation targets.** They are construction weights, not semantic quality
scores; `cs` is a deterministic rescaling of `w`. They may be retained only for
stratification, audit tables, and the explicit weight-oracle leakage control.

## Caveats

- L15 has no bridges: all 1,018,408 parented L15 docs are children of L13.
- Bridge counts are approximately 1:1 with the previous level's edge count; a
  few structural/root docs do not emit bridges (e.g. 22 missing at L12).
- Quantiles for L3–L10 are full-population; L11–L14 use MongoDB `$percentile`
  (approximate) over the full population.
- The `26/(40−L)` law is empirical but exact on every doc checked; it is likely a
  pipeline normalization constant, not a semantic "strength" signal.
