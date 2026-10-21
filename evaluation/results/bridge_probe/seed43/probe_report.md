# Learned bridge-selection probe

Generated: 2026-10-21T11:14:03.506293+00:00  
Dataset: `evaluation/results/bridge_probe/dataset.npz`  
Cases: **2154** ([9, 7] levels), 9 hard near-neighbour negatives per case  
Runtime: 260.5 s

## Split

| split | cases | L9 | L7 | exact-group overlap | token overlap |
|---|---:|---:|---:|---:|---:|
| train | 1536 | 1168 | 368 | 0 | 0.0% |
| val | 300 | 228 | 72 | 0 | 0.1% |
| test | 318 | 251 | 67 | 0 | 0.1% |

Random split (leakage comparison only):

_Random split not run._

## Key findings

- L9 grouped vector-only: **81.7% strict top-1** vs A3 **62.5%**; delta **+0.191** (95% CI +0.131 to +0.251).
- L9 vector+geom logreg: **84.9% strict top-1**, MRR tie 0.906 (Δ vs A3 +0.223 [+0.163, +0.283]).
- L7 grouped vector-only: **86.6%** vs A3 **76.1%**; delta **+0.104** (95% CI +0.015 to +0.194).
- The exact-weight identity baseline is 100% by construction; `weight_delta` is therefore excluded from all learned models.

## Grouped split (primary)

### L9

| model | n | top-1 | strict top-1 | MRR tie | Δ strict vs A3 [95% CI] |
|---|---:|---:|---:|---:|---:|
| A0_random | 251 | 8.0% | 8.0% | 0.278 | — |
| A1_jaccard | 251 | 96.0% | 2.0% | 0.197 | — |
| A2_centroid | 251 | 8.8% | 8.8% | 0.234 | — |
| A3_m_cosine | 251 | 62.5% | 62.5% | 0.720 | — |
| L9 vector-only logreg | 251 | 81.7% | 81.7% | 0.884 | +0.191 [+0.131, +0.251] |
| L9 vector+geom logreg | 251 | 84.9% | 84.9% | 0.906 | +0.223 [+0.163, +0.283] |

### L7

| model | n | top-1 | strict top-1 | MRR tie | Δ strict vs A3 [95% CI] |
|---|---:|---:|---:|---:|---:|
| A0_random | 67 | 6.0% | 6.0% | 0.268 | — |
| A1_jaccard | 67 | 98.5% | 1.5% | 0.194 | — |
| A2_centroid | 67 | 4.5% | 4.5% | 0.194 | — |
| A3_m_cosine | 67 | 76.1% | 76.1% | 0.797 | — |
| L7 vector-only logreg | 67 | 86.6% | 86.6% | 0.904 | +0.104 [+0.015, +0.194] |
| L7 vector+geom logreg | 67 | 89.6% | 89.6% | 0.925 | +0.134 [+0.045, +0.239] |

## Random split (leakage comparison)

Bridge-word sets are unique in this dataset, so the random split has little exact/token overlap with train. It is a stability check rather than the primary estimate.

_Random split not run._

## Leakage diagnostic

The true bridge is the only candidate with `weight_delta == 0` because the near negatives are weight-band matched but not exact. The `L0_weight_identity_oracle` row therefore scores 100% by construction. All learned models exclude `weight_delta`.

An archived run with `weight_delta` included is at `probe_report_with_weight_leak.json`; its text+geom HGB scored 100%, which is the artifact this control detects.

## Notes

- Grouped split hashes the sorted bridge-word set; exact bridge-word sets are disjoint between train and test. Token-level overlap is reported because different bridge-word sets can still share tokens. Random split is a secondary stability check; all 2,154 bridge-word sets are unique, so it is not a strong leakage stress test.
- All cases are clean (non-degenerate), weight-band matched, and exclude near-exact duplicate bridges (cos > 0.9999 to the true bridge).
- weight_delta is excluded from all model features. The true bridge is the only candidate with delta exactly 0, so a weight_delta feature is an oracle identifier (L0_weight_identity_oracle); it is retained only as a leakage diagnostic, not as a selector.
- A1 Jaccard is tie-dominated at L9/L7, so strict top-1 and MRR_tie are the relevant columns.
- The vector-only model uses m*b and |m-b|. It does not see words, child vectors, weights, or bridge ids.
- text_geom is the stronger text/geometry re-ranker baseline; text_only isolates word overlap/counts without any vector or child geometry.
- 5/2154 cases had no candidate inside the initial weight band and used the final fallback; weight features are excluded.
- Models are selected on validation cases and refit on train+val; test is evaluated once.
