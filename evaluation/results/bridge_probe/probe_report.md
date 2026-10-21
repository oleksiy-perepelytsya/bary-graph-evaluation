# Learned bridge-selection probe

Generated: 2026-10-21T11:08:23.106540+00:00  
Dataset: `evaluation/results/bridge_probe/dataset.npz`  
Cases: **2154** ([9, 7] levels), 9 hard near-neighbour negatives per case  
Runtime: 759.7 s

## Split

| split | cases | L9 | L7 | exact-group overlap | token overlap |
|---|---:|---:|---:|---:|---:|
| train | 1494 | 1158 | 336 | 0 | 0.0% |
| val | 321 | 239 | 82 | 0 | 0.1% |
| test | 339 | 250 | 89 | 0 | 0.1% |

Random split (leakage comparison only):

| split | cases | L9 | L7 | exact-group overlap | token overlap |
|---|---:|---:|---:|---:|---:|
| train | 1508 | 1153 | 355 | 0 | 0.0% |
| val | 323 | 247 | 76 | 0 | 0.0% |
| test | 323 | 247 | 76 | 0 | 0.1% |

## Key findings

- L9 grouped vector-only: **80.4% strict top-1** vs A3 **62.0%**; delta **+0.184** (95% CI +0.120 to +0.244).
- L9 vector+geom logreg: **84.0% strict top-1**, MRR tie 0.895 (Δ vs A3 +0.220 [+0.160, +0.284]).
- L9 text+geom HGB: **81.2% strict top-1**, MRR tie 0.861 (Δ vs A3 +0.192 [+0.124, +0.264]).
- L9 text-only HGB: **9.2% strict top-1**, MRR tie 0.366 (Δ vs A3 -0.528 [-0.596, -0.456]).
- L7 grouped vector-only: **83.1%** vs A3 **74.2%**; delta **+0.090** (95% CI +0.022 to +0.157).
- Label-shuffle control on L9: **12.0%** strict top-1 (chance ≈ 10% for 10 candidates), so the pipeline itself is not leaking.
- The exact-weight identity baseline is 100% by construction; `weight_delta` is therefore excluded from all learned models.

## Grouped split (primary)

### L9

| model | n | top-1 | strict top-1 | MRR tie | Δ strict vs A3 [95% CI] |
|---|---:|---:|---:|---:|---:|
| A0_random | 250 | 13.2% | 13.2% | 0.323 | — |
| A1_jaccard | 250 | 96.8% | 1.2% | 0.191 | — |
| A2_centroid | 250 | 8.0% | 8.0% | 0.232 | — |
| A3_m_cosine | 250 | 62.0% | 62.0% | 0.708 | — |
| L9 vector-only logreg | 250 | 80.4% | 80.4% | 0.869 | +0.184 [+0.120, +0.244] |
| L9 vector+geom logreg | 250 | 84.0% | 84.0% | 0.895 | +0.220 [+0.160, +0.284] |
| L9 text-only HGB | 250 | 97.6% | 9.2% | 0.366 | -0.528 [-0.596, -0.456] |
| L9 text+geom HGB | 250 | 81.2% | 81.2% | 0.861 | +0.192 [+0.124, +0.264] |
| L9+L7 vector+geom logreg | 250 | 81.2% | 81.2% | 0.880 | +0.192 [+0.128, +0.256] |
| L9+L7 text+geom HGB | 250 | 80.8% | 80.8% | 0.860 | +0.188 [+0.116, +0.260] |

### L7

| model | n | top-1 | strict top-1 | MRR tie | Δ strict vs A3 [95% CI] |
|---|---:|---:|---:|---:|---:|
| A0_random | 89 | 10.1% | 10.1% | 0.306 | — |
| A1_jaccard | 89 | 100.0% | 1.1% | 0.191 | — |
| A2_centroid | 89 | 9.0% | 9.0% | 0.251 | — |
| A3_m_cosine | 89 | 74.2% | 74.2% | 0.793 | — |
| L7 vector-only logreg | 89 | 83.1% | 83.1% | 0.866 | +0.090 [+0.022, +0.157] |
| L7 vector+geom logreg | 89 | 85.4% | 85.4% | 0.890 | +0.112 [+0.045, +0.191] |
| L7 text+geom HGB | 89 | 80.9% | 80.9% | 0.871 | +0.067 [-0.011, +0.146] |
| L9+L7 vector+geom logreg | 89 | 84.3% | 84.3% | 0.891 | +0.101 [+0.045, +0.169] |
| L9+L7 text+geom HGB | 89 | 85.4% | 85.4% | 0.898 | +0.112 [+0.034, +0.202] |

## Random split (leakage comparison)

Bridge-word sets are unique in this dataset, so the random split has little exact/token overlap with train. It is a stability check rather than the primary estimate.

| level | model | grouped strict | random strict | random − grouped |
|---:|---|---:|---:|---:|
| 9 | A3_m_cosine | 62.0% | 66.8% | +4.8% |
| 9 | L9 vector-only logreg | 80.4% | 83.4% | +3.0% |
| 9 | L9 vector+geom logreg | 84.0% | 86.2% | +2.2% |
| 9 | L9 text+geom HGB | 81.2% | 87.4% | +6.2% |
| 9 | L9+L7 vector+geom logreg | 81.2% | 85.8% | +4.6% |
| 9 | L9+L7 text+geom HGB | 80.8% | 88.3% | +7.5% |
| 7 | A3_m_cosine | 74.2% | 71.1% | -3.1% |
| 7 | L9+L7 vector+geom logreg | 84.3% | 86.8% | +2.6% |
| 7 | L9+L7 text+geom HGB | 85.4% | 93.4% | +8.0% |
## Stability check (second grouped split)

Seed 43 changes train/val/test membership. The second grouped split reproduces the vector-only and vector+geom gains. L9 C remains at the 0.001 grid boundary in both seeds; L7 selects interior C values. A finer L9 regularization sweep is the next cheap check.

| level | model | seed 42 strict | seed 43 strict | seed 43 Δ vs A3 [95% CI] | n |
|---:|---|---:|---:|---:|---:|
| 9 | A3_m_cosine | 62.0% | 62.5% | — | 251 |
| 9 | L9 vector-only logreg | 80.4% | 81.7% | +0.191 [+0.131, +0.251] | 251 |
| 9 | L9 vector+geom logreg | 84.0% | 84.9% | +0.223 [+0.163, +0.283] | 251 |
| 7 | A3_m_cosine | 74.2% | 76.1% | — | 67 |
| 7 | L7 vector-only logreg | 83.1% | 86.6% | +0.104 [+0.015, +0.194] | 67 |
| 7 | L7 vector+geom logreg | 85.4% | 89.6% | +0.134 [+0.045, +0.239] | 67 |


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
- Models are selected on validation cases and refit on train+val; test is evaluated once.
- 5/2154 cases had no candidate inside the initial weight band and used the final fallback; weight features are excluded.
- All logistic models selected C=0.001, the lower edge of the grid; a finer regularization sweep is the next cheap check.
- Per-level vector+geom beats pooled L9+L7 on L9 (84.0% vs 81.2%), so pooling is not recommended for the L9 metric.
- Random-vs-grouped differences are small for the vector models (L9 +2 to +3 points), and all 2,154 bridge-word sets are unique: exact-group overlap is 0 and token overlap is ~0.1% in both splits. The random split is therefore a stability check, not a strong leakage stress test. text_geom is more split-sensitive (81.2% grouped vs 87.4% random) and should be treated as less robust.
- Grouped and random test splits share 0 exact bridge-word sets; token overlap is ~0.1% in both, so the random split mainly measures sampling variance.
- no_overlap subset results track all closely: L9 vector-only 80.2% vs 80.4%, L9 vector+geom 83.8% vs 84.0%; full subset metrics are in the JSON.
- text-only HGB is tie-dominated like A1: competition top-1 97.6% but strict top-1 9.2%; it is not a useful selector.
