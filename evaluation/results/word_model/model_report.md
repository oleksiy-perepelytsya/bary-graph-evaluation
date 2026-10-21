# Pooled L9-L2 word model

Generated: 2026-10-21T21:35:13.851295+00:00
Runtime: 1.0 min

## Model

Single ridge regression from `mean(cm1 words)`, `mean(cm2 words)`, level one-hot features and word-count features to the real MB vector. Predictions are L2-normalized. The bridge is not an input.

- selected alpha: `1000.0`
- fit on: `train+val`
- features: `2060`

## Split

| L | train | val | test |
|---:|---:|---:|---:|
| 9 | 7,210 | 1,549 | 1,540 |
| 8 | 2,870 | 615 | 615 |
| 7 | 1,580 | 339 | 339 |
| 6 | 645 | 138 | 138 |
| 5 | 291 | 62 | 62 |
| 4 | 177 | 38 | 38 |
| 3 | 54 | 12 | 12 |
| 2 | 53 | 12 | 12 |

Groups are exact cm1+cm2 word sets, so identical inputs never cross splits.

## Validation alpha scan (overall)

| alpha | cos median | cos mean | relres median |
|---:|---:|---:|---:|
| 100 | 0.951 | 0.945 | 0.312 |
| 300 | 0.953 | 0.947 | 0.307 |
| 500 | 0.954 | 0.947 | 0.304 |
| 1000 | 0.954 | 0.948 | 0.303 |
| 2000 | 0.954 | 0.948 | 0.304 |
| 3000 | 0.953 | 0.948 | 0.305 |
| 10000 | 0.951 | 0.945 | 0.315 |

Selected alpha: **1000.0** (highest validation mean cosine).

## Test per level

| L | n | uniform | +bridge words | edge ref | model | delta mean [95% CI] | uniform relres | model relres | level-mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1540 | 0.896 | 0.968 | 0.973 | 0.956 | +0.066 [+0.063, +0.069] | 0.457 | 0.298 | 0.848 |
| 8 | 615 | 0.912 | 0.955 | 0.939 | 0.953 | +0.042 [+0.039, +0.045] | 0.420 | 0.307 | 0.869 |
| 7 | 339 | 0.802 | 0.931 | 0.953 | 0.945 | +0.130 [+0.122, +0.138] | 0.630 | 0.332 | 0.900 |
| 6 | 138 | 0.890 | 0.936 | 0.982 | 0.966 | +0.075 [+0.070, +0.081] | 0.470 | 0.262 | 0.911 |
| 5 | 62 | 0.788 | 0.915 | 0.992 | 0.959 | +0.160 [+0.144, +0.173] | 0.652 | 0.288 | 0.945 |
| 4 | 38 | 0.899 | 0.928 | 0.996 | 0.977 | +0.075 [+0.066, +0.084] | 0.449 | 0.214 | 0.935 |
| 3 | 12 | 0.797 | 0.897 | 0.993 | 0.970 | +0.143 [+0.095, +0.187] | 0.635 | 0.247 | 0.966 |
| 2 | 12 | 0.903 | 0.923 | 0.987 | 0.976 | +0.088 [+0.072, +0.108] | 0.441 | 0.221 | 0.954 |
| all | 2,756 | 0.891 | 0.957 | 0.968 | 0.954 | +0.072 [+0.069, +0.074] | 0.468 | 0.302 | — |

`+bridge words` and `edge ref` are diagnostics, not model inputs. The model sees only child words.

## Test controls (overall)

| predictor | cos median | cos mean | relres median |
|---|---:|---:|---:|
| uniform child-word mean | 0.891 | 0.877 | 0.468 |
| role-equal child mean | 0.891 | 0.877 | 0.467 |
| child+bridge words (diagnostic, not input) | 0.957 | 0.950 | 0.293 |
| train+val level mean | 0.873 | 0.859 | 0.504 |
| train+val global mean | 0.870 | 0.857 | 0.510 |
| c1+c2 edge reference | 0.968 | 0.946 | 0.252 |
| c1+c2+bridge edge reference | 0.992 | 0.984 | 0.128 |
| ridge model | 0.954 | 0.949 | 0.302 |
| ridge, shuffled training targets | 0.867 | 0.852 | 0.517 |
| ridge, cm1/cm2 swapped at test | 0.955 | 0.949 | 0.301 |

## Bridge residual

Residual `r = m - prediction`. A positive `cos(r, bridge)` means the bridge vector still carries unexplained MB signal.

| L | n | model relres | cos(r, bridge vec) | cos(r, bridge words) | uniform relres | cos(uniform r, bridge) | cos(pred, bridge) | cos(m, bridge) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1540 | 0.298 | 0.252 | 0.278 | 0.451 | 0.653 | 0.790 | 0.895 |
| 8 | 615 | 0.307 | 0.275 | 0.353 | 0.414 | 0.597 | 0.867 | 0.964 |
| 7 | 339 | 0.332 | 0.247 | 0.314 | 0.615 | 0.533 | 0.872 | 0.973 |
| 6 | 138 | 0.262 | 0.243 | 0.302 | 0.464 | 0.580 | 0.888 | 0.951 |
| 5 | 62 | 0.288 | 0.195 | 0.227 | 0.648 | 0.441 | 0.912 | 0.975 |
| 4 | 38 | 0.214 | 0.143 | 0.140 | 0.441 | 0.576 | 0.930 | 0.962 |
| 3 | 12 | 0.247 | 0.139 | 0.237 | 0.624 | 0.387 | 0.943 | 0.973 |
| 2 | 12 | 0.221 | 0.176 | 0.205 | 0.439 | 0.602 | 0.926 | 0.971 |
| all | 2,756 | 0.302 | 0.253 | 0.295 | 0.461 | 0.617 | 0.839 | 0.936 |

Centered within-level bridge permutation null: median cos = -0.014 (p95 = -0.009). Uncentered permutation null = -0.119 (shared level geometry, not a zero-centered null).

## Key findings

- Overall test median cosine: 0.891 -> 0.954; mean delta +0.072 [+0.069, +0.074].
- Median residual norm: 0.461 -> 0.302 (34.5% lower).
- Median cosine(residual, bridge): 0.617 -> 0.253; centered permutation null -0.014.
- L9: uniform 0.896 -> model 0.956; +bridge words 0.968; edge ref 0.973.
- `cm1`/`cm2` swap: 0.955 vs 0.954; the learned map is approximately symmetric in the two child roles.

## Files

| file | content |
|---|---|
| `predictions_test.parquet` | test predictions, residuals and per-unit bridge diagnostics |
| `ridge_model.npz` | ridge coefficients, scaler, alpha, feature names |
| `split.json` | unit ids per split |
| `model_report.json` | full metrics and configuration |

## Notes

- The bridge is never a model input. `bridge_words` and the edge references are diagnostics.
- The split groups exact cm1+cm2 word sets; related MBs at different levels can still share sub-words.
- Low-level test sets are small (L2/L3 ~12 units), so their CIs are wide.
