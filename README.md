# BaryGraph Evaluation

Evaluation code and reports for the BaryGraph layered knowledge graph. The main
question is how much of a MetaBary (MB) vector is explained by its child words,
and how much residual signal remains that a bridge relation would need to carry.

The current focus is a word-level model over **L9–L2** (18,401 MB units):
given only the words reachable through `cm1` and `cm2`, predict the real MB
vector. The bridge words/vector are stored for residual and oracle analysis,
but are **never model inputs**.

## Headline results

Pooled ridge regression, grouped 70/15/15 split (identical `cm1`+`cm2` word sets
never cross splits), evaluated on 2,756 held-out units:

| predictor | median cosine | mean cosine |
|---|---:|---:|
| uniform child-word mean | 0.891 | 0.877 |
| level mean (train+val) | 0.873 | 0.859 |
| **ridge model (child words only)** | **0.954** | **0.949** |
| child + bridge words (diagnostic, not input) | 0.957 | 0.950 |
| `c1+c2` edge reference | 0.968 | 0.946 |

- Mean improvement over the child-word baseline: **+0.072 cosine**, 95% CI
  `[+0.069, +0.074]`.
- Median residual norm falls from **0.461 to 0.302** (−34.5%).
- Median `cos(residual, bridge)` falls from **0.617 to 0.253**; the centered
  within-level permutation null is **−0.014**. The bridge still carries
  measurable, level-dependent residual signal, strongest at L9/L8.
- `cm1`/`cm2` swap changes median cosine from 0.954 to 0.955: the learned map is
  approximately symmetric in the two child roles.

Full metrics: [`evaluation/results/word_model/model_report.md`](evaluation/results/word_model/model_report.md).

## Repository layout

```
docs/                          architecture notes and papers
lib/                           read-only MongoDB and vector helpers
scripts/
  parse_word_units.py          phase 1: build the word-level dataset
  train_word_model.py          phase 2: train/evaluate the pooled model
  build_probe_data.py          bridge-probe dataset construction
  train_probe.py               bridge-probe classifiers
  exp_bridge.py                bridge experiment driver
  sweep_probe_c.py             regularization sweep
  summarize_probe.py           report aggregation
tests/unit/                    pytest unit tests
evaluation/results/            tracked reports; large data artifacts are ignored
```

Large generated files (`*.parquet`, `*.npz`) are excluded from git by
`.gitignore`. Regenerate them with the commands below.

## Setup

Requires Python 3.10+ and a running MongoDB with the `barygraph_poc.barygraph`
collection (read-only access is sufficient).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Connection defaults can be overridden with `MONGO_URI`, `MONGO_DB` and
`MONGO_COLLECTION`:

```bash
export MONGO_URI='mongodb://localhost:27117/?directConnection=true'
export MONGO_DB='barygraph_poc'
export MONGO_COLLECTION='barygraph'
```

## Reproduce

Phase 1 — parse L9–L2 into word-level units and a word vocabulary:

```bash
python3 scripts/parse_word_units.py --levels 9-2 --workers 8 --chunk 100
```

This writes `units.parquet`, `words.parquet`, `parse_report.json` and
`README.md` to `evaluation/results/word_units/`.

Phase 2 — train and evaluate the pooled ridge model:

```bash
python3 scripts/train_word_model.py --seed 42
```

This writes `model_report.md`, `model_report.json`, `predictions_test.parquet`,
`ridge_model.npz` and `split.json` to `evaluation/results/word_model/`.

Run the tests:

```bash
python3 -m pytest tests/unit -q
```

## Word vector rule

For each `(word, pos)` key, the canonical vector is the mean of its L14
word-node vectors when at least one is reachable; otherwise it is the mean of
its L15 sense vectors. The vocabulary row records `source_level`, `n_nodes`,
`n_l14` and `n_l15`.

## Model scope

The reported model is a **ridge regression with 2,060 features**, not a
large language model. It maps mean `cm1` words, mean `cm2` words, level
one-hot features and word-count features to a 1024-d vector, then L2-normalizes
the prediction. The reported scores are held-out test metrics, not training-set
reconstruction. See the report's notes for the small L2/L3 test sets and the
grouped-split caveats.
