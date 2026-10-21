# Design notes: vector-first semantic bridge selection

Status: design / not yet implemented  
Context: companion to [`level_gradient.md`](level_gradient.md)  
Scope decision: train and evaluate on **MetaBary (MB) levels 9, 11, 13** and
their bridges (levels 10, 12, 14), excluding `accumulated_weight` and
`connection_strength` from features and targets.

Clarification (project owner): run training on **MBs, not bridges** — the MB
vector is the model input; the bridge is the validation target. In the requested
L13–L9 range this is literally MB levels 9/11/13, because levels 10 and 12
contain no root MBs.

## 1. Direction (from the project owner)

> The model should take the MB vector, provide ideas it can build up from the
> vector itself, and afterward check how this vector is exactly represented in
> the semantic plane (the list of all words included). Based on this, estimate
> how far the initial assumption was from the real semantic field.

In other words: use the vector as the source of a semantic **hypothesis**, then
use the explicit word field as **ground truth** to measure how faithful that
hypothesis is. The bridge is the candidate that makes the vector-implied field
closest to the real field. This inverts the earlier probe, where words were used
as flat re-ranking features and lost to ties.

## 2. Level scope

- MB = root `baryedge` (no parent, `source != structural`). Bridge level = MB
  level + 1. This is the codebase's MetaBary convention (the earlier experiment
  defaulted to MB levels 13, 11, 9, 7).
- Use **MB levels 9, 11, 13** (the requested L13–L9 range). Training runs on the
  MB vectors; bridges at levels 10/12/14 are the targets.

| MB level | MB count (root) | bridge level |
|---:|---:|---:|
| 9 | 1,683 | 10 |
| 11 | 55,526 | 12 |
| 13 | 188,337 | 14 |

- MB levels **10 and 12 are empty** (L10: 21 `parent=null`, all structural;
  L12: 0). The range therefore reduces to the odd MB levels by construction.
- Rationale: 9/11/13 spans the full regime change — maximum bridge dominance
  (bridge L10), the weight trough (bridge L12), and recovery (bridge L14).
- Exclude MB L7 (already sampled; extreme weight regime, 75% of bridges > 1.0).
- **Degeneracy caveat:** MB 13 → bridge L14 was heavily degenerate in the earlier
  200-MB sample (144/182 triads). The degeneracy filter is mandatory and must be
  re-measured on the rebuilt cache. MB 11 previously produced the strongest
  text+geom result (85.8% tight), so it is the internal benchmark level.

## 3. Do not use weights

`accumulated_weight` (`w`) and `connection_strength` (`cs`) are construction
weights, not semantic quality. Formally, `cs` is a deterministic per-level
rescaling of `w` (see `level_gradient.md`):

```text
connection_strength = min(1, 26 / (40 − L)) × accumulated_weight
```

Consequences for this experiment:

- Never feed `w` or `cs` as model features.
- Never use them as labels/targets for "bridge quality".
- Do not band-match negatives by `w`; it biases the pool toward
  weight-similar bridges and reintroduces the excluded signal.
- `w`/`cs` may be kept for three narrow uses only: (a) stratification tables,
  (b) audit/QA, (c) the explicit `L0_weight_identity_oracle` leakage control.

## 4. Model sketch

### 4.1 Inputs

- `m` — the MB vector.
- `c1`, `c2` — the two child edge vectors (already available).
- `b` — a candidate bridge vector (the true one plus hard negatives).

### 4.2 Hypothesis from the vector

Generate a hypothesis `H(m)` about the semantic content the MB vector encodes,
without reference to any candidate bridge. Two viable layers, to be compared:

1. **Vector decoding:** nearest neighbours of `m` in the global word/edge vector
   space, optionally filtered by level/pos; produce a scored word set
   `{(word_i, score_i)}`.
2. **Learned decoder:** fit a small map `m → word distribution` on training MBs
   (multi-label), then generate the top-k hypothesis set at inference.

This step is deliberately candidate-agnostic so the hypothesis cannot leak the
answer.

### 4.3 Semantic plane (ground truth)

The MB's real semantic field is the full leaf word set reached through the
`cm1_id`/`cm2_id` lineage (`cm_leaf_words`):

```text
W(MB) = words(lineage(cm1)) ∪ words(lineage(cm2))
```

For a candidate bridge `b`, the completed field is `W(MB) ∪ words(b)`.
The current `WordCache` caps at `max_words=8`, which truncates exactly the
object this design depends on. The new build must raise or remove the cap
(full field), and record the realized size distribution per level.

### 4.4 Faithfulness / residual metrics

For hypothesis `H` against a field `W`:

- **Precision** — fraction of hypothesized words actually in `W`.
- **Coverage / recall** — fraction of `W` explained by `H`.
- **Residual distance** — distance between the vector-implied distribution and
  the empirical word distribution. Candidate choices: cosine between `m` and the
  centroid of `H`; optimal transport between `H` scores and `W` membership;
  rank-weighted overlap.
- **Assumption error** — how far `H(m)` is from `W(MB)` *before* seeing `b`.
  This is the "how wrong was the first vector assumption" number.

### 4.5 Bridge selection

For each hard negative `b⁻` and the true bridge `b⁺`, score how well the
completed field matches the vector-implied hypothesis:

```text
score(b) = F( H(m), W(MB) ∪ words(b) )
```

Select the candidate maximizing faithfulness. Primary metrics are the same
rank metrics used by the probe (strict top-1, MRR_tie) so results are directly
comparable, plus the new coverage/residual metrics.

## 5. Why this should beat the text-only re-ranker

The earlier `L9 text-only HGB` scored **9.2% strict top-1** (near chance) because
word overlap/Jaccard is dominated by ties: the true bridge and the negatives
share almost the same high-frequency words. The new design does not rank on raw
overlap; it uses words as the semantic target that a vector-derived hypothesis
is scored against. If the vector genuinely encodes the bridge relation, the
faithfulness residual should separate the true bridge even when the word sets
look similar.

## 6. Data requirements and gaps

The existing cache (`evaluation/results/bridge_probe/dataset.npz`) is
**L9/L7 MB levels** → bridge L10/L8, 9 weight-band-matched negatives, and
`max_words=8`. Rebuild is required:

| Requirement | Current | Needed |
|---|---|---|
| Levels | MB 9, 7 | MB 9, 11, 13 (bridges 10, 12, 14) |
| Word field | cap 8 | full leaf set (cap raised/removed) |
| Negatives | weight-band matched | vector-distance hard negatives; no weight band |
| Degenerate triads | filtered | filter; re-measure MB 13 (bridge 14) degeneracy |
| Word sets | bridge/child only | full `W(MB)` plus candidate words |

Notes:

- Earlier `bridge_exp` found heavy degeneracy at **MB 13** (bridge 14):
  144/182 triads degenerate, only 38 clean. This is now inside the scope, so the
  filter is mandatory and the clean count must be checked before committing to
  MB 13.
- `MB 11` previously produced the strongest text+geom result (85.8% tight), so
  it is a good internal benchmark level.
- MB 9 has the smallest population (1,683 roots / 10,299 bridges); MB 13 has the
  largest (188,337 / 509,204). Expect a very unequal level balance; stratify.
- Full leaf sets are expensive; measure per-level word counts and runtime with a
  small `--n` pilot before the full build.

## 7. Evaluation protocol

- Grouped split by bridge-word hash (no bridge-word leakage), seeds 42 and 43.
- Primary observed metric: **strict top-1**; secondary **MRR_tie**, consistent
  with the existing probe so numbers are comparable.
- Baselines: `A3_m_cosine`, vector-only metric, vector+geom, text+geom, random.
- New semantic metrics: hypothesis precision, coverage, residual, assumption
  error.
- Controls: label-shuffle (expect ≈ chance), weight-oracle diagnostic
  (excluded from selection, kept only to prove `w` is not the signal).
- No `w`/`cs` in any model or selection step.

## 8. Open questions / next steps

1. Build a full-word-field cache for MB levels 9/11/13; first measure word-set
   sizes and runtime at a small `--n` pilot before the full build. Re-measure
   MB 13 (bridge 14) degeneracy at the same time.
2. Implement hypothesis generation (`H(m)`) and the faithfulness scorer.
3. Switch negative sampling to vector-distance only.
4. Run grouped seed 42/43 against baselines + controls.
5. Decide from the residual/coverage results whether a GPU or a trained adapter
   is justified; keep CPU-only until then.

Do not commit unless explicitly requested.
