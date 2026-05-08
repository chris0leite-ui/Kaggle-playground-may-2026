# D2-A — OOF target encoding (2026-05-04)

TE keys: `['Driver', 'Race', 'Compound', 'Driver_Race', 'Race_Compound']`;
α=80.0; inner 5-fold KFold(seed=42). Raw categoricals KEPT alongside TE
(per analyticaobscura). Recipe direct from
`audit/2026-05-04-similar-comps-research.md` Source 1 #2.

## Results — NULL on both anchors

| anchor | OOF AUC | fold_std | per-fold | Δ vs baseline_two_anchor |
|---|---:|---:|---|---:|
| A — StratKFold | 0.93670 | 0.00076 | 0.9374 / 0.9361 / 0.9371 / 0.9355 / 0.9374 | **−40.5bp** |
| B — GroupKFold(Race) | 0.91628 | 0.01586 | 0.9058 / 0.9058 / 0.8969 / 0.9362 / 0.9327 | **−43.1bp** |

Anchor A std=0.00076 is tight — regression is consistent across folds,
not noise. Both anchors agree on direction. **Gate G1 fails.** Not
submitting; saving submission slot.

## Patch fixed pandas-2.x bug

`scripts/d2a_target_encoding.py` had a `Xtr.loc[idx, "te_<k>"] = arr`
assignment that fails under pandas 2.x strict-coercion (TypeError on
float32 → unknown-column). Fixed by building a `np.float64` column
in-place then assigning whole-column.

## Diagnosis — why TE regressed

Three plausible causes (none confirmed without ablation):

1. **Raw cat already saturating the signal.** LGBM handles 887-level
   `Driver` natively via histogram-based categorical splits; adding
   smoothed TE doubles the same information at lower fidelity (α=80
   shrinks rare drivers toward 0.199 prior). Net: LGBM picks the noisy
   TE first because it's a single-split gain over 887-category split.
2. **Interaction TE on `Driver_Race` (~14k pairs) is noise-dominated.**
   With ~31 rows per pair, smoothed TE at α=80 returns ≈ prior; LGBM
   wastes splits on a near-constant column.
3. **Inner-OOF TE under-shrinks vs analyticaobscura's recipe.** Source
   1 #2 used 5 base models in a stack; TE may only help in the BLEND,
   not the standalone model. Single-model G1 then fails even though
   blend-G2 might pass.

## 3 untried mechanisms (Rule 4 — never lock and stop)

1. **TE replacing raw cat (drop raw)** — test cause #1. If raw is
   saturating, removing it forces LGBM to use the TE column. Cheap;
   one-line edit. Predict: −20 to +5bp; if positive, TE works but only
   without raw.
2. **TE only on `Driver` and `Race` (drop Compound + interactions)** —
   test cause #2. With only 2 low-cardinality TE columns, noise is
   bounded. Predict: −10 to +10bp; close to zero confirms TE-itself
   isn't a winner standalone.
3. **TE as part of a 2-comp blend with baseline** — test cause #3.
   Even if D2-A is −40bp standalone, 0.5·baseline + 0.5·D2-A may lift
   on an OOF basis (blend-G2). Cheap: re-use existing OOF .npy. If
   blend lifts ≥10bp on OOF, single-model gate is too strict for this
   mechanism family.

Priority: **(3) blend-G2 first** (free — uses existing artifacts), then
(2) cheap standalone, then (1).

## Artifacts emitted

- `scripts/artifacts/oof_d2a_te_strat.npy`, `test_d2a_te_strat.npy`,
  `d2a_te_strat_results.json`
- `scripts/artifacts/oof_d2a_te_groupkf.npy`, `test_d2a_te_groupkf.npy`,
  `d2a_te_groupkf_results.json`
- `submissions/submission_d2a_target_encoding.csv` (NOT submitted)
