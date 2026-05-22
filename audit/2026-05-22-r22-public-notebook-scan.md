# R22 public-notebook IDEA-scan — 2026-05-22

PI authorized 2026-05-22. Triggered by PI question: "top scorers
hit high scores on first attempts — what are we missing?"

Scope: structural extraction only — no CSV blending per PI directive.

## Notebooks pulled (top 9 by score)

| Notebook | LB-claim | Substance | Verdict |
|---|---|---|---|
| `raunakdey07/f1-pit-stops-blender-0-95453` | 0.95453 | (not pulled — title=blender) | BANNED |
| `anthonytherrien/predicting-f1-pit-stops-nn-residual-network` | n/a | 688 LOC ResNet | trains ResNet then **blends with public CSV** — NN-family already explored (R35 FT-Transformer 0.954086) |
| `azzamradman/knock-the-blender-with-one-liner` | n/a | 4 LOC | pure CSV passthrough — BANNED |
| `safar1/lb-score-0-95452` | 0.95452 | 8 LOC | pure CSV passthrough — BANNED |
| `nawfeelrahman1124444/s6e5-0-95452` | 0.95452 | 32 LOC | pure CSV passthrough — BANNED |
| `safar1/lb-score-0-95449` | 0.95449 | 25 LOC | CSV blender of two public sources — BANNED |
| `nina2025/ps-s6e5-hb8` / `hb5` | n/a | 310 / 425 LOC | elaborate Optuna-weighted CSV blender — BANNED |
| **`arunklenin/ps6e5-f1-pit-stops-prediction-fe-ensemble`** | 0.95453 (in title) | 523 LOC | **REAL FE + external augmentation + tail CSV blend** |

Most "top of LB" notebooks are CSV blenders downstream of `raunakdey07`
or `nina2025`. Only one (arunklenin) does standalone training.

## The finding — external-data training augmentation

`arunklenin` lines 49–79: reads `f1_strategy_dataset_v4.csv`
(`aadigupta1601/f1-strategy-dataset-pit-stop-prediction` — the
exact dataset our `comp-context.md:59` flagged as
`external_data_strategy: use` on Day 1), aligns columns,
concatenates to the training set, dedupes, trains XGB+LGB on
the augmented set with Optuna-tuned blend weights.

**Caveat:** arunklenin's final submission line 509 is
`0.1*own + 0.8*raunakdey_blender + 0.1*nina_blender`. His 0.95453
title-score is dominated by the banned public CSV, NOT his
augmented model. The augmented-model contribution alone is unknown
from the notebook output.

## Why Day-2 missed this

`audit/archive_pre_d13/2026-05-04-d2-probe1-external-join.md`
tested external as a **per-row JOIN** (match rate 5.56% → dead).
**Never tested CONCAT/augmentation** (101K extra labeled rows added
to the training pool, ignoring per-row alignment). These are
different mechanisms; the JOIN-null does not kill CONCAT.

## Heuristic probes (R6) — 2026-05-22

### Schema overlap
Ext shape (101371, 16). Aligns to comp's 15 non-id columns; only
`Normalized_TyreLife` differs (host-forbidden). Comp Year-distribution
(2022–2025: 19%/31%/29%/21%) matches ext (22%/25%/27%/27%).

### Adversarial validation (R25)
- All-features AV-AUC: **0.9866** (almost perfectly separable)
- Numeric-only AV-AUC: **0.7698** (separable but much less)
- Per-feature: Driver=**0.9599**, LapTime_Delta=0.6642, LapTime=0.6480,
  RaceProgress=0.6263, LapNumber=0.6234, all others ≤ 0.61, Year=0.5508,
  Position_Change=0.5029.

### The structural reveal
- Comp train has **887 unique drivers**: 31 real F1 initials (HAM, MAG,
  VET, PIA, …) + 856 synthetic D### codes.
- Ext has **31 unique drivers**, all real F1, ALL present in comp.
- **6.893% of comp train uses real drivers** (30,269 rows); **6.907% of
  comp test uses real drivers** (12,996 rows). Equal fractions ⇒ test
  preserves train's real/synth ratio.
- Pit rate on real-driver comp rows: **0.3327**; on synthetic-driver
  comp rows: **0.1891**. Real-driver subpopulation is structurally
  different (laps closer to pit decisions).
- Ext provides **3.35× more real-driver training rows** (101K vs 30K)
  for that 7% of test.

## Candidate mechanism — real-driver cohort specialist

Train a specialist on `comp_real_driver_rows + ext_rows` (131K rows),
score on the 12,996 real-driver test rows. Synthetic 93% of test stays
on existing PRIMARY (LB 0.95402). Merge predictions by Driver-set
membership.

### Pre-flight 5-question check (R16)
1. Family in `mechanism_families_explored`? **NO** — external-data
   augmentation as cohort-specialist is a new family.
2. Rank-lock vulnerable bucket? **No** — different training data ⇒
   structurally orthogonal to existing 18-base stack.
3. Predicted standalone OOF (real-driver subset only): from PRIMARY
   ~0.954 baseline + 3.35× training data on a homogeneous subpop ⇒
   estimate +1 to +5 bp on the 7% subset. Net comp-AUC lift:
   weighted (0.069 × 0.04) ≈ +0.3 to +1.5 bp.
4. Predicted ρ vs PRIMARY: depends on merge architecture; if cohort-
   specialist replaces PRIMARY on 7% rows only ⇒ ρ ≈ 0.997 (different
   ranks on 13K rows out of 188K).
5. Closest gate-PASS precedent: per-cohort cohort-listwise R21 PASSED
   at G2; per-Year specialists were predicted at ±2 bp.
6. **Q6 (forced) — training objective matches row-AUC metric?**
   **YES** — same target (PitNextLap), same metric (ROC AUC).
   Specialist trained with logloss BCE which optimises AUC well.

### Risks
- AV-AUC 0.9866 means naive full-row concat could inject OOD
  signal even for the 7% real-driver subset. Probe must run
  fold-level AV check on the augmented training set.
- Pit-rate shift (0.33 → 0.25 in ext): probabilities need
  recalibration before merging back into the 0.20-baseline comp space.
- Sample-weighting ext rows (e.g., w=0.5) may be needed; an Optuna
  sweep over weight is a clean R16-friendly probe.
- Per R24/R25: any TE/groupby aggregations on the augmented set must
  refit per fold.

### Cost
Single-model 5-fold on 131K rows ≈ 10–30 min CPU (small subset). Full
sweep with sample-weight tuning ≈ 1 hour. Well under R2 1-hour single-
fold cap.

## Other ideas extracted (lower priority)

- arunklenin's **per-stint lag features** (LapTime_lag{1,2,3},
  LTDelta_lag{1,2,3}, CumDeg_lag1, Pos_lag1) grouped by
  (Year, Race, Driver, Stint). Already covered by our HANDOVER #1
  sequence-fingerprint axis (K=4+1 +0.15 bp marginal).
- arunklenin's **Race-Top-10 OHE + PCA on rare-race-rest** encoding.
  Probably absorbed by our existing TE pool.
- arunklenin's **OptunaEnsemble with logit-parametrised weights**
  (QMC + TPE 250+ trials). Sophistication beyond our current
  simple-mean blend, but the LB-ceiling argument (R31 rank-blend
  saturation) probably applies — single-model weights don't fix
  rank-lock.

## Recommended next move (PI decision)

Open new ISSUES.md leaf: **real-driver-cohort-specialist with
external-data augmentation**. Run R0 standalone-OOF probe on the
real-driver subset; if standalone +bp on the 7%, then R1 weighted-
merge into PRIMARY and ρ-band check before any submit.

Net realistic LB lift estimate: **+0.3 to +1.5 bp** (top-15% → top-10%
boundary at +1.8 bp). Top-5% (+4.7 bp) still out of reach on this axis
alone. Cost: ≤1 hour CPU. Risk: low — augments not modifies PRIMARY.

Days remaining: 10. Hedge ladder unaffected.
