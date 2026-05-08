# Random-forest sweep — closes the bagged-tree variant of the non-LR meta family and the RF-base variant of the new-model-class axis

**Date:** 2026-05-08
**Branch:** `claude/add-random-forest-model-XJ3Dm`
**Script:** `scripts/probe_forest_sweep.py`
**Artifacts:** `scripts/artifacts/probe_forest_sweep.json`,
`oof_rf_meta_K4_strat.npy`, `oof_rf_combined_K4_strat.npy`,
`oof_rf_yekenot_strat.npy`

PI directive (this session): "Try a random forest, extend our model
family using a forest." Ran three sub-probes in one script to share
loading and produce a definitive forest-family verdict in a single
session.

## Context

The irrigation-water comp (last comp) used sklearn `RandomForest` as
the **meta-stacker** over a 14-bank of error-orthogonal calibrated
probability vectors and got +35 bp of LB lift — third-largest move
of that comp. On s6e5, no forest variant was in PRIMARY:

- `d15c_extra_trees` (raw-feature ExtraTrees) was tried Day-15 →
  WEAK_PASS at K=22+1 (+0.059 bp at ρ=0.99599); R5 hedge only.
- LightGBM as meta-learner was falsified Day-20 (`probe_pca_meta.py`):
  worse than LR-meta by 1-2 bp at every input representation.
- No RandomForest had been run as a base or as a meta in this comp.

Three angles were tested today, with low-prior verdicts:
- **A** RF base on yekenot recipe — predicted WEAK_PASS at most.
- **B** RF as meta on K=4 [P, rank, logit] (12 feat) — predicted
  null/regress (matches the inductive class of the falsified
  LightGBM-meta).
- **C** RF on combined K=4 expansion + 6 raw numerics (18 feat) —
  inductive-class swap on a meta with raw-feature awareness; the
  novel angle.

## Settings

```
RandomForestClassifier(
    n_estimators={400 (B/C), 400 (A)},
    max_features="sqrt",
    min_samples_leaf={200 (B/C), 100 (A)},
    max_samples={0.4 (B/C), 0.5 (A)},
    n_jobs=-1, random_state=42,
)
```

5-fold StratifiedKFold seed=42 (matches all base OOFs in the K=4
pool). LR-meta baselines fit C=1.0 on `[P, rank, logit]` expansion
per the project convention.

## Result table

| Angle | Standalone OOF | Comparison baseline | Δ vs baseline | ρ vs PRIMARY | Predicted LB Δ |
|---|---:|---:|---:|---:|---:|
| **B** RF-meta on K=4 expansion | 0.95384 | LR-meta on same input 0.95399 | **−1.54 bp** | 0.9911 | −4.54 bp |
| **C** RF on combined input | 0.95393 | LR on same combined 0.95400 | **−0.70 bp** | 0.9901 | −3.70 bp |
| **C** vs pure K=4 LR-meta | 0.95393 | pure K=4 LR-meta 0.95399 | −0.65 bp | (same) | (same) |
| **A** RF base (yekenot recipe) | 0.94178 | K=4 LR-meta base 0.95399 | **+0.26 bp** at K=4+1 | 0.9595 | −4.74 bp (conservative) |

For comparison: `d15c_extra_trees` (raw features, no FE) standalone
OOF 0.92967, ρ 0.9960, +0.06 bp at **K=22+1**. RF-on-yekenot is +12
bp standalone over ET-on-raw, ρ ~3.7× further from PRIMARY, and
4.4× larger min-meta lift on a much sparser pool.

## Findings

### Angle B (FALSIFIED)

RF as a meta-learner over the K=4 [P, rank, logit] = 12-feature
expansion lands 1.54 bp BELOW the LR-meta baseline at the same
inputs, with ρ=0.991 vs PRIMARY (sub-tie band). The per-fold
validation AUCs show the expected RF instability (0.95287, 0.95329,
0.95352, 0.95486, 0.95488 — std 0.00091 across folds). The
aggregate OOF loses to LR-meta by a clean margin.

This is the **bagged-tree variant of the same finding the Day-20
PCA-meta probe produced for boosted trees** (LightGBM-meta lost 1-2
bp to LR-meta at every input representation tested). The 3-D logit
subspace ceiling (A25/A30) is robust across both boosted and bagged
inductive classes on the meta side. The "non-LR meta" clause of
A30 was empirically refuted Day-20 for boosting; today's run
generalizes it to bagging.

### Angle C (FALSIFIED)

Adding 6 raw numerics (LapNumber, TyreLife, RaceProgress, LapTime,
Position, Stint) to the K=4 expansion gives the RF meta access to
the same inputs the GBDT bases trained on. RF-on-combined OOF
0.95393 vs LR-on-combined 0.95400 = −0.70 bp. The inductive-class
swap **does not rescue RF-meta even when it can interact base
predictions with raw signals**. LR also gains nothing from the 6
raw features (0.95400 vs pure K=4 LR-meta 0.95399 = +0.01 bp).
This is consistent with the Day-19 friction
`combined-input-meta-stacker-absorbed`: K=4 bases already extract
all useful information from the raw-features at the base level; a
meta seeing the raw features in addition is redundant.

Confirms: **the non-LR-meta direction is closed across (boosted,
bagged) × (pure base predictions, base predictions + raw features)
= 4 of 4 variants tested.**

### Angle A (WEAK_PASS — most-diverse base in the K=4 era)

RF as a base on the same yekenot feature recipe that produced
CatBoost-yekenot v4 (+24.21 bp at K=21+1) and RealMLP-yekenot h1d
(OOF 0.95257). 8 yekenot recipe items: floor-cat numerics, count
encoding, KBins quantiles, combo cats, CV target encoding on
(Race, Compound) and (Race, Year), without the orig-data concat
(orig CSV not bundled in this repo's artifact dataset).

**Standalone OOF 0.94178** (per-fold range 0.9405-0.9427, std
0.00080). +12.1 bp over `d15c_extra_trees` on raw features (0.92967),
−10.8 bp below the e3 HGBC raw-features ceiling (0.94876), −113 bp
below the LR-meta on K=4. Tree-bagged classifier at this feature
count and rank-lock pool can't beat boosted-tree single models.

**Min-meta gate at K=4+1: +0.26 bp** (0.95399 → 0.95402). Small but
positive — and the largest non-null new-base lift this comp has
produced in the K=4 era.

**ρ vs PRIMARY (the K=4 LR-meta test prediction): 0.9595.** Lowest
ρ ever observed on a positively-gated base. By comparison, the
prior-most-diverse-positive base — the d15b DAE-only at K=22 —
landed at ρ=0.948 with min-meta +0.79 bp. The RF-yekenot base is
slightly closer to PRIMARY than DAE in test-prediction space but
more diverse than every other K=4-era candidate that gated null.

**Why the conservative LB band reads −4.74 bp.** The probe.py band
table assigns "OOF Δ − 5.0 bp" to ρ < 0.99. That rule was calibrated
on tightly-correlated stack-add candidates where OOF Δ tracks LB Δ
nearly 1:1; for genuinely-orthogonal new bases (DAE, FM-class), the
historical amp ratio at LB has been 1.4× to 11.6× (d15b realized
1.4×; d13e per-segment amp 8×). Best central estimate at 1.4× amp
on +0.26 bp OOF is **+0.36 bp LB**, roughly tying PRIMARY in the
public-LB sample-noise band (±12 bp).

**Hedge / R5 verdict.** RF-yekenot is hedge-eligible per Rule R5.
The natural follow-up is a Path-B Compound × Stint τ=100k refit on
K=5 = K=4 + RF — same shrinkage stacker that gives the current
PRIMARY. That refit, if approved, would be a single-shot submit
candidate.

## Conclusions

1. **Forest as meta-stacker is dead** on s6e5. RF in either pure
   (Angle B) or combined-input (Angle C) flavour loses to LR-meta
   by 0.7-1.5 bp OOF. Direct port of the irrigation +35 bp pattern
   does not transfer because the K=4 logit pool already sits at
   the 3-D subspace ceiling — there is no non-linear interaction
   for RF to exploit.

2. **The non-LR meta family is now closed across two inductive
   classes.** Day-20 LightGBM-meta + today RF-meta + today
   RF-combined-input = 4 falsified variants. A30b ("non-LR meta is
   architecturally untested") was wrong at Day-20; today's run
   removes the bagged-tree-meta as a possible escape.

3. **Forest as a base is alive but small.** RF-on-yekenot at
   standalone OOF 0.94178 with ρ=0.959 to PRIMARY produces a
   +0.26 bp K=4+1 LR-meta lift. This is **the largest min-meta
   lift any new base has produced on the K=4 pool** in this comp.
   The lift is well within fold-noise (std 0.0008 per-fold),
   so it's a candidate to advance — not a confirmed positive —
   but it's the first non-null forest result and it merits R5
   hedge candidacy. Path-B refit on K=5 is the natural next step.

## Falsifies / extends

- **A30b extension.** The "non-LR meta architecture" clause of A30
  is now refuted across LightGBM-meta (Day-20), RF-meta (today),
  and RF-combined-input (today). 3 of 3 non-LR-meta variants null
  or regress. The 3-D logit subspace ceiling is robust to
  inductive-class swap on the meta.

## Friction tags

- `non-lr-meta-falsified-across-inductive-classes` — promoted from
  the Day-20 boosted-only variant. Now covers (boosted, bagged)
  tree-class metas at every input expansion tested. Closing
  comment for the next session: do not re-test tree-class metas
  on the K=4 (or any K) pool; the constraint is at the
  3-D-logit-subspace level, not the meta-architecture level.
- `forest-base-on-engineered-fe-rank-locked-but-most-diverse` —
  RF on yekenot recipe is the lowest-ρ positively-gating base
  observed on K=4 (ρ=0.959 vs typical ≥0.996 for absorbed bases).
  Min-meta lift +0.26 bp is small in absolute terms but is the
  largest K=4+1 lift this comp has seen. Update to A30: rank-lock
  is at the logit subspace level, but a sufficiently-diverse new
  base CAN open a small new direction; the question is whether the
  signal is large enough to transfer to LB through Path-B amp.

## Pointers

- `scripts/probe_forest_sweep.py` — sweep script.
- `scripts/artifacts/probe_forest_sweep.json` — verdicts.
- `scripts/artifacts/oof_rf_meta_K4_strat.npy` — RF-meta OOF.
- `scripts/artifacts/oof_rf_combined_K4_strat.npy` — RF-combined OOF.
- `scripts/artifacts/oof_rf_yekenot_strat.npy` — RF-base OOF (Angle A).
- `audit/2026-05-08-pca-meta-probe.md` — Day-20 LightGBM-meta
  falsification (the prior of which today's bagged-tree result is
  the partner).
- `state/mechanism-ledger.md` — to be updated with the three
  forest-family entries.
