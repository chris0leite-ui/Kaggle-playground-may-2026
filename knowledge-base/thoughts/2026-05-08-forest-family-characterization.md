# Forest family characterized end-to-end on s6e5

**Branch:** `claude/add-random-forest-model-XJ3Dm`
**Audit:** `audit/2026-05-08-rf-forest-sweep.md`

This session ran the forest-class probe sweep through to the
operational answer. PI directive at start: extend the model family
with a forest, motivated by the irrigation-water comp's +35 bp
RF-meta result. Answer at end: forest family caps at +0.25 bp K=4+1
LR-meta and Path-B absorbs that lift at LB.

## Three things worth remembering

### 1. The +0.25 bp K=4+1 lift is robust but small.

Across 4 independent RF runs (Angle A yekenot-only / Kitchen-sink
57-feat / Optuna seeds 42 and 7) the K=4+1 LR-meta lift sits in
**+0.24-0.27 bp with std 0.013 bp**. Reproduced across feature
width, hyperparameters, and seed. The signal is real, not fold
noise. It cannot be tuned past this value because the ceiling is
set by the meta architecture (3-D logit subspace, A30), not by
RF itself. Forest base contributes diverse rank information
(ρ=0.96 vs PRIMARY — the lowest ρ ever observed on a positively-
gating base in the K=4 era), but the meta absorbs that diversity
into the same logit-direction subspace.

### 2. Path-B C×S absorbs single-base orthogonal additions below ~+0.5 bp.

K=5 = K=4 + RF Path-B C×S τ=100k OOF 0.95405 vs K=4 PRIMARY 0.95403
= **+0.02 bp**. ρ=0.999917 → tie-band at LB per Rule 27. The +0.25
bp K=4+1 LR-meta gain melts when Compound × Stint per-segment
shrinkage averages the new base's contribution across segments.

This generalizes the Day-15 friction
`path-b-amp-only-fires-on-meta-arch-not-base-add`. Quantitatively:
- DAE base (d15b): +0.715 bp OOF on K=22 → 1.4× amp at LB → +1.0 bp.
  Path-B retained the lift.
- RF base (today): +0.25 bp K=4+1 on K=4 → ~0× amp through Path-B.
  Per-segment shrinkage absorbed the lift entirely.

The threshold for Path-B retention sits between +0.3 and +0.5 bp
standalone OOF lift on a base with ρ≈0.95 vs PRIMARY. Below that,
the per-segment shrinkage averages the contribution to noise.

### 3. The irrigation +35 bp RF-meta does NOT transfer.

Irrigation comp's RF-meta worked on a 14-bank of *already-distilled
error-orthogonal probability vectors* — pre-curated high-signal
features. Raw + engineered features (yekenot recipe, constraint
violations, inter-stint memory) is a fundamentally different regime
where RF doesn't get the same lift from breadth. Adding 19 features
to RF actually HURT standalone OOF by 1.24 bp on this comp because
weak features dilute split capacity at the random-feature-subset
level (RF can't ignore weak features the way boosting can).

## Operational implications going forward

- **Stop the forest axis.** All four reasonable forest configurations
  (default RF / kitchen-sink / Optuna-tuned ×2 seeds) hit the same
  +0.25 bp ceiling. Path-B absorbs the lift at LB. There is no
  remaining forest-family lever with non-trivial EV.
- **The hedge ladder gains 4 entries.** RF-yekenot stack-add plus
  three τ variants of the K=5 Path-B refit (5k / 20k / 100k). All
  R5-eligible for the final-window probe. τ=5k has 121 vs 29
  asymmetric flips on the rare-class top-1% — R7 override territory,
  risky to submit standalone.
- **Rule 27 fired correctly.** ρ=0.999917 > 0.999 abort threshold;
  PI held the slot. 41 of 270 submissions remain used (no change).
- **For the next comp:** the irrigation-RF-meta pattern is regime-
  dependent. The +35 bp move worked because the input features were
  already-distilled probability vectors. On comps where the input
  is raw + lightly-engineered features, expect RF to underperform
  LR-meta and to be hyperparameter-insensitive. Save Optuna budget
  for cases where the input regime is closer to irrigation's.

## Connection to the broader strategic picture

The non-LR meta family is now closed across two inductive classes:
- Day-20 PCA-meta probe: LightGBM-meta worse than LR by 1-2 bp.
- Today: RF-meta (Angles B, C) worse than LR by 0.7-1.5 bp.
- Today: RF-base + Path-B refit: lifts melt to tie-band.

The 3-D logit subspace ceiling (A30) is empirically robust across
boosted vs bagged tree classes on the meta side, AND across single-
base orthogonal additions on the base side. The remaining strategic
options match the existing hypothesis-board open priorities:
1. R5 hedge preparation for the final-window probe (highest EV).
2. RealMLP n_ens=24 (sqrt(n_ens) law gives ≤1 bp).
3. Per-Year CatBoost-yekenot specialists (low confidence).
4. Wrap-up posture; reserve compute for next comp.
5. FastF1 hard-join (cost-prohibitive given match-rate cap).
