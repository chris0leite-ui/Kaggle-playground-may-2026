# 2026-05-14 — Evidence the K=11 ceiling is a noise floor, not an architecture limit

Tonight's iteration exhausted five distinct mechanism classes on top of
K=11 + LR-meta + Path-B (LB 0.95386). All five came back NULL or
REGRESSION. Across them, three independent signals converge on the same
hypothesis: we are at or very near the **Bayes-optimal ceiling** of
predictable signal for row-level prediction on this synthetic dataset.

## The three independent signals

1. **The cross-validation gate is no longer trustworthy at wide ρ.**
   A rich-feature LightGBM stacked on top of K=11 lifted cross-validation
   by 18.194 basis points and split-stability-checked clean across two
   seeds. Submitted, the leaderboard *regressed* 15.4 basis points. The
   cross-validation gate was fitting noise patterns that exist on
   training but not on test. This is the canonical signature of
   reaching the noise floor of the data: the residual variance
   *predictable from features* is exhausted, so any new "lift" the gate
   sees is overfitting.

2. **The synthetic generator decouples the two pit columns.** Naive
   reading of the schema says PitNextLap[lap L] = PitStop[lap L+1] when
   both rows are observed. In train: only 80.95% agreement. Per-group
   Spearman between PitStop sum and PitNextLap sum is 0.60. Even when
   *both* of (PitStop_next=1, Stint_next > Stint) fire — what looks
   like an absolute "the driver pits on the next lap" signal — actual
   PitNextLap=1 only 20.7% of the time, essentially the global base
   rate. The synth generator has a stochastic component that
   *intentionally* breaks the temporal consistency.

3. **No row-feature mechanism extracts more than K=11's LR combiner
   does.** Tree non-linearity on top of K=11 + 13 context features:
   regression. Adaptive per-row blending driven by 11-base disagreement
   across 72 hyperparameter combinations: maximum +0.019 bp. Different
   per-segment shrinkage strengths (τ in {5k, 20k, 100k}) on the same
   bases all live in the same logit subspace at ρ 0.9992-0.9998. The
   LR-meta + Path-B combination is already locally optimal for the
   feature space at hand.

## What this means for strategy

The competition has 5 days remaining. K=11 + Path-B at LB 0.95386 is the
strongest defensible PRIMARY. Hedge candidates (K=8 at 0.95382;
K=11 τ-variants at predicted 0.95380-0.95385) are locked in for
final-window R7 selection.

**Mechanism classes still untouched** that could break the ceiling:

- **Cross-domain training** with TRUE labels from the real F1 dataset
  (not just nearest-neighbour lookup). The V4 historical precedent is
  +0.8 bp at the leaderboard.
- **Multi-seed bagging of the FULL K=11 pipeline.** Average across
  random_state in {42, 43, 44, 45}. Reduces meta-layer variance the
  same way bagging traditionally reduces tree variance. Expensive
  (~12 hours full re-build) but the operation is mechanism-pure.
- **Bayesian group-constraint with smoothing.** Even at Spearman 0.60,
  the per-group PitStop sum is a soft prior on per-group PitNextLap
  sum. A small-weight Bayesian shrinkage may shift inter-group rankings
  enough to lift +0.1-0.3 bp. The hard-replacement version was DEAD
  but a partial-trust version is still in play.

**Mechanism classes confirmed dead tonight:**

- Loss-function diversity at the base layer (the wider-feature LGBM
  was a feature-set diversity probe more than a loss probe, but the
  result confirms loss-as-mechanism is dead on top of K=11).
- Tree recalibration of K=11 predictions.
- Per-row adaptive blending by 11-base disagreement.
- Observable lead-feature trick (lap L+1's PitStop column does not
  recover PitNextLap[L]).
- Path-B τ-axis variation as a blend diversity source.

## Process learning

The K=12 regression cost a leaderboard slot and was the highest-magnitude
miss of the session. The lesson — that cross-validation gates fail at
wide ρ — is now formalised in the harness's verdict bands and should
carry over to future sessions and future competitions. **Never trust a
high cross-validation lift from a low-ρ base addition. Always
sanity-check ρ_test against the empirical [0.999, 0.9999] transfer
band before submitting.**
