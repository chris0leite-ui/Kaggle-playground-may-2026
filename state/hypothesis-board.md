# Hypothesis board

Live working notes on what's still open, what's killed, and what's
parked. For history, read `audit/research/` and the postmortems.

## Insights that are still load-bearing

- **The per-segment shrinkage trick fires on a redundant pool, not on a
  saturated one.** It worked when the stack had 21-22 bases of similar
  GBDT/FM material (high redundancy, low effective rank). It stopped
  firing once the stack was small (effective rank ≈ pool size) or once
  the pool already routed by the segmenting variable at the base level.
- **The synthetic data-generating process is conditionally near-independent
  per row.** Self-supervised feature engineering — predict one column
  from the others — produces residual variance ≈ marginal variance.
  Five separate probes confirmed this. Per-row feature engineering on
  these 14 raw columns is a dead axis.
- **The stack of 21-22 bases is rank-locked at the meta-stacker.** Even
  bases with 0.7-0.9 correlation to PRIMARY add zero. The escape was a
  feature-recipe transfer (yekenot), not another base.
- **Target reformulation is a leakage trap on this comp.** Any per-group
  computation that uses the label has to be redone per cross-validation
  fold; otherwise OOF inflates 88-100%. Three reformulations (inv-laps,
  pit-horizon, reverse-cumulative) all collapsed under the strict audit.
- **Public leaderboard is row-i.i.d. with train.** Adversarial-validation
  classifier scores AUC = 0.502; Stratified-Kfold is the LB proxy, not
  GroupKFold. Path-B amp does not transfer to private LB if the test
  partition turns out to be different (unknowable until comp ends).

## Killed — do not retry

- Target-reformulation single-add (inv-laps, pit-horizon, reverse-cumulative,
  stint-progress) — all leaky.
- Anything stacked on top of those leaky targets (path-B with inv-laps
  injected; the "megapool" 25-base variant; the 23-base DAE + inv-laps).
- Multi-level 4-tier per-segment stacker — 5 configurations all null.
- Day-16 virgin axes — 11 of 11 null, falsified, or killed.
- TabPFN v2.5 / v2.6 — AUC ceiling 0.944; v2.6 ran out of memory on
  Kaggle P100.
- 16+ field factorisation machines — saturated at 12 fields.
- Drop-GBDT pool refactor — leak-eaters carry signal that survives at
  the meta on this row-i.i.d. test set.
- Simple K=21 blending (mean / geometric-mean / rank / trimmed) — −19
  to −32 bp; the LR-meta does real work.
- α-calibrated τ resweep — already at the local optimum.
- Multi-target neural network with auxiliary heads — null.
- Masked-column self-prediction (DGP-residuals) — null per the
  conditional-near-independence finding.
- Twin-pool 2-meta blending — collapses rank info.
- Conformal isotonic recalibration of PRIMARY — already globally
  calibrated.
- Adversarial-validation sample weighting — bounded by AV-AUC = 0.502
  (i.e., none).
- Yao/Vehtari covariance-modulated per-segment stacker — overshrinks
  along the highly-correlated base directions the stacker uses for
  routing.
- **Non-LR meta architecture (LightGBM on PCA / raw expansion).**
  PCA-meta probe 2026-05-08 PM: LightGBM meta is *worse* than LR meta
  by 1-2 bp at every input representation tested (PCA top-K for K
  in 3..27, K=10 / K=27 raw [P, rank, logit] expansion). EXP-NEW
  closes FALSIFIED. The "non-LR meta" clause of A30 is empirically
  refuted. See `scripts/probe_pca_meta.py`,
  `audit/2026-05-08-pca-meta-probe.md`, A30b.
- **Path-B segmentation in PC space.** PCA on K=27 logit pool
  decorrelates the routing variables; Path-B C×S on top-K PCs scores
  −28 to −34 bp vs K=10 plain LR. Path-B fires on redundant pools,
  not orthogonal ones. See A30c.

## Open priorities (best EV / cost first)

(Reordered 2026-05-08 PM after EXP-NEW falsification: non-LR meta is
closed, so the "structurally untested architecture" priority drops out.)

1. **R5 hedge preparation for the final-window probe.** List the
   OOF-best candidates that were rejected for public-LB regression.
   Hedge ladder already populated. Cost 30 minutes. **Highest-value
   next move now that all single-axis lift candidates are NULL.**
2. **RealMLP with 24 ensembles** (instead of the current 4). Yekenot's
   published recipe. Predicted +1 to +3 bp standalone; cost 3.5 hours
   GPU on Kaggle. Low confidence — sqrt(n_ens) law gives ≤1 bp.
3. **Per-Year CatBoost-yekenot specialists.** Day-12 found 2023 was the
   easiest year. Predicted ±2 bp; cost 30 minutes GPU. Low confidence.
4. **Wrap-up posture.** Top-11% achieved. Reserve compute for the next
   competition. Durable artifacts already shipped (LR-diagnostic suite,
   BOTE harness, decisions.jsonl, PCA-meta probe).
5. **FastF1 lap-by-lap pit-call hard-join.** Only single-mechanism path
   to top-5. Predicted +10 to +30 bp. Cost: 1-2 days of work, which is
   prohibitive given days remaining and the 1.4% match-rate cap from
   synthetic driver codes.

(Dropped: "Sequence-level DGP fingerprinting" was already closed by
A28 / EXP-1 — the GRU at K=10+1 is NULL; rank-lock is pool-size-
independent.)

## Hedge ladder (final-window candidates)

These don't beat PRIMARY but are eligible for the final-window R7
override probe:

- per-segment Compound × Stint, τ=100k (held)
- ExtraTrees stack-add
- LightGBM-on-kNN stack-add
- 22-base + d12 LR-meta + per-segment, τ=100k
- DAE-only PRIMARY from Day 15 (the τ=20k variant)
