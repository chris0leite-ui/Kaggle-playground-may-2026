# Day-4 slot-2 build: YetiRank + Naive Bayes diversity probe

Two new bases built today as candidates for slot 2 on top of M5q
(Strat 0.95057, LB 0.95005). Both are structurally orthogonal to the
M5q pool, both add L1 weight in the LR meta, but BOTH tie M5q at LB
precision when added.

## Results

| Base | Strat OOF | Δ base bp | ρ vs M5q test | M5q+X OOF | Δ M5q bp | Stack ρ vs M5q | L1 rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| d4_cb_yetirank | 0.90508 | −356.7 | **0.66614** | 0.95057 | +0.0 | 0.99966 (TIE) | 11/15 |
| d4_nb | 0.87984 | −609.1 | 0.85259 | 0.95059 | +0.24 | 0.99981 (TIE) | 8/15 |
| **M5z (both, K=16)** | — | — | — | **0.95060** | **+0.30** | **0.99957 (TIE)** | yr=9, nb=14 |

YetiRank achieves the lowest ρ vs M5q test we've ever measured
(0.666). NB at 0.853 also clears the diversity bar. Despite this,
the LR-on-strong-anchor stack saturates and the test rank stays
within Kaggle's 5-decimal quantization of M5q.

## What this proves (Day-3 hypothesis re-confirmed)

`lr-meta-rank-lock-strong-anchor` was logged Day-3 friction. Today
extends the evidence: **even bases with ρ=0.666 underlying
diversity cannot break the lock** when added on top of M5q. The
LR meta over a 14-base GBDT-heavy pool is rank-saturated; adding
orthogonal bases redistributes L1 weight internally (pushing
a_horizon to L1=0.115 in M5z, near-zero) but does not move the
test ranking.

Strategic translation: **slot 2 cannot break LB tie via layered add.**
The remaining EV-positive moves are anchor REPLACEMENT, not addition:
1. Multi-seed RealMLP bag → replaces single-seed RealMLP base.
2. TabNet → second NN family; potential anchor-swap, not just add.
3. NN-with-TE smoke before any Optuna sweep.

(See `audit/2026-05-05-nn-stack-priorities.md` — priority order.)

## YetiRank build details

- `loss_function='YetiRank'` (CatBoostRanker; default Classic mode).
- `group_id` = (Year, Race, Driver) — natural F1 stint unit; 40,869
  groups, median 11 rows/group, ~1 positive/group.
- Cat cols: Driver, Race, Compound, Year (mech #1 win retained).
- Slow+wide hyperparams: lr=0.03, iter=4000 cap, l2=8, depth=6,
  od_wait=100. All folds early-stopped 322-509 iters → cap not
  binding, AUC is genuine not a floor.
- Output: rank-normalized within fold/test (YetiRank scores are
  unbounded; rank-norm preserves AUC, keeps logit features in LR
  meta from blowing up).
- Strat-only (R1). 5-fold wall: 805s = 13.4min (probe projected 14min).
- Per-fold AUC: 0.90269–0.90930 (sd=0.00241; tight).

## NB build details

- GaussianNB (`var_smoothing=1e-7`).
- Numerics → QuantileTransformer (output=normal, n_quantiles=1000).
- Low-card cats (Year, Compound, Stint) → one-hot, full-vocab.
- High-card cats (Driver, Race) → smoothed within-fold target encoding
  (α=20).
- 5-fold Strat, 11.8s total. Per-fold AUC 0.87883-0.88220 (sd=0.00114).

## What's worth keeping

- Both OOF/test artifacts saved for potential anchor-REPLACEMENT
  experiments (e.g., a stack with YetiRank or NB *replacing* M5h
  members rather than adding).
- M5x and M5z held submissions exist but should NOT be submitted
  on the basis of this probe — pre-submit-diff would gate them out
  (ρ ≥ 0.9997).
- Friction lesson reinforced: 3rd independent confirmation that
  layered orthogonal bases on M5q do not move LB.

## Held submissions

- `submissions/submission_m5x_yetirank.csv` — M5q + YetiRank, TIE_EXPECTED.
- `submissions/submission_m5z_yetirank_nb.csv` — M5q + YetiRank + NB,
  TIE_EXPECTED.

End — 65 lines.
