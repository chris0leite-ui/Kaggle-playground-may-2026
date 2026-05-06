# Day-13c — Strat pool refactor: drop-d9c works, drop-GBDT FALSIFIED

**Branch**: `claude/review-ml-handover-VTvWw`.
**Load-bearing**: revises Move C thesis from Day-12 Option 1 reading.

## Hypothesis

d12 Option 1 found GBDTs leak −200 to −343bp under GroupKFold and
FM/rule bases leak only −9 to −54bp. d13b GKF stack matrix confirms
d9c_FM is substitutable by d9f + d13a partition combination (−0.01bp
GKF). Hypothesis: dropping leak-eating GBDTs on Strat will also be
free (or improve), validating a deeper Move C refactor.

## Build

LR-meta only (no retraining, ~30s wall) on existing _strat.npy
artifacts. PRIMARY = d9i_S1_K21_swap_aug2way (Strat 0.95071, LB 0.95034).

Four pool variants (subtract from S3 K=24 baseline):
- T0_S3_K24 — baseline reproduction
- T1_drop_d9c — K=23 (drop d9c_FM only — Move C minimal)
- T2_drop_d9c_e5 — K=22 (also drop e5_optuna_lgbm — heaviest GBDT
  leak-eater per Day-12, GKF Δ=−215bp)
- T3_drop_3leak — K=21 (also drop cb_slow-wide-bag, GKF Δ=−247bp)

Submit gate: OOF ≥ 0.95071 AND ρ < 0.9995 → PRIMARY-candidate.

## Results

| Variant | K | Strat OOF | Δ vs PRIMARY | ρ | Verdict |
|---|---:|---:|---:|---:|---|
| T0_S3_K24 | 24 | 0.95073 | +0.20bp | 0.99976 | TIE |
| **T1_drop_d9c** | **23** | **0.95073** | **+0.19bp** | **0.99981** | **TIE** ✓ Move C |
| T2_drop_d9c+e5 | 22 | 0.95046 | **−2.54bp** | 0.99862 | REGRESS ✗ |
| T3_drop_d9c+e5+cb_swb | 21 | 0.95045 | −2.62bp | 0.99850 | REGRESS ✗ |

## Two hard findings

### 1. d9c_FM is substitutable on Strat too — Move C minimal CONFIRMED

T1 (K=23) ≈ T0 (K=24) within noise (−0.01bp Strat). Mirrors d13b GKF
result (SWAP_21 ≈ FULL_22). **Drop d9c is free** on both axes.

Practical: T1 K=23 frees one pool-budget slot vs T0 K=24 for future
new bases. The L1 ranking shifts modestly (no FM_d9c slot to compete
for), but no other base loses meaningful weight.

### 2. GBDT leak-eaters are NOT substitutable — Move C deep FALSIFIED

T2 (drop e5_optuna_lgbm in addition) → −2.54bp Strat regress.
T3 (also drop cb_slow-wide-bag) → −2.62bp.

The LR-meta routes around the FM/rule bases by leveraging GBDT
row-extreme signal that exists in the i.i.d. public LB. Under GKF
the GBDTs leak fold-mate signal (−209 to −247bp), but **public LB
is row-iid (U3)** — the leakage they "eat" under GKF is real signal
in the test rows.

## Reframe: public LB ≠ GKF on pool composition

| Decision | GKF Says | Strat Says | Verdict |
|---|---|---|---|
| Drop d9c_FM (FM redundancy) | Free (−0.01bp) | Free (−0.01bp) | DO IT ✓ |
| Drop e5_optuna_lgbm | Free (huge GKF gain) | Costs 2.5bp | DON'T |
| Drop cb_slow-wide-bag | Free | Costs more | DON'T |

GKF substitutability is necessary but **not sufficient** for pool removal.
For leak-eaters, only Strat is the truth.

## Revised Move C thesis (load-bearing for Day-14+)

- ✓ Drop d9c_FM cheaply (1 pool slot freed)
- ✗ Cannot drop GBDT leak-eaters (LR-meta routes through them on row-iid)
- → Diversification frontier remains within leakage-robust population
  (more FM-input axes, hier-meta variants, NN model classes)

This unifies Day-12 Option 9 finding ("K=21 stack works because
LR-meta routes between leakage-eaters and leakage-robust bases —
public LB is row-iid so PRIMARY survives, but the diversification
we need is WITHIN the leakage-robust population") with Day-13 PM
results.

## Submit candidate

T1_drop_d9c K=23 (Strat 0.95073, ρ 0.99981) is marginally cleaner
than T0 K=24 (1 fewer redundant base, same Strat). FM-class
amplification didn't fire on d13a S3 K=24 LB (TIE 0.95032), so T1
likely submits TIE too — not a calibration probe of high information
value. Held; d9c-drop refactor should be incorporated into next
PRIMARY base build instead of submitted standalone.

## Pointers

- `scripts/d13c_strat_pool_refactor.py`
- `scripts/artifacts/d13c_strat_refactor_results.json`
- Companion: `audit/2026-05-13-d13b-gkf-full-stack.md` (GKF-side)
- Friction: `gkf-vs-strat-stack-pool-refactor-asymmetry`
- Held submission: `submissions/submission_d13c_T1_drop_d9c.csv`
