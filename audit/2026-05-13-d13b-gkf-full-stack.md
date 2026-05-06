# Day-13b — Full GroupKF stack matrix: d9c_FM redundant given d9f + d13a

**Branch**: `claude/review-ml-handover-VTvWw`.

## Hypothesis

After d13a confirmed both 5/3 FMs are leakage-robust under GKF, the
question for Move C is: do FM-partition variants STACK additively or
SUBSTITUTE each other in a leak-blocked stack? If d9c_FM (unified
8-field) is substitutable by d9f (4/4) + d13a (5/3) partition pair,
then d9c can be dropped from the pool — Move C minimal refactor.

## Build

- d9f had GKF OOFs but no GKF *test predictions* — rebuild needed.
  Re-trained d9f_FM_A_4 + d9f_FM_B_4 under GroupKFold(Race, Driver,
  Year, Stint) — reproduces Day-12 GKF OOFs to 5 decimals (deterministic
  seed sanity ✓).
- 17 GKF leak-blocked bases (POOL_KEEP minus realmlp [GPU-only],
  m2_xgb [no GKF artifact], R14_L4 [no GKF here]) + 4 FM combos:
  - BASE_18 (17 + d9c_FM only)
  - PLUS_d9f K=20 (add d9f_FM_A + d9f_FM_B)
  - PLUS_d13a K=20 (add d13a FM_A_53 + FM_B_53)
  - **FULL_22 K=22** (all 4 partition FMs)
  - **SWAP_21 K=21** (drop d9c_FM, keep d9f + d13a)

## Results

| Variant | K | GKF AUC | Δ vs BASE_18 |
|---|---:|---:|---:|
| BASE_18 (17 + d9c_FM) | 18 | 0.94575 | — |
| PLUS_d13a K=20 | 20 | 0.94599 | +2.33bp |
| PLUS_d9f K=20 | 20 | 0.94602 | +2.72bp |
| **SWAP_21** (drop d9c) | 21 | 0.94606 | **+3.08bp** |
| **FULL_22** (all 4 FMs) | 22 | **0.94607** | **+3.20bp** |

## Three findings

1. **d9f and d13a STACK, not substitute.** FULL_22 (+3.20bp) beats
   PLUS_d9f (+2.72) and PLUS_d13a (+2.33). Different partition shapes
   carry different signal even though they share fields (D,C,S,T
   overlap).

2. **d9c_FM is REDUNDANT given d9f + d13a.** SWAP_21 (drop d9c)
   = 0.94606 vs FULL_22 = 0.94607 — only −0.01bp. The unified 8-field
   FM is fully substitutable by 4-field + 4-field + 5-field + 3-field
   partition combination.

3. **L1 routing in FULL_22**: rules dominate (rule_drv_cmp 2.634,
   rule_yr_rc 2.568); R6_next 2.016 (#3); d9c_FM #4 (1.617);
   FM_B_53 #5 (1.560); R10_drv_eb #6 (1.455); FM_A_53 #7 (1.451);
   d9f_FM_A #8 (1.388); d9f_FM_B #9 (1.322); R7_prev #10 (1.195).
   **All 5 FMs in top-9; GBDTs (cb_year-cat, e5_lgbm, f1_hgbc)
   demoted out of top-9** under leakage-blocked OOF.

## Strat-side cross-check

d13c (sister probe) confirms the d9c-redundancy on Strat: T1 K=23
(drop d9c) = T0 K=24 (with d9c) — **−0.01bp Strat regress** (within
noise). So the GKF FM-substitutability claim transfers cleanly to
the public-LB axis.

**Critical caveat**: dropping GBDT leak-eaters (e5/cb-bag) costs
−2.5 to −2.6bp Strat — they ARE load-bearing on row-iid public LB.
See d13c audit for the falsified branch of the Move C thesis.

## Move C refactor design (validated)

- ✓ Drop d9c_FM from the pool (free pool-budget slot for new bases)
- ✓ Anchor pool on d9f (4/4) + d13a (5/3) partition FM pair
- ✗ Do NOT drop GBDT leak-eaters — they carry public-LB row-iid signal

## Pointers

- `scripts/d13b_full_gkf_stack.py`
- `scripts/artifacts/d13b_full_gkf_stack_results.json`
- Companion: `audit/2026-05-13-d13c-strat-pool-refactor.md` (Strat-side)
- HEDGE artifacts: `test_d13b_FULL_22_gkf_strat.npy`,
  `test_d13b_SWAP_21_gkf_strat.npy` (R5 final-3-day candidates)
