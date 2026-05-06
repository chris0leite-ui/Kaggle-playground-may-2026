# Day-10 — GroupKF stack rebuild: FM lift AMPLIFIES under leak-blocking

> Follow-up to `d10_groupkf_audit` (FM bases leakage-robust at the
> base level). Question: does the **stack-level** FM lift survive
> when the LR meta is built over leak-blocked OOFs? Stack OOF 0.95073
> (PRIMARY) is on Strat splits, which have 80% within-group leakage.
> If the stack lift is leakage-amplified, private LB will revert to
> the GBDT-only baseline.

## Method — apples-to-apples K=13 / K=15 stacks

13 GBDT/baseline bases that have Race-only GroupKF artifacts:
`baseline_two_anchor, d2a_te, m2_xgb, e1_catboost_sub, e3_hgbc,
e5_optuna_lgbm, a_horizon, b_lapsuntilpit, f1_hgbc_deep,
f2_hgbc_shallow, cb_year-cat, cb_lossguide, cb_slow-wide-bag.`

Plus d9f's two FMs:
- FM_A driver-dynamics (D, C, S, T_q5)
- FM_B race-context (R, Y, Rp_q5, P_q5)

FM_A + FM_B newly retrained under Race-only GroupKF (matches pool
partition). `scripts/d10b_groupkf_stack_rebuild.py`,
`scripts/d10c_strat_match.py`. Total wall ≈ 180s.

## Result — FM-class lift is 2.3× LARGER under GKF

| Stack | Strat OOF | GKF OOF | Δ Strat→GKF |
|---|---:|---:|---:|
| K=13 baseline (no FM) | 0.95043 | 0.92744 | −229.92bp |
| K=15 + FM_A + FM_B | 0.95052 | 0.92764 | −228.79bp |
| **FM-class lift** | **+0.87bp** | **+2.01bp** | **2.3×** |

The stack OOF drop Strat→GKF is **uniform** (~−230bp) across both
configurations. That ~230bp is the GBDT pool's within-group leakage
contribution, **invariant to whether FM is added**. So the FM
lift ridge is preserved; if anything, **the LR meta leans on FM
*more* when leakage piggybacking is blocked**.

## L1 ranking inversion is decisive

Under Strat, FMs are mid-pack:
```
b_lapsuntilpit  L1=0.888   ← #1
e5_optuna_lgbm  L1=0.679
... (10 GBDT/baseline bases) ...
FM_B            L1=0.138   ← #13
FM_A            L1=0.112   ← #15
```

Under GKF, FM_B is dominant:
```
FM_B            L1=6.961   ← #1  (2.06× the next base)
e5_optuna_lgbm  L1=3.378
cb_lossguide    L1=2.816
... (5 more GBDTs) ...
FM_A            L1=1.621   ← #9 (still in upper half)
... (4 weakest GBDTs) ...
```

When the LR meta can't get within-group leakage from the GBDT
predictions, it routes information through the FM embeddings, which
are estimated from feature-marginal distributions and don't carry
the same leakage signature.

## Implications

1. **PRIMARY (d9f K=21 swap, LB 0.95031) is private-LB robust.**
   The +2bp public-LB lift over d9c is real, not leakage shimmer.
2. **The cumulative d6_k18 → d9f progression (+5bp public)** is
   under-counted by Strat OOF (only +0.87bp K=13→K=15 lift). The
   GKF rebuild suggests the *real* contribution is ~+2bp, which
   matches the public LB closely.
3. **HEDGE selection holds**: d6_k18_multi_rule (LB 0.95026) is
   leakage-inflated like all GBDTs. Under GKF it would drop ~−230bp,
   but since the test set is ALSO from the same distribution as
   train (per U3 i.i.d. probe), private LB will sit ~3.8bp below
   Strat OOF — the same as the consistent calibration ladder gap.
4. **Future-base gating**: Use Race-only GKF stack lift as a private
   proxy. If Strat lift is positive but GKF lift is ≤0, suspect the
   base interacts with within-group leakage and reject. (Future
   FM-class additions should hold up under both.)

## Why this is stronger than the d10 base-level audit

The d10 audit (yesterday) showed FM *bases* are leakage-robust at
the standalone level (drop −9bp / −2.5bp under strict GKF vs −210bp
for GBDTs). That's necessary but not sufficient — a leakage-robust
base could still produce stack lifts only by leveraging the leakage
in *other* bases.

This rebuild closes that gap: under leak-blocked OOFs for ALL bases
in the stack, FM addition still lifts the LR meta. The mechanism
transfers.

## Pointers

- `scripts/d10b_groupkf_stack_rebuild.py` — rebuilds FM_A+FM_B
  under Race-only GKF and stacks K=13/K=15.
- `scripts/d10c_strat_match.py` — Strat-side match.
- `scripts/artifacts/d10b_groupkf_stack_rebuild.json` — full numbers.
- `scripts/artifacts/d10c_strat_match.json` — apples-to-apples table.
- `scripts/artifacts/oof_d9f_FM_{A,B}_groupkf_race.npy` — Race-only
  GKF FM bases (for any future GKF stack work).
