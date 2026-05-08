# M5h — L1coef-pruned stack (drop M5f bottom-2 by LR-meta L1) (2026-05-04)

Pool (13): ['baseline', 'd2a_te', 'm2_xgb', 'e1_cb_sub', 'e3_hgbc', 'e5_optuna_lgbm', 'a_horizon', 'b_lapsuntilpit', 'f1_hgbc_deep', 'f2_hgbc_shallow', 'cb_year-cat', 'cb_lossguide', 'cb_slow-wide-bag']

Dropped from M5f: ['m3_catboost', 'm4_relstate'] (L1coef = 0.112, 0.141 — bottom 2 of 15)

## Two-anchor results vs M5f

| anchor | M5h | M5f | Δ vs M5f | Δ vs M5d (LB 0.94963) |
|---|---:|---:|---:|---:|
| Strat | **0.95043** | 0.95042 | +0.1bp | +2.0bp |
| GroupKF | **0.93087** | 0.93105 | -1.8bp | +9.3bp |

## Hypothesis

Smaller pool (13 vs 15) should reduce OOF→LB gap-widening (M5b 7-base gap −3.5bp vs M5d 12-base gap −6.0bp pattern). Strat OOF lift ++0.1bp is within fold noise; the gap-tightening is the real expected win.

Submission: submissions/submission_m5h_lr_meta_l1pruned.csv (held).
