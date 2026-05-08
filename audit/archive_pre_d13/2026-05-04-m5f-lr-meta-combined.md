# M5f — LR meta combining M5d (HGBC) + M5e (CB) additions (2026-05-04)

Pool (15): ['baseline', 'd2a_te', 'm2_xgb', 'm3_catboost', 'm4_relstate', 'e1_cb_sub', 'e3_hgbc', 'e5_optuna_lgbm', 'a_horizon', 'b_lapsuntilpit', 'f1_hgbc_deep', 'f2_hgbc_shallow', 'cb_year-cat', 'cb_lossguide', 'cb_slow-wide-bag']

## Two-anchor results vs M5c / M5d / M5e

| anchor | M5f | M5c | M5d (main) | M5e (mine) | Δ vs M5d | Δ vs M5e |
|---|---:|---:|---:|---:|---:|---:|
| Strat | **0.95042** | 0.95000 | 0.95023 | 0.95027 | +1.9bp | +1.5bp |
| GroupKF | **0.93105** | 0.92963 | 0.92994 | 0.93084 | +11.1bp | +2.1bp |

M5d LB anchor (main): 0.94963 (Strat OOF→LB gap −6.0bp).
M5e: held, projected LB ~0.94992 (using M5b's −3.5bp gap).

Submission: submissions/submission_m5f_lr_meta_combined.csv (held).
