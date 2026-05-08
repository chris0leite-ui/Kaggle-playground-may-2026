# M5g — corr-pruned stack ρ≥0.97 (2026-05-04)

Pool (5): ['d2a_te', 'e3_hgbc', 'a_horizon', 'b_lapsuntilpit', 'cb_slow-wide-bag']

Dropped (10):
  - e1_cb_sub (kept m3_catboost, ρ=+0.9974)
  - m3_catboost (kept cb_year-cat, ρ=+0.9949)
  - baseline (kept m4_relstate, ρ=+0.9937)
  - f2_hgbc_shallow (kept e3_hgbc, ρ=+0.9926)
  - f1_hgbc_deep (kept e3_hgbc, ρ=+0.9917)
  - m2_xgb (kept e5_optuna_lgbm, ρ=+0.9887)
  - m4_relstate (kept e5_optuna_lgbm, ρ=+0.9879)
  - cb_year-cat (kept cb_lossguide, ρ=+0.9852)
  - e5_optuna_lgbm (kept e3_hgbc, ρ=+0.9811)
  - cb_lossguide (kept e3_hgbc, ρ=+0.9804)

## Results

| anchor | M5g | M5f | Δ vs M5f | Δ vs M5d (LB 0.94963) |
|---|---:|---:|---:|---:|
| Strat | **0.94961** | 0.95042 | -8.1bp | -6.2bp |
| GroupKF | **0.92915** | 0.93105 | -19.0bp | -7.9bp |

Submission: submissions/submission_m5g_lr_meta_pruned.csv (held).
