
## TL;DR

- Quick-LGBM AUC 0.9419 on Strat fold; SHAP top-6: ['TyreLife', 'Year', 'Stint', 'LapTime_Delta', 'Position_Change', 'Race']
- PDP shows year-2023 has flat curves on all top features (model has learned to dampen prediction for Year=2023) — confirms generator-flat hypothesis
- Highest AVP-ΔAUC bases: cb_slow-wide-bag=+23.5bp, cb_year-cat=+1.9bp, e5_optuna_lgbm=+1.8bp, e3_hgbc=+1.0bp, d3b_seqfe=+0.7bp
- OOF PCA shows 2-3 distinct clusters where bases disagree most (see oof_pca_disagreement.png)
# Phase D — Model-driven diagnostics

- Quick LGBM fold-0 AUC = 0.94187 (calibration vs e3_hgbc 0.94870)

- SHAP top-6: ['TyreLife', 'Year', 'Stint', 'LapTime_Delta', 'Position_Change', 'Race']


- AVP analysis on K=20 bases: ['e3_hgbc', 'e5_optuna_lgbm', 'cb_lossguide', 'cb_year-cat', 'cb_slow-wide-bag', 'a_horizon', 'b_lapsuntilpit', 'd3a_te_unified', 'd3b_seqfe', 'd6_rule_residual', 'd6_rule_compound_stint', 'd6_rule_driver_compound', 'd6_rule_year_race', 'd9c_fm', 'd9f_FM_A', 'd9f_FM_B', 'd9h_FM_aug12', 'd9i_FM_A_aug', 'd9i_FM_B_aug', 'realmlp']

- Meta in-sample AUC (full-data fit): 0.94987


## AVP per-base ΔAUC (in-sample, drop-1 vs full meta)

```

  cb_slow-wide-bag                      slope=+0.001  ΔAUC=+23.49bp
  cb_year-cat                           slope=+0.037  ΔAUC=+1.87bp
  e5_optuna_lgbm                        slope=+0.102  ΔAUC=+1.77bp
  e3_hgbc                               slope=+0.105  ΔAUC=+0.97bp
  d3b_seqfe                             slope=-0.030  ΔAUC=+0.72bp
  realmlp                               slope=+0.063  ΔAUC=+0.70bp
  b_lapsuntilpit                        slope=+0.005  ΔAUC=+0.46bp
  d3a_te_unified                        slope=-0.038  ΔAUC=+0.25bp
  d6_rule_year_race                     slope=-0.003  ΔAUC=+0.13bp
  d6_rule_driver_compound               slope=-0.007  ΔAUC=+0.08bp
  cb_lossguide                          slope=+0.059  ΔAUC=+0.06bp
  a_horizon                             slope=+0.004  ΔAUC=+0.05bp
  d9f_FM_B                              slope=-0.008  ΔAUC=+0.03bp
  d9c_fm                                slope=-0.015  ΔAUC=+0.03bp
  d9i_FM_A_aug                          slope=-0.008  ΔAUC=+0.03bp
  d6_rule_compound_stint                slope=-0.025  ΔAUC=+0.01bp
  d6_rule_residual                      slope=-0.024  ΔAUC=+0.01bp
  d9f_FM_A                              slope=-0.004  ΔAUC=-0.02bp
  d9i_FM_B_aug                          slope=+0.005  ΔAUC=-0.02bp
  d9h_FM_aug12                          slope=+0.002  ΔAUC=-0.03bp
```


## OOF-disagreement clusters

```
           size  target_rate  mean_pred  base_std
cluster                                          
0        154455        0.008      0.045     0.077
1         47142        0.579      0.566     0.152
2         48179        0.852      0.788     0.139
3         42665        0.288      0.338     0.166
4         30498        0.004      0.063     0.130
5         38784        0.006      0.086     0.154
6         67863        0.076      0.147     0.152
7          9554        0.003      0.067     0.171
```

Clusters with high `base_std` and `target_rate` ≠ `mean_pred` are where new diversity helps; low-std clusters with mean_pred close to target_rate are saturated.

