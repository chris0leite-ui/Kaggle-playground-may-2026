
## TL;DR

- 19 bases analyzed; FM family mean ΔAUC 9bp vs GBDT 233bp
- Calibration analysis on M5q shows per-cohort isotonic delta in `F_summary.md`
- Single-feature LR table reveals which raw features carry leakage-eaten signal
# Phase F — Leakage asymmetry


## Per-base ΔAUC table (Strat − GroupKF, in bp)

```
                   base           family  auc_strat  auc_groupkf  auc_within_race_strat  delta_strat_minus_groupkf_bp  delta_strat_minus_withinrace_bp
                 m2_xgb             GBDT     0.9451       0.9108                 0.9386                      342.3056                          64.2575
              a_horizon GBDT-formulation     0.9064       0.8747                 0.9061                      316.6319                           2.6187
        e1_catboost_sub         CatBoost     0.9460       0.9164                 0.9399                      295.8012                          60.6365
         b_lapsuntilpit GBDT-formulation     0.8984       0.8695                 0.8957                      289.1546                          26.7714
            cb_year-cat         CatBoost     0.9468       0.9199                 0.9408                      268.7114                          60.3248
       cb_slow-wide-bag         CatBoost     0.9479       0.9232                 0.9421                      246.8665                          58.5018
         d3a_te_unified        TargetEnc     0.9369       0.9128                 0.9288                      240.7546                          81.0795
           cb_lossguide         CatBoost     0.9470       0.9238                 0.9410                      231.9911                          59.5021
         e5_optuna_lgbm             GBDT     0.9474       0.9259                 0.9413                      215.0746                          60.7641
        f2_hgbc_shallow             GBDT     0.9486       0.9271                 0.9427                      215.0178                          58.8168
           f1_hgbc_deep             GBDT     0.9487       0.9274                 0.9428                      213.1272                          59.0787
              d3b_seqfe       SeqFE-GBDT     0.9425       0.9214                 0.9356                      211.7919                          69.0756
                e3_hgbc             GBDT     0.9488       0.9279                 0.9429                      209.0701                          58.9972
                 d2a_te        TargetEnc     0.9367       0.9163                 0.9288                      204.2598                          79.5354
    baseline_two_anchor             GBDT     0.9408       0.9206                 0.9335                      201.6524                          72.6009
      d6_rule_year_race        RuleResid     0.9459       0.9416                 0.9397                       43.0347                          61.7821
             d9b_R14_L4         SparseLR     0.9137       0.9095                 0.9018                       41.5617                         119.2064
d6_rule_driver_compound        RuleResid     0.9446       0.9406                 0.9382                       40.1500                          64.1633
                 d9c_fm               FM     0.9207       0.9198                 0.9094                        9.0832                         113.0633
```


## Family aggregate ΔAUC

```
                  count    mean     min     max
family                                         
GBDT-formulation      2  302.89  289.15  316.63
CatBoost              4  260.84  231.99  295.80
GBDT                  6  232.71  201.65  342.31
TargetEnc             2  222.51  204.26  240.75
SeqFE-GBDT            1  211.79  211.79  211.79
RuleResid             2   41.59   40.15   43.03
SparseLR              1   41.56   41.56   41.56
FM                    1    9.08    9.08    9.08
```


## Per-cohort isotonic calibration headroom (in-sample upper bound)

| Cohort | Base AUC | Calibrated AUC | Δ bp (in-sample) |
|---|---:|---:|---:|
| Year | 0.95057 | 0.95089 | +3.20 |
| Stint | 0.95057 | 0.95097 | +4.06 |
| Compound | 0.95057 | 0.95090 | +3.28 |

_(In-sample upper bound; real OOF lift ~half. Useful for ranking which cohort split has most slack.)_


## Per-feature leakage AUC (single-feature LR)

```
               feature  auc_strat  auc_groupkf  delta_bp
         LapTime_Delta     0.5658       0.4736  922.1886
              Position     0.5163       0.4727  436.0641
                 Stint     0.6836       0.6741   95.4829
Cumulative_Degradation     0.6111       0.6044   67.2229
          RaceProgress     0.6644       0.6593   50.8782
              TyreLife     0.6989       0.6944   45.3588
             LapNumber     0.7024       0.6989   34.0250
                  Year     0.4053       0.4412 -359.8180
```

Features with high single-feature AUC and small Δbp are the leakage-robust signal — they generalize across Race. Features with Δbp > 50 carry Race-specific information and don't cleanly transfer to held-out Races.

