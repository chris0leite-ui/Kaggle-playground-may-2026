# Pool disagreement diagnostic — 2026-05-04

Question: where is the M5h pool uncertain vs locked-in consensus?

## Per-row disagreement (std across 13 bases, e1_cb_sub test rows)

| percentile | std |
|---:|---:|
| p10 | 0.0499 |
| p25 | 0.0828 |
| p50 | 0.1217 |
| p75 | 0.1548 |
| p90 | 0.1847 |
| p99 | 0.2401 |

High-disagreement subset (top decile by std): 18817 rows (10.0%)

## Per-base diversity (Spearman ρ vs mean of other 12)

Lower ρ = more orthogonal contribution to the consensus rank.

| base | ρ vs others |
|---|---:|
| a_horizon | 0.72855 |
| b_lapsuntilpit | 0.73449 |
| cb_slow-wide-bag | 0.83964 |
| cb_lossguide | 0.88369 |
| d2a_te | 0.89714 |
| baseline | 0.90906 |
| cb_year-cat | 0.91247 |
| f2_hgbc_shallow | 0.91350 |
| e3_hgbc | 0.91435 |
| f1_hgbc_deep | 0.91537 |
| m2_xgb | 0.91616 |
| e1_cb_sub | 0.92045 |
| e5_optuna_lgbm | 0.92224 |

## Use for slot 9-10 selection

When evaluating RealMLP / EBM / H1 / LR-FE as new pool members:

1. Compute Spearman ρ vs mean of M5h pool. Lower ρ = more diversity.

2. Compute mean |new_pred − consensus| on HIGH-DISAGREEMENT rows.

   Higher = more rank-shift potential.

3. Combine: a new base with low ρ AND high disagreement on the

   uncertainty subset is the candidate most likely to break the

   pool's locked-in consensus rank → LB lift potential.
