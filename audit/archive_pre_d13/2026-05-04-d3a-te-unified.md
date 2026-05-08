# D3a — Unified TE base + M5i/M5j stack tests (2026-05-04)

HANDOVER Step 1: orthogonal-signal play — extend d2a TE pool with the
two Day-1-missed levers (Driver×Compound, Race×LapBin via
`qcut(RaceProgress, q=10)`).

## Standalone d3a (single-model probe)

| anchor | d3a OOF | std | d2a OOF | Δ vs d2a |
|---|---:|---:|---:|---:|
| Strat (LB-proxy) | **0.93692** | 0.00083 | 0.93670 | **+2.2bp** |
| GroupKF (Race) | 0.91284 | 0.01652 | 0.91628 | -34.4bp |

Strat lift +2.2bp = within fold-noise. GroupKF regression (-34bp) is
expected: Race×LapBin TE for unseen Races falls back to global mean →
unhelpful under Race-leaveout.

## Stack tests vs M5h baseline (Strat 0.95043, GroupKF 0.93087)

### M5i — full pool (14 = M5h + d3a)

| anchor | M5i | Δ M5h |
|---|---:|---:|
| Strat | 0.95043 | -0.0bp (exact tie) |
| GroupKF | 0.93096 | +0.9bp |

L1-coef ranking (Strat, full 14):

```
b_lapsuntilpit       0.774   ← top
e5_optuna_lgbm       0.667
cb_year-cat          0.388
baseline             0.368
f1_hgbc_deep         0.347
e3_hgbc              0.328
e1_cb_sub            0.311
cb_slow-wide-bag     0.290
cb_lossguide         0.288
f2_hgbc_shallow      0.242
d2a_te               0.196
m2_xgb               0.194
a_horizon            0.156
d3a_te_unified       0.079   ← DEAD LAST
```

Median L1 = 0.301; below-median prune dropped 7/14 (incl. d2a, m2_xgb,
a_horizon, cb_lossguide, cb_slow-wide-bag, f2_hgbc_shallow, d3a_te_unified).
Pruned-pool Strat 0.95023 (-2.0bp) → median rule **too aggressive** for
this pool, where the L1 distribution has a top-tier vs bottom-tier
break around 0.20, not 0.30.

### M5j — swap (d2a out, d3a in; pool size 13)

| anchor | M5j | Δ M5h |
|---|---:|---:|
| Strat | 0.95044 | +0.1bp |
| GroupKF | 0.93092 | +0.5bp |

L1-coef ranking (Strat, swap):

```
cb_slow-wide-bag     1.507   ← top (no longer "tied" with d2a)
e5_optuna_lgbm       1.209
d3a_te_unified       1.065   ← 3rd, vs 0.079 when both d2a+d3a present
a_horizon            0.882
b_lapsuntilpit       0.758
cb_year-cat          0.510
baseline             0.476
e3_hgbc              0.449
cb_lossguide         0.408
m2_xgb               0.321
f1_hgbc_deep         0.300
e1_cb_sub            0.292
f2_hgbc_shallow      0.289
```

d3a's L1 jumped 13× when d2a was removed → **confirms d2a+d3a are
highly redundant**. LR meta splits weights between them when both
present (both <0.20 L1); when only one present, that one absorbs the
combined role.

## Conclusions

1. **PI swap hypothesis correct, but no measurable lift.** d3a can
   replace d2a cleanly (M5j Strat 0.95044 ≈ M5h 0.95043). Pool stays
   at 13. But +0.1bp Strat is noise — no expected LB improvement.

2. **The new keys (Driver×Compound, Race×LapBin) underdelivered** as
   orthogonal signal. The LGBM with native cat handling already
   extracts most of what TE adds. Standalone +2.2bp didn't survive
   meta-stacking once redundant features were available elsewhere.

3. **OOF→LB-transfer collapse confirmed at the mechanism level.** Step 1
   targeted ~+5bp orthogonal lift; delivered +0.1bp in stack. M5h Strat
   0.95043 is the OOF ceiling on this pool *and* the pool is now
   saturated for any LGBM-TE-shaped lever.

4. **Median L1 prune is too aggressive** for the M5h-style pool. The
   L1 distribution has a top-cluster (≥0.29) and a bottom-cluster
   (≤0.20). A *tier-break* prune (drop bases with L1 < 0.20) is
   strictly more principled. Suggested as M5h-prune-v2 if needed
   later.

## Next step (HANDOVER plan)

Step 1 underdelivered → move to **Step 2: Sequence-FE base**
(`laps_since_last_pitstop`, `cumulative_pitstops_this_race`,
`rolling_target_rate(window=5)` over (Race, Driver) groups). 97.4% of
test rows have an in-test (Race, Driver) successor → the sequence
structure is genuinely unexploited. Higher expected lift than TE.

If Step 2 also underdelivers → fall back to **submit M5j as slot 7**
(d3a swap) to record the LB calibration of the swap, and move to
Step 3 (RealMLP on Kaggle GPU).

## Artifacts

- `scripts/d3a_te_unified.py`
- `scripts/m5i_d3a_l1prune.py`
- `scripts/m5j_d3a_swap.py`
- `scripts/artifacts/d3a_te_unified_strat_results.json`
- `scripts/artifacts/d3a_te_unified_groupkf_results.json`
- `scripts/artifacts/m5i_lr_meta_results.json`
- `scripts/artifacts/m5j_swap_results.json`
- `submissions/submission_d3a_te_unified.csv` (held)
- `submissions/submission_m5i_lr_meta.csv` (held)
- `submissions/submission_m5j_swap.csv` (held)
