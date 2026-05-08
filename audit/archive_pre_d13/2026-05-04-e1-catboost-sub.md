# E1 — Row-subsample CatBoost (subsample=0.8) two-anchor (2026-05-04)

Tests whether row-subsampling bounds M3's Race-overfit. M3 baseline:
Strat 0.94612, GroupKF 0.91645 (-41.4bp Race-overfit).

## Two-anchor results

| anchor | OOF AUC | fold_std | per-fold | Δ baseline | Δ vs M3 |
|---|---:|---:|---|---:|---:|
| Strat | **0.94596** | 0.00071 | ['0.9469', '0.9451', '0.9465', '0.9452', '0.9461'] | +52.1bp | -1.6bp |
| GroupKF | **0.91638** | 0.01245 | ['0.9166', '0.9050', '0.9027', '0.9354', '0.9264'] | -42.1bp | -0.7bp |

## Wall: 535s. Best-iter mean: Strat 799, GroupKF 308.
