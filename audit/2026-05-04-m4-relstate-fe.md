# M4 — Relative-state FE + LGBM two-anchor (2026-05-04)

Method: concat train+test, sort by (Race, Driver, LapNumber), compute relative-state features (skipping ones already present in host data), restore original row order, run LGBM 5-fold under two CV anchors.

Features added: ['Recent_Degradation', 'Traffic_Pressure_Proxy']
Features already present (skipped): ['Position_Change', 'LapTime_Delta', 'RaceProgress', 'Cumulative_Degradation']
Final feature count: 16 = 14 baseline cols + 2 added

## Sort/restore-order diagnostic
- assert train_fe[id] == train[id]: PASS
- assert test_fe[id] == test[id]: PASS
- OOF .npy aligned with train.csv row order (verified by id-equality before save).

## Two-anchor results

| anchor | OOF AUC | fold_std | per-fold | Δ vs baseline |
|---|---:|---:|---|---:|
| Strat (seed=42) | **0.94244** | 0.00080 | ['0.9433', '0.9418', '0.9423', '0.9414', '0.9434'] | +16.9bp |
| GroupKFold(Race) | **0.92195** | 0.01289 | ['0.9173', '0.9094', '0.9082', '0.9400', '0.9338'] | +13.6bp |

## G1 verdict
- Strat anchor: **PASS** (Δ=+16.9bp; PASS≥−5, SOFT≥−10)
- GroupKFold anchor: **PASS** (Δ=+13.6bp)

## Wall times
- smoke: see scripts/m4_relstate_smoke.py output
- probe: see scripts/m4_relstate_probe.py output
- full Strat 5-fold: 99s
- full GroupKF 5-fold: 57s
- total: 159s

## Top 10 fold-0 Strat feature importances (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | Year | 507950 |
| 2 | Stint | 383563 |
| 3 | Driver | 380447 |
| 4 | TyreLife | 226160 |
| 5 | Race | 217266 |
| 6 | LapTime_Delta | 136459 |
| 7 | LapNumber | 85999 |
| 8 | RaceProgress | 74618 |
| 9 | Compound | 49461 |
| 10 | Position_Change | 38386 |
