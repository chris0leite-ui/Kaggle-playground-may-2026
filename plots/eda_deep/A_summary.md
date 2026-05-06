
## TL;DR

- 0/11 numeric features show significant train↔test drift (KS p<1e-3); top drifters: []
- Year 2023 pit rate 0.0096 vs other-years 0.2841 — 29× lower; KS divergence on every feature confirms generator shift
- Stint 1 has 6.0% pit rate; Stint 2 jumps to 39.1% (universal blind spot per P4)
- HARD compound has highest pit rate (32.8%) — pit-out → fresh-tyre lap counts as PitNextLap=1
# Phase A — Univariate fact sheet

- train: (439140, 16)  test: (188165, 15)
- target prior: 0.1990
- categoricals: Driver=887, Compound=5, Race=26

## KS-test train-vs-test (numeric)

| Feature | KS-stat | p-value | significant? |
|---|---:|---:|:---:|
| LapNumber | 0.0032 | 1.25e-01 | no |
| Stint | 0.0027 | 2.84e-01 | no |
| TyreLife | 0.0029 | 2.17e-01 | no |
| Position | 0.0024 | 4.09e-01 | no |
| LapTime (s) | 0.0015 | 9.26e-01 | no |
| LapTime_Delta | 0.0029 | 2.18e-01 | no |
| Cumulative_Degradation | 0.0028 | 2.65e-01 | no |
| RaceProgress | 0.0031 | 1.56e-01 | no |
| Position_Change | 0.0011 | 9.95e-01 | no |
| PitStop | 0.0001 | 1.00e+00 | no |
| Year | 0.0021 | 6.27e-01 | no |

## Class prior by Year × Compound

```
                   count    mean
Year Compound                   
2022 HARD          22025  0.4646
     INTERMEDIATE   4193  0.1097
     MEDIUM        45546  0.1798
     SOFT           9926  0.3228
     WET            1299  0.0262
2023 HARD          60996  0.0080
     INTERMEDIATE   1383  0.0202
     MEDIUM        58264  0.0095
     SOFT          15457  0.0153
     WET              47  0.0000
2024 HARD          53463  0.5385
     INTERMEDIATE   8440  0.2148
     MEDIUM        59548  0.0882
     SOFT           5652  0.2976
     WET               7  0.0000
2025 HARD          34034  0.4801
     INTERMEDIATE   3366  0.1028
     MEDIUM        47783  0.1540
     SOFT           7709  0.3078
     WET               2  0.0000
```

## Year-2023 anomaly: KS divergence vs other years

Per-feature KS(2023 distribution vs combined 2022/2024/2025).

| Feature | KS-stat | p-value | mean(2023) | mean(other) |
|---|---:|---:|---:|---:|
| LapNumber | 0.0808 | 0.00e+00 | 25.000 | 22.255 |
| Stint | 0.1014 | 0.00e+00 | 1.951 | 1.716 |
| TyreLife | 0.0516 | 4.14e-218 | 14.869 | 13.839 |
| Position | 0.0306 | 8.57e-77 | 9.325 | 9.767 |
| LapTime (s) | 0.0786 | 0.00e+00 | 91.370 | 90.759 |
| LapTime_Delta | 0.5520 | 0.00e+00 | -0.492 | -5.243 |
| Cumulative_Degradation | 0.4678 | 0.00e+00 | -14.598 | -30.720 |
| RaceProgress | 0.1787 | 0.00e+00 | 0.425 | 0.298 |
| Position_Change | 0.3809 | 0.00e+00 | 0.092 | 0.106 |
| PitStop | 0.1793 | 0.00e+00 | 0.012 | 0.192 |

## Driver tail

- unique drivers: 887
- median rows per driver: 174
- drivers with ≤50 rows: 354
- pit rate among low-count (≤50): 0.0075
- pit rate among top-100 drivers: 0.2236
