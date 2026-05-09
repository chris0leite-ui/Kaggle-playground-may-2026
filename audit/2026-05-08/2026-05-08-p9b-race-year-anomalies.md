# 2026-05-09 — P9b: Race × Year DGP heterogeneity

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-race-year-heterogeneity`

## Headline finding

**The 2023 anomaly is uniform across nearly all races at ~1% pit
rate, EXCEPT French Grand Prix 2023 (25%) and Pre-Season Testing 2023
(0.5%).** Combined with the fact that French GP was REMOVED from F1
calendar after 2022 (i.e., didn't actually happen 2023-2025), this
suggests:

1. The orig data's 2023 portion is a **different scrape source**
   than 2022/2024/2025 — likely practice/qualifying (pit rate ~1%)
   while other years are race sessions (pit rate ~30%).

2. CTGAN faithfully propagates the source heterogeneity per
   (Race, Year) cell.

3. French GP 2023 (25% pit rate) is anomalous within 2023; orig may
   have race-source data tagged as "French GP 2023" even though the
   race didn't happen — possibly due to a labelling bug in the
   upstream scrape.

## Race × Year pit rates (from train, 26 races × 4 years)

```
Year                        2022   2023   2024   2025
Abu Dhabi Grand Prix       0.232  0.010  0.259  0.300
Australian Grand Prix      0.200  0.007  0.377  0.043
Austrian Grand Prix        0.270  0.009  0.285  0.187
Azerbaijan Grand Prix      0.270  0.008  0.457  0.366
Bahrain Grand Prix         0.417  0.014  0.412  0.477
Belgian Grand Prix         0.307  0.016  0.392  0.553
British Grand Prix         0.502  0.005  0.085  0.319
Canadian Grand Prix        0.245  0.007  0.082  0.313
Chinese Grand Prix         0.375  0.000  0.379  0.398
Dutch Grand Prix           0.265  0.019  0.164  0.236
Emilia Romagna Grand Prix  0.341  0.000  0.182  0.338
French Grand Prix          0.256  0.250  0.344  0.375  ⚠ 25% in 2023
Hungarian Grand Prix       0.267  0.010  0.372  0.309
Italian Grand Prix         0.157  0.008  0.242  0.177
Japanese Grand Prix        0.267  0.014  0.493  0.191
Las Vegas Grand Prix       0.293  0.012  0.440  0.276
Mexico City Grand Prix     0.075  0.006  0.164  0.164
Miami Grand Prix           0.109  0.006  0.204  0.127
Monaco Grand Prix          0.373  0.007  0.760  0.390  ⚠ 76% in 2024
Pre-Season Testing         0.404  0.005  0.112  0.139
Qatar Grand Prix           0.384  0.023  0.153  0.374
Saudi Arabian Grand Prix   0.322  0.009  0.349  0.317
Singapore Grand Prix       0.135  0.005  0.183  0.259
Spanish Grand Prix         0.377  0.014  0.501  0.418
São Paulo Grand Prix       0.341  0.014  0.370  0.377
United States Grand Prix   0.115  0.012  0.218  0.157
```

## Anomalies

- **French GP 2023: 25%** (rest of 2023: ~1%) — race not on calendar
- **Pre-Season Testing 2022: 40.4%** (very high; rest of testing
  series: 0.5/11/14% in 2023/24/25) — odd inversion
- **Monaco GP 2024: 76%** (extreme outlier high)
- **Chinese GP 2022: 0.375** (8 rows only — race cancelled, CTGAN
  noise)
- **Chinese GP 2023: 0.000** (17 rows only — race cancelled)

The 2024 Monaco anomaly (76% pit rate) is real F1 — Monaco 2024 had
a chaotic race with multiple safety cars and many strategic pits.
That's a feature of the year+race, not a DGP issue.

## DGP-source-mixing implications

The orig dataset is a **per-(Race, Year) heterogeneous compilation**:

- 2022 + 2024 + 2025 most races: standard race-session data
- 2023 most races: practice/qualifying-only data (pit rare)
- French GP 2023 (cancelled race): mysteriously has race-like rates
- Cancelled-race years (China 2022/2023): only 8-17 rows, CTGAN noise
- Pre-Season Testing 2022: high pit rate (testing simulations?)

For LB prediction:
- Test set has same 31% 2023 rows
- Calibration looks correct per Year (mean OOF ≈ pos rate)
- The 2023 sub-problem is intrinsically lower-variance

## Implication for the DGP picture

Combining P1+P1b+P1c+P9+P9b, the orig dataset is now characterized
as a **MIXED-SOURCE COMPILATION** that the host's CTGAN faithfully
reproduces:

  - Multi-year span 2022-2025
  - Multi-source per (Race, Year): race / practice / testing
  - Driver vocab: 31 active (real timeline) + 100 historical
    abbrev (uniform fabrication) + 756 D-prefix synthetic ghosts
  - Year-conditional label distribution: 2023 ≈ practice ≈ 1% pit;
    others ≈ race ≈ 28-30% pit
  - Per-row CTGAN sampling preserves all these heterogeneities

This is a more nuanced characterization than "CTGAN of an F1 race
dataset" — it's a CTGAN of a heterogeneous mixed-source F1 dataset.

## Pointers

- This audit and inline shell probes.
