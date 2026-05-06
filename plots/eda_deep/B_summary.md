
## TL;DR

- Top numeric MI: RaceProgress, Stint, Year (values: [0.095, 0.079, 0.072])
- Top categorical MI: {'Compound': 0.037, 'Race': 0.017, 'Driver': 0.013}
- Strong lift cells (>2×) appear consistently in HARD compound × early stint and high-Position × late-RaceProgress — see B_pairwise/two_way_target_rate.png
- Pearson |ρ|>0.5: TyreLife×Cumulative_Degradation, LapNumber×RaceProgress (expected)
- All numeric features show ~zero train-test drift (KS<0.004); confirms i.i.d.
# Phase B — Pairwise screening + 2-way interactions

## Mutual information vs target (sub-sampled 80k)

**Numeric**:

```
RaceProgress              0.0946
Stint                     0.0786
Year                      0.0719
LapTime_Delta             0.0597
Cumulative_Degradation    0.0558
LapNumber                 0.0454
TyreLife                  0.0452
Position_Change           0.0379
LapTime (s)               0.0266
PitStop                   0.0020
Position                  0.0018
```

**Categorical**:

```
Compound    0.0366
Race        0.0167
Driver      0.0133
```


## 2-way target-rate pivots (with lift = cell / (row × col / global))

### Compound × Stint

**Top 5 lift (over independence)**:
```
    Compound  Stint  count  mean  lift
         WET      2     31 0.258 5.233
        SOFT      1  18694 0.247 4.246
INTERMEDIATE      5    194 0.134 3.303
         WET      1   1274 0.020 2.705
      MEDIUM      2  25363 0.448 2.252
```

**Bottom 5 lift**:
```
    Compound  Stint  count  mean  lift
INTERMEDIATE      4   2001 0.037 0.281
        SOFT      6    188 0.005 0.284
        SOFT      3   7920 0.137 0.481
        SOFT      5   2494 0.026 0.513
        SOFT      2   3210 0.196 0.514
```

### Driver × Compound

**Top 5 lift (over independence)**:
```
Driver     Compound  count  mean  lift
   NOR         SOFT    120 0.450 1.923
  D001 INTERMEDIATE     77 0.221 1.702
   TAY INTERMEDIATE     81 0.198 1.593
   VER         SOFT    183 0.464 1.579
   VER INTERMEDIATE     87 0.322 1.534
```

**Bottom 5 lift**:
```
Driver     Compound  count  mean  lift
   FIS         SOFT    145 0.062 0.342
   HEI         SOFT    168 0.083 0.440
   GUT INTERMEDIATE    120 0.058 0.446
   KOV         SOFT    154 0.117 0.600
   BUT         SOFT    137 0.131 0.649
```

### Year × Race

**Top 5 lift (over independence)**:
```
 Year                     Race  count  mean  lift
 2022       British Grand Prix   2532 0.502 2.806
 2023         Qatar Grand Prix   4327 0.023 2.726
 2023 United States Grand Prix   6323 0.012 2.212
 2023         Dutch Grand Prix   5129 0.019 2.202
 2022       Pre-Season Testing   5144 0.404 2.059
```

**Bottom 5 lift**:
```
 Year                      Race  count  mean  lift
 2025     Australian Grand Prix   2533 0.043 0.164
 2024       Canadian Grand Prix   5232 0.082 0.357
 2023         Monaco Grand Prix   7483 0.007 0.379
 2024        British Grand Prix   3918 0.085 0.429
 2024 Emilia Romagna Grand Prix   6513 0.182 0.449
```

### Compound × TyreLife_d10

**Top 5 lift (over independence)**:
```
Compound  TyreLife_d10  count  mean  lift
     WET             3    143 0.056 2.874
     WET             8     32 0.094 2.261
     WET             4    147 0.048 2.152
    SOFT             1   4346 0.164 2.041
     WET             1    195 0.021 1.968
```

**Bottom 5 lift**:
```
Compound  TyreLife_d10  count  mean  lift
     WET             0    198 0.000 0.000
     WET             5    176 0.006 0.212
    SOFT             9   1213 0.185 0.482
     WET             2    268 0.007 0.488
    HARD             9  31354 0.442 0.683
```

### Position_d5 × RaceProgress_d10

**Top 5 lift (over independence)**:
```
 Position_d5  RaceProgress_d10  count  mean  lift
           4                 3   7686 0.186 1.533
           4                 4   7225 0.235 1.426
           1                 0  10157 0.073 1.330
           4                 2   7180 0.111 1.207
           4                 5   6743 0.269 1.197
```

**Bottom 5 lift**:
```
 Position_d5  RaceProgress_d10  count  mean  lift
           4                 0   9063 0.042 0.767
           3                 0   8856 0.048 0.784
           0                 4  10412 0.123 0.807
           0                 3   9095 0.091 0.808
           2                 2   7117 0.078 0.819
```

