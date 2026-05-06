
## TL;DR

- 2023 pit rate is uniformly ~1% across ALL races (std≈0); 2022/2024/2025 have wide race-to-race spread → 2023 generator ignores race-specific strategy
- Stint × Compound × prev_Compound spread is dominated by SOFT→HARD/MEDIUM and MEDIUM→HARD Stint-2 transitions; rate up to ~70%
- Compound × TyreLife × RaceProgress: late-RP × mid-TL on HARD has highest lift cells (4-6×) — this is where the next FM feature should encode
- Cumulative_Degradation is highly correlated (ρ>0.95) with TyreLife per-compound → marginal info likely small; Compound-residualized TL is the same signal
- Top-15 drivers' Position effect is real but small relative to Compound effect
# Phase C — Three-way heatmaps

## 1. Stint × Compound × prev_Compound

**Top 8 high-pit cells (n≥200)**:

```
prev_Compound     Compound  Stint  count  mean
         HARD       MEDIUM      2   1980 0.506
       MEDIUM       MEDIUM      2  17756 0.483
         HARD         HARD      2  61703 0.444
 INTERMEDIATE INTERMEDIATE      3   1638 0.422
 INTERMEDIATE         HARD      2    202 0.356
         SOFT         HARD      3   1431 0.349
       MEDIUM INTERMEDIATE      2    360 0.347
 INTERMEDIATE INTERMEDIATE      2   1674 0.345
```

## 2. Compound × TyreLife-decile × RaceProgress-decile

**Top 8 high-lift cells (n≥200, lift = rate / global rate)**:

```
    Compound  TL_d10  RP_d10  count  mean  lift_vs_global
        SOFT       5       2    634 0.666           3.345
        SOFT       3       1    499 0.659           3.313
INTERMEDIATE       9       9    330 0.630           3.168
        SOFT       2       0    589 0.626           3.148
        HARD       9       8   9836 0.600           3.013
        HARD       7       6   4308 0.558           2.803
        HARD       9       7   5644 0.549           2.761
        HARD       8       7   5380 0.544           2.734
```

## 3. Year × Race × Stint

**Per-Year aggregate range across Races**:

```
         min     max    mean     std
Year                                
2022  0.0752  0.5016  0.2702  0.1115
2023  0.0046  0.0231  0.0103  0.0048
2024  0.0816  0.7602  0.3054  0.1601
2025  0.0426  0.5535  0.2882  0.1206
```

**Interpretation**: 2023 has tight intra-Year std (no race deviates much from the global 0.96% pit rate); 2022/2024/2025 have wide spreads (race-specific strategy patterns). The 2023 generator is a flat-rate model, NOT a per-race model.

## 4. Driver × Position × Compound

**Position-bin × Compound mean (top-15 drivers)**:

```
Compound   HARD  INTERMEDIATE  MEDIUM   SOFT   WET
Pos_bin                                           
1-5       0.397         0.183   0.156  0.198  0.05
6-10      0.444         0.140   0.132  0.153  0.00
11-15     0.409         0.098   0.143  0.167  0.00
16+       0.333         0.041   0.135  0.202  0.00
```

## 5. TyreLife × Cumulative_Degradation × Compound

**Pearson(TyreLife, Cum_Deg) per Compound** — tests degradation's marginal info:

```
Compound
HARD           -0.077
INTERMEDIATE   -0.256
MEDIUM         -0.081
SOFT           -0.105
WET            -0.202
```

**Interpretation**: if ρ→1 per Compound, Cum_Deg is a deterministic function of TyreLife and adds no info to a Compound-conditioned base.

