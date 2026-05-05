# M5h calibration diagnostics — 2026-05-04

Aggregate Strat OOF AUC: **0.95043**  Brier: 0.06955

## Reliability bins (decile)

 bin     n  pred_mean  obs_rate       gap
   0 43914   0.000367  0.000387  0.000021
   1 43914   0.000902  0.000934  0.000032
   2 43914   0.002072  0.002368  0.000296
   3 43914   0.005244  0.005374  0.000130
   4 43914   0.012663  0.011477 -0.001186
   5 43914   0.032114  0.029262 -0.002852
   6 43914   0.107206  0.110329  0.003123
   7 43914   0.331076  0.332172  0.001096
   8 43914   0.626109  0.626998  0.000889
   9 43914   0.872198  0.870520 -0.001679


## Calibration variants vs M5h baseline

| variant | OOF AUC | Δ M5h (bp) | notes |
|---|---:|---:|---|
| baseline (uncalibrated) | 0.95043 | 0.0 | reference |
| global isotonic | 0.95053 | +1.0 | should be ~0 (AUC monotone-invariant) |
| per-Race rank-normalize | 0.92828 | -221.6 | rescale within Race |
| per-Race isotonic | 0.95161 | +11.8 | 26 isotonic fits; non-monotonic across Race |
| per-Year isotonic | 0.95075 | +3.2 | 4 fits; coarser |
| per-(Year,Race) isotonic | 0.95290 | +24.6 | finest grouping |

Best: **perYearRace_iso** at AUC 0.95290 (Δ M5h +24.6bp).

## Implications

Global isotonic is AUC no-op (confirms diagnostic).
Per-group calibrations CAN move AUC because they break global monotonicity.
If the best variant gives ≥+5bp OOF AUC, it is a slot-7 candidate (free LB lift).