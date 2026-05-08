# A — Horizon-shift reformulation (PitInNext3Laps) (2026-05-04)

Train HGBC on PitInNext3Laps[t] = OR(PN[t], PN[t+1], PN[t+2]); evaluate raw output as proxy for PitNextLap. Horizon prior 0.425 vs orig 0.199.

## Two-anchor results (AUC vs ORIGINAL target)

| anchor | OOF AUC orig | OOF AUC horizon | Δ vs baseline |
|---|---:|---:|---:|
| Strat | **0.90640** | 0.86813 | -343.5bp |
| GroupKF | **0.87474** | 0.81694 | -458.5bp |

Wall: 256s. Held for M5c stack refit.
