# U2 probe — lead(PitStop) single-feature strength (2026-05-04)

Train rows: 439,140; Test rows: 188,165
Test rows with computable lead_PitStop: **97.36%**

## Single-feature 5-fold StratifiedKFold OOF AUC

| feature(s) | OOF AUC | per-fold |
|---|---:|---|
| lead_PitStop (next-lap PitStop, with -1 sentinel for last lap) | **0.51233** | 0.5110, 0.5118, 0.5127, 0.5135, 0.5125 |
| PitStop (this-lap, baseline reference) | **0.52139** | 0.5192, 0.5189, 0.5211, 0.5236, 0.5215 |
| lag_PitStop (prev-lap, with -1 sentinel for first lap) | **0.52877** | 0.5284, 0.5297, 0.5295, 0.5292, 0.5289 |
| TyreLife (top numeric by F-stat) | **0.69874** | 0.6995, 0.7006, 0.6994, 0.6968, 0.6987 |
| lead_PitStop + PitStop + TyreLife (3-feature heuristic) | **0.71354** | 0.7133, 0.7137, 0.7144, 0.7133, 0.7139 |

## Verdict
**Normal feature.** Single-feature OOF = 0.51233. One useful signal among many; baseline-without-it is still a meaningful calibration anchor.
