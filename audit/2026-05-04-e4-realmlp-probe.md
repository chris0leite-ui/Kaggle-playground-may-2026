# E4 — RealMLP-TD CPU single-fold probe (2026-05-04)

Per relaxed 1h cap: single-fold wall not 5-fold projection.

## Result

- Fold 0 (StratKFold, full data, CPU): AUC = **0.94722**
- Δ vs baseline (0.94075): +64.7bp
- Fit wall: 2369s (39.5 min) — within 1h cap.
- Total wall: 2371s.

## Verdict

PROMISING — AUC > 0.94. Worth full 5-fold + adding to M5 pool.
