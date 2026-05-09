# 2026-05-09 — P11: per-cell calibration of K=4 PRIMARY (REGRESSES)

`branch: claude/find-dgp-research-ClsQE`
`tag: post-hoc-calibration + dgp-aware`

## TL;DR

Per-fold per-(cell) residual calibration of K=4 PRIMARY (resid = y −
p_primary; per-cell mean shrunk toward 0 with smooth=200; add to
predictions). All 5 cell axes tested REGRESS:

| Cell axis | OOF | Δ vs PRIMARY |
|---|---:|---:|
| Stint_Year | 0.95395 | **−0.76 bp** |
| Compound_Stint | 0.95393 | −0.96 bp |
| Year_Compound | 0.95393 | −0.96 bp |
| Year_ss_bin | 0.95395 | −0.78 bp |
| Compound_Year_ss_bin | 0.95385 | −1.74 bp |

Confirms HANDOVER's "Conformal isotonic recalibration of PRIMARY —
already globally calibrated" finding. Extends it to per-(Stint, Year),
per-(Compound, Stint), per-(Year, Compound), per-(Year, ss_bin),
per-(Compound, Year, ss_bin) cells. ALL fail.

## Why

The K=4 PRIMARY's residuals correlate weakly with Stint (ρ +0.144),
RaceProgress (ρ +0.119), TyreLife (ρ +0.091) — but these
correlations are TYPICAL of a well-fit binary classifier on a
nearly-saturated label structure. Adjusting predictions by per-cell
residual mean injects noise (~σ²/n_cell variance per cell) that
exceeds the systematic bias signal.

Even with smooth=200 (heavy shrinkage toward 0), the cell adjustments
add noise faster than they correct bias. Smaller smoothings would
regress further; larger smoothings → no adjustment → identity.

## Pointers

- `scripts/dgp_v2/p11_cell_calibration.py`

## Friction tag

`per-cell-residual-calibration-regresses-K4-primary` — even with
smoothing, per-cell adjustment of K=4 OOF regresses 0.76-1.74 bp
across 5 cell axes. PRIMARY is row-level optimally calibrated.
