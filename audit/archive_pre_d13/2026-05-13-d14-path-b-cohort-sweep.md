# 2026-05-13 — d14 Path B cohort sweep results

Built and gated 9 Path B variants extending d13/d13e:
  Year (4), Year × Stint (24), Race (26) × τ ∈ {5k, 20k, 100k}
~22 min wall, 0 tokens.

## Result table

| Cohort | n_seg | τ | OOF | Δ d9f | Δ PRIMARY | ρ d9f | ρ PRIMARY |
|---|---:|---:|---:|---:|---:|---:|---:|
| Year | 4 | 5000 | 0.95068 | −0.45 | −1.45 | 0.998 | 0.996 |
| Year | 4 | 20000 | 0.95071 | −0.23 | −1.24 | 0.999 | 0.996 |
| Year | 4 | 100000 | 0.95074 | +0.09 | −0.92 | 0.999 | 0.996 |
| Year × Stint | 24 | 5000 | 0.95073 | −0.02 | −1.03 | 0.995 | 0.994 |
| **Year × Stint** | **24** | **20000** | **0.95080** | **+0.71** | **−0.30** | 0.997 | 0.995 |
| Year × Stint | 24 | 100000 | 0.95079 | +0.65 | −0.35 | 0.999 | 0.996 |
| Race | 26 | 5000 | 0.95050 | −2.27 | −3.28 | 0.995 | 0.992 |
| Race | 26 | 20000 | 0.95069 | −0.43 | −1.43 | 0.998 | 0.994 |
| Race | 26 | 100000 | 0.95074 | +0.11 | −0.90 | 0.999 | 0.996 |
| **PRIMARY** Compound × Stint τ=20k | 24 | 20000 | **0.95083** | +1.00 | 0 | 0.996 | 1.000 |

## Read

**No new variant beats current PRIMARY on OOF.**  Best contender —
Year × Stint τ=20k — sits at −0.30 bp.  By the 18× Compound × Stint
amplification ratio that would project LB ≈ 0.95044 (regress -5 bp
vs PRIMARY 0.95049).

**Compound axis dominates Year axis** for Path B specialization:
- Year=2023 is a flat-rate generator (Phase A: pit rate 0.0096
  uniform across Compound, Stint, Race); per-Year LR specialization
  just memorizes the global per-Year mean → little segment-conditional
  signal beyond "if Year==2023 → clamp low".
- Compound segments have rich within-strategy structure
  (Stint × Compound × prev_Compound spreads pit rate 18.9 → 75.4%
  per Phase C); per-Compound LR has more room to specialize.
- The very EDA finding (2023 anomaly) that motivated Year-cohort
  variants is the reason they underperform.

**Race-cohort scaling**:
- τ=5k overfits at 17k rows / segment (-2.27 bp);
- τ=100k pulls so hard toward global that lift collapses (+0.11 bp);
- τ=20k middle ground (-0.43 bp).
- Race cohort doesn't carry enough signal to cohort-specialize.

## Held submission files

```
submissions/submission_d14_path_b_year_tau{5000,20000,100000}.csv
submissions/submission_d14_path_b_year_stint_tau{5000,20000,100000}.csv
submissions/submission_d14_path_b_race_tau{5000,20000,100000}.csv
```

All 9 are HELD (no PI sign-off requested; no token spent).  Best
single candidate Year × Stint τ=20k is most-eligible if a Day-14
calibration probe is desired (tests whether Year-axis amplifies
similarly to Compound-axis).  Predicted LB regression of -3 to -5 bp.

## Cohort lever conclusion

Path B FAMILY is alive (3/3 axes lifted: Compound +2bp, Stint +7bp,
Compound × Stint +18bp).  Specifically the **Compound × Stint
τ=20k PRIMARY at LB 0.95049** captured the bulk of the cohort
specialization signal in this K=21 pool.  Further cohort axes
(Year, Race, Year × Stint) do not add orthogonal signal at
the OOF level.

## Next moves not in this audit's scope

- Multi-cohort meta-blend (LR over Compound × Stint OOF +
  Year × Stint OOF as inputs).  Untested.
- Multi-level hierarchy (Stint within Compound within Year).
- TabPFN-2.5 GPU push (Day-12 prep, only +10 bp tail shot remaining).

## Pointers

- `scripts/d14_path_b_cohort_sweep.py` — sweep script (extends
  `scripts/d13_path_b_hier_meta.py` with Year, Year × Stint, Race
  cohorts via `make_segments_extra`).
- `scripts/artifacts/d14_path_b_cohort_sweep_results.json` — full results.
- `scripts/artifacts/oof_d14_path_b_*_strat.npy` (9 files) — OOFs.
- `scripts/artifacts/test_d14_path_b_*_strat.npy` (9 files) — test preds.
