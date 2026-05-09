# 2026-05-09 — P10: Path-B with stint_start_imputed cohort (NULL)

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-pathb-cohort + meta-arch-variant`

## TL;DR

Tested two NEW cohort axes for Path-B partial-pooling on K=4 bases,
using the DGP-recovered orig-stint identifier `stint_start_imputed`:

| Cohort | n cells | τ=5k OOF | τ=20k OOF | τ=100k OOF |
|---|---:|---:|---:|---:|
| K=4 plain LR-meta (ref) | — | — | 0.95399 | — |
| stint_start_imputed_bin | 8 | 0.95401 | 0.95402 | 0.95402 |
| Compound × ss_bin | 36 | 0.95399 | 0.95402 | 0.95402 |
| **Compound × Stint (PRIMARY)** | 30 | — | — | **0.95403** |

All variants land within ±0.01 bp of the Compound × Stint reference.
The DGP-recovered cohort (theoretically more aligned with orig
structure) gives the **same** Path-B amplification as the fabricated
synth Stint. NULL.

## Confirmation of d18 friction

This adds another data point to:

  `path-b-amp-only-fires-on-meta-arch-not-base-add` (Day-15 origin)
  +
  `cohort-axis-variation-isnt-the-amp-axis` (d18 K1/K2/K3 corollary)

Even when the cohort axis is **the recovered TRUE orig stint
identifier** (P1 finding: synth Stint label is fabricated; ss_imputed
is the real partition), Path-B amp is unchanged. The shrinkage prior
operates on the LR weights regardless of how the data is segmented.

## Implication for the rank-lock picture

The Path-B amp at K=4 (~+0.04 bp over global LR-meta) is set by the
shrinkage operation itself, NOT by which cohort variable you partition
on. Any reasonable partition of the data produces ~the same Path-B
lift.

This further constrains the search for additional lift:

  - **NOT in cohort variable**: 4 cohorts tested (Compound×Stint,
    Compound×mode_TyreLife, Compound×mode_LapTime_Delta,
    mode_TyreLife×Stint, ss_bin, Compound×ss_bin), all within ±0.5 bp.
  - **Likely in the shrinkage prior shape** (Student-t vs Gaussian,
    Yao/Vehtari covariance) — untested with K=4.
  - **Or a non-LR meta** — closed (LightGBM, RF, kernel SVM, NCA-kNN
    all NULL).
  - **Or external data** — closed per PI direction.

## Pointers

- `scripts/dgp_v2/p10_pathb_stint_start.py`
- `scripts/artifacts/oof_p10_pathb_*_strat.npy`

## Friction tag

`pathb-cohort-recovered-orig-stint-equal-to-fabricated-stint` —
DGP-aware cohort axis (stint_start_imputed_bin) gives ±0 bp Path-B
amp vs synth Stint. Cohort axis truly isn't the amp lever.
