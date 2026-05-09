# 2026-05-08 — P7: driver-atypicality + tuple-count base (K=4+1 NULL +0.17 bp)

`branch: claude/find-dgp-research-ClsQE`
`tag: dgp-driver-vocab + tuple-count`

## TL;DR

P7 is a single LightGBM 5-fold base trained on standard 14 features
+ the P1b driver-atypicality features + the P1c tuple-count features:

  - `driver_year_count` — rows for this driver in this year (synth)
  - `driver_year_cv` — per-driver year-CV (low for active steady,
    high for rookies/retirees)
  - `driver_total_count` — total rows for this driver
  - `is_active_in_year` — heuristic (`driver_year_count > 100`)
  - `is_d_prefix` — synthetic D### code indicator
  - `stint_start_imputed` — LN − TL + 1
  - `tcnt_lt_tl` — count of rows with this (LapTime, TyreLife) pair
  - `tcnt_R_Y_C_SS` — count of rows with this orig-stint cell key
  - `log_tcnt_R_Y_C_SS` — log of above
  - `tcnt_C_S_L` — count of rows with this (Compound, Stint, LapNumber) tuple

| Metric | Value |
|---|---:|
| Standalone OOF | 0.94291 |
| Δ vs PRIMARY (0.95403) | −111 bp |
| ρ vs PRIMARY (test) | 0.9580 |
| K=4+1 LR-meta lift | **+0.17 bp** |
| G3 flip ratio (top-1%) | 0.080 (1452 +→−, 116 −→+) |
| Fold std | 0.00073 |

Verdict: NULL (below +0.5 bp gate). Highest K=4+1 lift among my P2-P7
DGP-aware probes (P2 +0.09, P5 +0.14, P7 +0.17). Still rank-locked.

## Family falsification (Rule 21)

Three structurally distinct DGP-aware FE bases all NULL at K=4+1:

| Variant | Feature focus | OOF | ρ vs PRIMARY | K=4+1 Δ |
|---|---|---:|---:|---:|
| P2 | stint_start_imputed + cell stats + 3 TEs | 0.93971 | 0.953 | +0.09 |
| P5 | recovery alone (no std14) | 0.92624 | 0.903 | +0.14 |
| P7 | driver-atypicality + tuple counts | 0.94291 | 0.958 | +0.17 |

Range: +0.09 to +0.17 bp. **Family `dgp_aware_fe_on_K4_primary` is
falsified at the +0.5 bp gate.** All three variants confirm:

  - ρ vs PRIMARY in the 0.90-0.96 band.
  - K=4+1 LR-meta lift in the 0.09-0.17 bp band.
  - The K=4 logit-direction subspace is robust to even ρ=0.90
    candidates.

## What this teaches us about the DGP

The DGP-recovery FE is REAL (P5 standalone OOF 0.926 — ~30 bp above
the random-feature floor of 0.5 + the "Compound+TyreLife only" floor
of ~0.85). But the RECOVERY signal overlaps almost entirely with the
K=4's existing tyre-degradation × compound axis.

**Why? The K=4 PRIMARY's bases (yekenot-RealMLP, CatBoost-yekenot,
HGBC-deep, LightGBM-on-orig) ALL extract the same tyre-progression
signal via raw features.** Our DGP-aware FE just re-expresses this
signal with extra structure; LR-meta absorbs it.

**A truly orthogonal signal would require either:**
  - Information not in the synth row at all (e.g. weather, safety
    car — outside the 14 features). External data (D-axis) is closed
    per PI direction.
  - A different task framing (LambdaRank, hazard model). Already
    closed at K=10+1 NULL per d16-d18 per-stint LambdaRank tests.
  - A meta-architecture variant that breaks the LR-meta logit-
    direction subspace. Closed: LightGBM-meta, RF-meta, NCA-kNN,
    kernel-SVM, all NULL on K=4 / K=10 / K=27.

## Implication

The "find DGP" mission has fully characterized the synthesizer
(P1, P1b, P1c) and confirmed that DGP-aware FE on top of K=4 PRIMARY
saturates at ~+0.2 bp K=4+1 LR-meta. The DGP knowledge is durable
research output; the LB ceiling at this PRIMARY composition is
empirically capped.

## Pointers

- `scripts/dgp_v2/p7_driver_atypicality.py`
- `scripts/dgp_v2/gate_p7_k4plus1.py`
- `scripts/artifacts/p7_driver_atypicality_results.json`
- `oof_p7_driver_atypicality_strat.npy`, `test_*`

## Friction tag

`dgp-aware-fe-rank-lock-saturates-at-0.2bp` — three independent
DGP-aware FE configurations cap K=4+1 LR-meta lift at +0.17 bp. Even
ρ=0.90 candidates absorb. The 1.33-rank K=4 logit subspace ceiling
is robust to FE substrate.
