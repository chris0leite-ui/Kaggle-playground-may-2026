# Day-14 DGP-residuals probe — masked-column self-prediction NULL (2026-05-06)

> PI thesis: synthetic data is NN-generated, so features are physical
> only "in a fuzzy way." Predict every column from the rest, use what
> emerges to improve target prediction.
>
> Test: train one LGBM regressor per "DGP-fingerprint" column, use
> OOF residuals + composite anomaly score as new features for the
> PitNextLap LGBM. First formal closure of the **masked-column
> self-prediction / denoising-autoencoder pretraining** family
> (SAINT / TabNet / VIME class).

## Setup

  - Reconstruction targets (4): `LapTime_Delta`, `Cumulative_Degradation`,
    `Position`, `LapNumber`. Excluded as targets: `RaceProgress`,
    `TyreLife`, `Stint`, `Year` (deterministic from stint arithmetic
    or categorical-ish; trivial residuals).
  - Per target: 5-fold StratifiedKFold LGBM regressor, predicts column
    from all remaining features (excludes target column AND
    `PitNextLap`/`id`).
  - Features built: `dgp_z_*` (4 z-scored residuals) +
    `dgp_anomaly_L1 = sum |z|`.
  - Final base: 5-fold LGBM classifier on standard 14 features + 5
    new DGP features → `oof_d14_dgp_residuals_strat.npy`.
  - Cost: 25 LGBM fits, 11.4 min wall.

## Results

| Gate | Δ | ρ vs PRIMARY | Verdict |
|---|---:|---:|---|
| **Standalone OOF** | **−88.26 bp** | 0.9599 | std AUC 0.94200 — weakest base in pool by margin (e3_hgbc 0.94876) |
| **K=2 min-meta** (PRIMARY + cand) | −0.025 bp | 0.9599 | NULL |
| **K=22 add** (K=21 + cand) | +0.172 bp | **0.9958** | noise-floor positive; pred LB **−1.3 bp** at ρ=0.996 (band: d_oof − 1.5) |
| G3 flip ratio (top-1%) | 0.311 | — | asymmetric (1037 down / 322 up) |
| Candidate L1 weight in K=22 meta | 0.291 (mid-pack: raw −0.050, rank −0.135, logit +0.105) |

**Verdict: FAIL.** Family `masked_column_self_prediction` /
`denoising_autoencoder_features` closed.

## The load-bearing diagnostic — DGP is conditionally near-independent

Reconstruction RMSE vs the marginal σ of each target:

| Target | RMSE (OOF mean) | Marginal σ (residual) | Variance explained |
|---|---:|---:|---:|
| LapTime_Delta | 41.05 | 41.06 | ≈ 0% |
| Cumulative_Degradation | 34.94 | 34.97 | ≈ 0% |
| Position | 3.491 | 3.491 | ≈ 0% |
| LapNumber | 1.559 | 1.559 | ≈ 0% |

**Across all 4 reconstruction targets, the regressor extracted
essentially zero variance reduction over the marginal.** The NN-DGP
generated the synthetic dataset with **conditionally near-independent
features within each row** — knowing the rest of the row tells you
almost nothing about any single column.

Two readings:

1. **The DGP synthesizer added per-feature noise that is near-i.i.d.
   within rows.** The host's "feature distributions are close to, but
   not exactly the same, as the original" turns out to mean each
   feature was sampled near-independently conditional on its marginal
   moments and a small amount of cross-feature structure that the
   GBDTs already exploit fully on the target.
2. **All cross-feature interaction signal has already been absorbed
   by FM_aug12.** The d14 Move D NULL (FM_aug16 with 4 new input
   types: −0.07 bp min-meta) is the same finding from the FM side.
   Neither GBDT nor FM extracts new variance from feature-on-feature
   modeling.

This jointly explains why:
  - FM-field-augmentation saturated at 12 fields (d14).
  - Single-base FE additions all NULL on Day-13/14 (G1, G2', G3, H1).
  - Path B cohort sweep on Year/Year×Stint/Race NULL (d14).
  - Move D (4 new FM input types) +20 bp standalone but −0.07 bp
    min-meta.

The remaining gap to top-5% (−29.6 bp) is **not a feature-engineering
problem**. It is leakage-population routing (d12 finding) plus model-
class diversification (TabPFN dead, DeepFM-lite untested).

## Why the K=22 add showed +0.172 bp despite K=2 −0.025 bp

The K=22 LR-meta has 21 strong calibrated bases. Adding a noisy
21st-rank base at ρ=0.996 with mid-pack |w|=0.29 gives a tiny ensemble
benefit (~0.17 bp) that vanishes when only PRIMARY is the anchor.
At ρ=0.996 the harness band predicts **LB Δ = +0.172 − 1.5 = −1.3 bp**
— below the +0.5 bp slot threshold. **DO NOT submit.**

## Pointers

  - Script: `scripts/d14_dgp_residuals.py` (BOTE-graded
    `single_base_fe_addition` family; ran in 11.4 min)
  - Gate JSONs:
    - `scripts/artifacts/d14_dgp_residuals_results.json`
    - `scripts/artifacts/probe_min_meta__d14_dgp_residuals.json`
  - OOF / test artifacts:
    `scripts/artifacts/{oof,test}_d14_dgp_residuals_strat.npy`
  - Friction tag: `synthetic-dgp-conditionally-near-independent`
    — record under audit/friction.md.

## Add to mechanism_families_explored

  - `masked_column_self_prediction` — d14 first probe, FAIL.

## Implication for remaining live moves (HANDOVER §"Remaining live moves")

The 4-of-4 NULL alt-axis pattern (Day-13/14) plus this 5th NULL
crystallize the same finding from a 5th angle: **per-row feature
conditioning gives no incremental signal beyond what GBDTs+FM already
extract**. The path forward is:

  1. **Meta-layer / pool-population innovations** (Path B variants on
     untried cohort axes — 2a Beta-Binomial shrinkage; 2b calibration-
     aware stacker; 2c GroupKF-meta as R5 HEDGE).
  2. **New model class** (1b EmbMLP, 1c DeepFM-lite — both untried).
  3. **External data** (4a Pirelli pit-window scrape — Tier-2).

Single-base FE additions (LGBM-class, FM-class, DGP-residual) are
DEAD across 5 independent probes and should be dead-listed.
