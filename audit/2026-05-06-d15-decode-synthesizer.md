# 2026-05-06 — d15: decode the synthesizer (3 lenses)

`branch: claude/decode-synthetic-data-uoPIn`
`tag: synthesizer-decode`
`mechanism family: external_data_aggregate (lens 2 PASS) / single_base_fe_addition (1+3 NULL)`

## TL;DR

- **Source confirmed**: `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`
  is the verified original. 31/887 synth Drivers exactly match; all 26
  synth Races present; all 5 synth Compounds present; Year ∈ {2022..2025}
  matches. The host removed `Normalized_TyreLife` and added 856 ghost
  Driver codes (D001–D856 plus historical 3-letter codes like MAS/RAI/BAR).
- **Synthesizer marginal-leak**: 97.55% of synth `LapTime (s)` values
  are *literally drawn from* the original's empirical distribution
  (LapTime_Delta 95%, RaceProgress 99.95%, Cumulative_Degradation 87%).
  Joint structure broken (only 5.8% of `(LapTime, TyreLife, Compound)`
  triples survive).
- **Lens 1 (decoded NTL feature)**: NULL at min-meta (Δ −0.008 bp).
  NTL absorbed by `TyreLife + RaceProgress + Stint` already in pool.
- **Lens 2 (orig-trained transfer base)**: **+0.778 bp at min-meta**
  with ρ=0.565 vs PRIMARY (most-diverse single base since d9f FM_A at
  ρ=0.487). Stacks with `d12_lr_meta` to **+1.394 bp** combined.
- **Lens 3 (physics-residual base)**: NULL (Δ −0.036 bp). Physics
  already absorbed by GBDT pool.

## Provenance check (Lens 1 fingerprint)

```
synth train: (439140, 16) | synth test: (188165, 15) | original: (101371, 16)
synth_total / original = 6.188×

Driver:    synth=887 levels, orig=31 levels, overlap=31 (100% of orig)
Compound:  synth=5,    orig=5+nan,  overlap=5
Race:      synth=26,   orig=26+2 pre-season, overlap=26
Year:      synth ∈ {2022..2025} == orig
```

Within the 31 "real-driver" synth rows (~7% of synth), 80.7% match
original on `(Driver, Year, Race, LapNumber)`. On those matched rows:
TyreLife exact-match 11.8%, Stint 69.8%, Compound 76.7%, LapTime
median |diff| 1.04s.

## Normalized_TyreLife formula (recovered)

`Normalized_TyreLife = TyreLife / D(Driver, Race, Year, Stint)` — D is
unique per stint (5119 stints all have nunique=1). D ≈ stint length
(end-of-stint TyreLife). The original is row-truncated for some stints
(D > observed max TyreLife), so synth-side estimate
`TyreLife / max(TyreLife within stint)` only correlates r=0.45 with
the true Normalized_TyreLife. Direct lookup recovers it exactly.

## Lens 2 finding — quantization fingerprint → orig-transfer base

```
column                      synth_unique  orig_unique  overlap   pct_synth_rows_match
LapTime (s)                       37700        40779    33386       97.55%
LapTime_Delta                     44485        44544    34579       94.98%
Cumulative_Degradation            87470        68869    56872       87.38%
RaceProgress                       1614         1437     1391       99.95%
TyreLife / Position / LapNumber / Stint / Year / PitStop:    100.00%
```

Synth's continuous numerics are sampled near-exactly from the original's
empirical marginal — a CTGAN/CopulaGAN signature, not TabDDPM-Gaussian.

**Base built**: LGBM trained ONCE on the original (99.6k rows after
dropping nan-Compound + Pre-Season races), with `Normalized_TyreLife`
included as a feature, predicting on synth train+test. No CV needed
because the model never sees synth labels.

```
orig held-out AUC:           0.99690     (DGP is ~deterministic in original)
synth-train AUC (transfer):  0.85138     (joint corruption costs 14.5pt)
ρ(test) vs PRIMARY:          0.5653      (most-diverse single base since d9f FM_A)
```

## Gate / min-meta results (3 bases this branch)

| Base                   | Standalone OOF | ρ vs PRIMARY | min-meta Δ vs K=21 | Verdict |
|------------------------|---------------:|-------------:|-------------------:|---------|
| d15_decode_ntl         | 0.94162        | 0.9577       | **−0.008 bp**      | NULL    |
| d15_orig_transfer      | 0.85138        | 0.5653       | **+0.778 bp**      | **PASS**|
| d15_physics_residual   | 0.94228        | 0.9606       | **−0.036 bp**      | NULL    |

Combined K=21+2 (orig_transfer + d12_lr_meta): **+1.394 bp OOF**, ρ vs
PRIMARY 0.99493. Per-base |w|: orig_transfer 0.28 (fully consumed by
LR-meta), d12_lr_meta 4.69 (dominates).

## Why Lens 1 / 3 NULL but Lens 2 PASS

- **Lens 1 (decoded NTL only on 5.5% of rows)**: even where NTL is
  exact, it varies smoothly within a stint and is ~`TyreLife/D` —
  highly redundant with `TyreLife + Stint + RaceProgress + Compound`
  already in pool.
- **Lens 3 (physics-residual)**: residuals after Ridge fit on
  `Driver+Race+Year+Compound+TyreLife+Position+...` — same feature
  space as the GBDT pool already exploits, just expressed differently.
- **Lens 2 (orig-trained transfer)**: model is fit on a *different
  joint distribution* (the un-corrupted original DGP). Predictions on
  synth carry `P(PitNextLap | features under DGP_orig)` which the
  synth-trained models cannot represent because synth's joint is
  corrupted. ρ=0.565 vs PRIMARY confirms structural orthogonality.

## What this opens up (Day-15 follow-ups)

1. **Tune the orig-trained base** (Optuna LGBM, CatBoost on cat-cols,
   FM-class transfer) — current setup is single LGBM, default-ish
   params. Multiple architectures of orig-transfer should diversify
   further (each probably hitting ρ ≈ 0.56 vs PRIMARY).
2. **Mixed-source training**: train LGBM on `concat(orig, synth)`
   with sample-weights ∝ density-ratio. Should outperform either
   alone in low-density-of-synth regions.
3. **Pseudo-label transfer back**: orig-trained model on synth test
   gives diverse predictions; use top-decile-confidence rows as
   pseudo-labels for an additional base trained on synth+pseudo.
4. **Decode `LapTime`/`Cumulative_Degradation` per row**: for the
   94.98%/87.38% of rows whose continuous values land in original's
   set, look up the original rows with that exact value and use their
   row-features (Driver, Year, Race, Stint, etc.) as candidates for
   the synth row's "true preimage". Then features like "median
   PitNextLap among original rows with this LapTime" become a leaked
   posterior.

## Submit #2 (2026-05-06 14:43 UTC) — clean architecture-controlled probe

`submissions/submission_d15_path_b_K22_orig_transfer.csv`
Hier-meta Compound×Stint τ=20k on K=22 = K=21 + d15_orig_transfer
(same architecture as PRIMARY).

```
                                  OOF       ρ vs PRIMARY  flips +/−   LB
PRIMARY (K=21 hier-meta):       0.95083        1.000000      0/0    0.95049
K=22 + orig_transfer hier-meta: 0.95094        0.998440     36/144  0.95049 (TIE)
                                +1.127 bp                  total 180 (R7 ≤200 → HEDGE)
```

Pre-submit diff: ρ=0.9984, 50.9% rows differ >1e-3, max abs 0.106.

**LB ties at 5-decimal display** (0.95049 == 0.95049). Predicted-LB
band per `probe.py` at this ρ was [0, 0.5, 2] bp; tie sits on the
conservative end. Mechanism (orig_transfer base, ρ=0.565 vs PRIMARY
single-row) is **not falsified** — OOF lift +1.127 bp confirms hier-meta
extracts incremental signal from the orthogonal new-class base —
but the public-LB delta lands inside Kaggle's quantization floor.

Compare to Submit #1 (LR-meta K=22, +0.778 bp OOF, LB −10 bp): the
hier-meta gives ~1.5× the OOF lift AND removes the meta-arch confound,
landing within ε of PRIMARY instead of regressing. Confirms the 14 bp
LR→hier-meta gap dominates +0.78 bp base-add gains.

## Status update

- **PRIMARY** unchanged: `d13e_compound_stint_tau20000` LB 0.95049.
- **HEDGE candidate added**: `d15_path_b_K22_orig_transfer` LB 0.95049
  at ρ=0.998 — eligible for R5 final-3-day OOF-best probe (Rule 4) and
  R7 HEDGE slot (180 flips < 200 cap, no PI sign-off needed).
- **Submit budget**: today 3/10, total 27/270.
- **Mechanism family `external_data_aggregate`**: confirmed working
  (LB tie, OOF +1.127). Rule 19 BOTE prior (P=0.20, band [0, 1, 4] bp)
  remains calibrated; the LB upside likely sits at the band's lower
  end on this comp because synth public-LB is row-iid and the GBDT
  pool already captures most of the DGP via TyreLife/RaceProgress.

## Day-15 follow-ups (refreshed)

The hier-meta(K=22) probe **rules out a single-LGBM orig_transfer as
a structural advance** but **opens 3 cheap diversification paths**:

1. **Multi-arch orig_transfer bag** (~1h CPU). Train CatBoost +
   XGBoost + Optuna-tuned LGBM on the original 99k rows; each predicts
   on synth. Each base ≈ ρ=0.56 vs PRIMARY (orthogonal-class). Bag
   them or add as 3 separate K=22/K=23/K=24 bases. EV: each
   incremental orig-arch may add +0.5-1 bp OOF; bag may compound to
   +2-3 bp at hier-meta.
2. **Mixed-source LGBM** (~2-3h CPU): `concat(orig 99k + synth 439k)`
   with sample-weights ∝ density-ratio (AV-classifier). Risk: AV-AUC
   was 0.502 in d12, so density ratio likely uninformative; but the
   orig rows act as ground-truth anchors which AV can't see.
3. **Decoded-LapTime per-row leak**: 95% of synth `LapTime` values
   exist in original. For each synth row, look up the original rows
   with that exact LapTime, take median PitNextLap → leaked-posterior
   feature. Add to a synth-trained LGBM. EV: small standalone (the
   join is many-to-many) but might be the cleanest "data leak" feature.

## Artifacts

```
scripts/d15_decode_ntl.py
scripts/d15_orig_transfer.py            ← THE WIN
scripts/d15_physics_residual.py
scripts/artifacts/oof_d15_decode_ntl_strat.npy
scripts/artifacts/test_d15_decode_ntl_strat.npy
scripts/artifacts/oof_d15_orig_transfer_strat.npy
scripts/artifacts/test_d15_orig_transfer_strat.npy
scripts/artifacts/oof_d15_physics_residual_strat.npy
scripts/artifacts/test_d15_physics_residual_strat.npy
scripts/artifacts/probe_min_meta__d15_*.json
data/original/f1_strategy_dataset_v4.csv  (gitignored — 13MB; reproduce via
  `kaggle datasets download -d aadigupta1601/f1-strategy-dataset-pit-stop-prediction -p data/original`)
```
