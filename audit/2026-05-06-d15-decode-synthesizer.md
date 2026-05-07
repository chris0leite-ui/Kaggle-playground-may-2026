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

## Leak-lookup probe (2026-05-06 evening) — soft pass, marginal stack

`d15_leak_lookup` builds 16 EB-smoothed lookup features from the
aadigupta1601 original (univariate `P(PitNextLap | feature=v)` for
LapTime/LapTime_Delta/RaceProgress/Cumulative_Degradation/TyreLife/
Position/LapNumber/Stint/Compound/Race; bivariate for (LapTime,
TyreLife), (TyreLife, Compound), etc.; trivariate (TyreLife, Compound,
Stint)). Synth row → look up ÷ apply.

Standalone strongest leak features by AUC: `leak_tl_cmp_stint` 0.812,
`leak_rp_stint` 0.787, `leak_rp_cmp` 0.760, `leak_stint` 0.747.

LGBM with leak + standard features: standalone OOF 0.94203 (-67 bp
vs e3). ρ vs PRIMARY 0.959 — much higher than orig_transfer's 0.565
(less diverse, leak is closer to existing pool's signal).

Min-meta(K=21 + leak): **+0.270 bp OOF**, ρ=0.9956. Soft pass.

Hier-meta(Compound × Stint, τ=20k):

| Pool                  | OOF       | Δ K=22(orig) | ρ vs PRIM | Flips    |
|-----------------------|----------:|-------------:|----------:|----------|
| K=22 (leak alone)     | 0.95085   | −0.90 bp     | 0.99982   |  71 (R7 ✓) |
| K=23 (leak + orig)    | **0.95096** | **+0.19 bp** | 0.99861 | 198 (R7 ✓) |

**Verdict: leak-lookup alone is weaker than orig_transfer at hier-meta.**
Together they stack +0.19 bp incremental; K=23 is highest OOF on this
branch (+1.29 bp over K=21 baseline). ρ=0.9986 predicts LB tie at
PRIMARY 0.95049 per `probe.py` band. NOT submitted — slot saved.

Two decoded-data mechanisms now calibrated:
  - **model-transfer** (orig_transfer): +1.13 bp at K=22 hier-meta, LB tie
  - **per-row leak-lookup**: +0.20 bp at K=22 alone, +0.19 bp incremental
    when added on top of orig_transfer (K=23)

Both saturate the public-LB quantization floor on this comp. The
underlying DGP signal is recoverable but small; OOF gains are real
but live below ~5 bp LB resolution. Net for the comp: 1 HEDGE-tier
candidate (K=22 orig_transfer LB 0.95049 ρ=0.998); K=23 leak+orig
held as additional HEDGE candidate.

Trained 3 additional orig-trained bases:
  - `d15_orig_cb`     CatBoost  | held-out 0.99400 | synth 0.83722 | ρ vs PRIMARY 0.587
  - `d15_orig_xgb`    XGBoost   | held-out 0.99372 | synth 0.86585 | ρ vs PRIMARY 0.639
  - `d15_orig_lgbm_t` LGBM tuned| held-out 0.99725 | synth 0.85253 | ρ vs PRIMARY 0.568

Inter-arch ρ (synth test):
  transfer ↔ lgbm_t = 0.988 (REDUNDANT — dropped from probe pool)
  cb ↔ xgb = 0.941 (most-diverse intra-orig pair)
  all others ≈ 0.95

Hier-meta(Compound × Stint, τ=20k) probe results:

| Pool                     | OOF      | Δ vs K=21 | Δ vs K=22 | ρ vs PRIM | flips top-1%   |
|--------------------------|---------:|----------:|----------:|----------:|----------------|
| K=22 (transfer)          | 0.95094  | +1.13 bp  | —         | 0.99844   | 180 (R7 ≤200)  |
| K=23 (transfer + cb)     | 0.95094  | +1.10 bp  | +0.005 bp | 0.99854   | 202            |
| K=24 (transfer + cb + xgb)| 0.95097 | +1.43 bp  | +0.33 bp  | 0.99828   | 293 (over R7)  |

**Verdict: NULL on multi-arch diversification.** The 4 orig-trained
bases carry the same underlying DGP signal in slightly different
forms; LR-meta absorbs the additional architectures with marginal
gains (+0.005 bp / +0.33 bp). The single-arch LGBM (`d15_orig_transfer`)
captures most of the available orig-side signal at lower cost.

K=24 not submitted: (1) +0.326 bp OOF at ρ=0.998 predicts LB tie at
best per `probe.py` band; (2) 293 flips > R7 200-cap requires explicit
PI sign-off; net EV not worth the slot.

Friction tag: `external-data-arch-bag-redundant-when-shared-training-data`.
For future external-data work: vary either training-data subset OR
target-engineering, not just architecture.

## Artifacts (consolidated, 2026-05-06 wrap)

### Reproduction

```bash
# Original dataset (gitignored, 13MB):
kaggle datasets download -d aadigupta1601/f1-strategy-dataset-pit-stop-prediction \
    -p data/original
unzip data/original/*.zip -d data/original
```

### Scripts (committed)

```
scripts/d15_decode_ntl.py             # Lens 1: NTL lookup + stint-fraction est (NULL)
scripts/d15_orig_transfer.py          # Lens 2 base: orig-trained LGBM (THE WIN)
scripts/d15_physics_residual.py       # Lens 3: Ridge-residual physics (NULL)
scripts/d15_orig_multi_arch.py        # CB+XGB+tuned-LGBM orig variants
scripts/d15_leak_lookup.py            # 16 EB-smoothed leak features
scripts/d15_path_b_K22_orig_transfer.py  # hier-meta K=22 probe
scripts/d15_path_b_orig_bag.py        # hier-meta K=23/K=24 bag probe
scripts/d15_path_b_leak.py            # hier-meta K=22(leak)/K=23(leak+orig)
```

### OOF / test artifacts (all committed, harness-format `(n,2)`)

```
oof_d15_decode_ntl_strat.npy           test_*  (NULL min-meta)
oof_d15_orig_transfer_strat.npy        test_*  (HEDGE candidate base)
oof_d15_orig_cb_strat.npy              test_*  (orig CatBoost)
oof_d15_orig_xgb_strat.npy             test_*  (orig XGBoost)
oof_d15_orig_lgbm_t_strat.npy          test_*  (orig tuned LGBM, redundant w/ transfer)
oof_d15_physics_residual_strat.npy     test_*  (NULL min-meta)
oof_d15_leak_lookup_strat.npy          test_*  (soft-pass min-meta)
oof_d15_path_b_K22_orig_transfer_strat.npy   test_*  ← LB 0.95049 TIE (HEDGE)
oof_d15_path_b_K23_orig_transfer_cb_strat.npy   test_*  (NULL incremental)
oof_d15_path_b_K24_orig_transfer_cb_xgb_strat.npy  test_*  (R7 violation)
oof_d15_path_b_K22_leak_strat.npy      test_*  (weaker than orig)
oof_d15_path_b_K23_leak_orig_strat.npy  test_*  ← BEST OOF 0.95096 (held)
```

### Submitted CSVs

```
submissions/submission_d15_K22_add_orig_transfer.csv          # LB 0.95039 (LR-meta confound)
submissions/submission_d15_path_b_K22_orig_transfer.csv       # LB 0.95049 TIE — HEDGE
```

### Held-not-submitted CSVs (ready for R5 final-window probe)

```
submissions/submission_d15_path_b_K23_orig_transfer_cb.csv
submissions/submission_d15_path_b_K24_orig_transfer_cb_xgb.csv  (R7 cap exceeded)
submissions/submission_d15_path_b_K22_leak.csv
submissions/submission_d15_path_b_K23_leak_orig.csv             # BEST OOF held
```

### Probe JSONs (uniform gate reports)

```
scripts/artifacts/probe_min_meta__d15_*.json
scripts/artifacts/d15_path_b_K22_orig_transfer_results.json
scripts/artifacts/d15_path_b_orig_bag_results.json
scripts/artifacts/d15_path_b_leak_results.json
```

## Next steps (ranked by EV / cost)

Ordered for the next agent who consolidates branches. **Current PRIMARY
on origin/main has advanced to LB 0.95059** (B-GPU's `d15b_dae_only` +
`inv_laps_until_pit` features); this branch's `d15_orig_transfer`
HEDGE was tied at the OLD PRIMARY 0.95049, so the gap to the NEW
PRIMARY is −10 bp. The decoded-data thesis remains under-tested
against the new pool composition.

1. **Re-test d15_orig_transfer against the NEW K=22 pool** (~30 min CPU,
   no submit). Take main's new pool (K=21 + inv_laps_until_pit) and
   build hier-meta(K=23) with d15_orig_transfer added. If OOF lifts
   meaningfully (>= +0.5 bp over the new PRIMARY OOF), recompute ρ vs
   new PRIMARY test. Cost-justified because orig_transfer is
   structurally orthogonal (ρ=0.565 vs OLD PRIMARY) and inv_laps_until_pit
   is a within-synth derived feature; their signals likely don't overlap.

2. **Stack d15_orig_transfer + d15b_dae_only** (~30 min CPU). DAE
   embeddings + orig-DGP transfer are orthogonal mechanism families
   (one is internal-augmentation, the other is external-data). Probe
   K=23 = K=21 + dae_only + orig_transfer. EV: each contributed
   ~+1 bp OOF independently; combined band [+0.5, +2, +4] bp.

3. **Mixed-source LGBM** (~2-3 h CPU, parked). `concat(orig 99k +
   synth 439k)` with sample-weights ∝ density-ratio. The d12 AV
   probe found AV-AUC=0.502 (no domain shift), so density-ratio
   may degenerate to uniform — but orig rows act as ground-truth
   anchors regardless. Different mechanism axis from d15_orig_transfer
   (mixed training vs pure transfer).

4. **Per-row preimage join** (~2 h CPU, novel). For each synth row,
   find original rows with EXACT match on `(LapTime, TyreLife,
   Compound, Race)` (5.84% coverage). Use those rows' Normalized_TyreLife,
   exact Stint, exact Driver as recovered features. Combines Lens 1
   (lookup) + Lens 2 (use original directly) at row level — different
   from `leak_lookup`'s aggregate-then-apply pattern.

### Friction notes

- `tag: external-data-arch-bag-redundant-when-shared-training-data` —
  varying model architecture on the same training data gives
  diminishing returns (~ρ 0.94-0.99 inter-arch). Vary training data
  subset OR target engineering instead.
- `tag: meta-arch-required-for-orthogonal-base-eval` — pre-submit BOTE
  must specify which meta architecture (LR vs hier-meta) is used.
  LR vs hier-meta = ~14 bp on this comp, dominates +0.5-1 bp base-add
  gains. Today's Submit #1 lost 10 bp purely to this confound.
- `tag: lb-quantization-floor-defeats-decoded-data` — at ρ ≥ 0.998
  vs PRIMARY, even +1.13 bp OOF lifts land within Kaggle's ~5 bp LB
  resolution. Decoded-data signals on this comp are real but bounded
  by public-LB granularity.
```
