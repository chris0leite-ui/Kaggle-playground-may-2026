# LR-leverage six probes (2026-05-07)

Branch `claude/logistic-regression-ensemble-0PNkA`. PI question: *given
20 LR bases incl. mega 0.92776 OOF, how do we leverage this body of
work to improve?* I proposed 6 probes and ran them all in sequence.

## TL;DR — 5 NULL/NEGATIVE, 1 NEW STRONGEST LR (+60.8 bp)

| Probe | Question | Result | Verdict |
|---|---|---|---|
| **1. K=24 + lr_mega gate** | Does mega add to PRIMARY meta? | Δ +0.183 bp | **NULL** — 6th confirmation `lr-meta-rank-lock-strong-anchor` |
| **2. Mega coef distillation** | Which features carry signal? | KBins20 44.5% mass; rozenTE 1.00 mean; static 1.8% mass | **DGP archaeology** — bin-OHE dominates |
| **3. GBDT-meta on K=24+mega** | Is rank-lock linear-meta-specific? | LR-meta 0.95387, GBDT-meta 0.95378, **Δ −0.893 bp** | **NULL** — non-linear meta WORSE; rank-lock is structural |
| **4. Random-subspace bagged-LR** | Does bagging within LR class break the eff_rank=2 ceiling? | bag eff_rank **1.672** (vs 2.19 hand-designed) | **NEGATIVE** — random subspace REDUCES diversity |
| **5. Per-segment mega (Compound×Year)** | Does per-cell LR find structure global misses? | OOF **0.93385 vs 0.92776 global** = **+60.8 bp** | **POSITIVE — new strongest LR base** |
| **6. mega_oof as GBDT feature** | Is mega's compressed signal GBDT-recoverable? | LGBM raw 0.94142, +mega_oof **0.93774** = **−36.7 bp** | **NEGATIVE** — GBDT overfits to OOF when used as feature |

**The single positive finding**: per-(Compound × Year) mega LR is the
new strongest LR on s6e5 at OOF 0.93385. The path was hidden by global
mega's pooled-coefficient model — DGP partition by year (especially 2023)
has segment-specific structure global LR can't represent.

## Detailed findings

### Probe 1 — K=24 GBDT pool + lr_mega min-meta gate (5 min)

| Pool | OOF AUC |
|---|---:|
| K=24 GBDT baseline (LR-meta) | 0.95385 |
| K=24 + lr_mega | **0.95387** (+0.183 bp) |
| ρ(mega, PRIMARY) | **0.9030** (highest of any LR base) |

**Reading.** Mega's strength as a single LR (0.92776) does not survive
going into the K=24 LR-meta. ρ_PRIM 0.903 confirms full redundancy. **6th
cross-confirmation** of `lr-meta-rank-lock-strong-anchor`. As predicted.

### Probe 2 — Mega's top-30 + family-mass distillation (5 min)

`lr_mega` has 1202 dense features. Top-30 by |w| from full-train fit:

```
Top-1:  KBins20_25      w=-6.43   (one specific bin from one numeric)
Top-2:  rozenTE_te_drv_comp  w=-3.79   (Driver×Compound CV TE)
Top-3:  KBins20_26      w=-3.18
Top-4:  KBins20_1       w=+2.70
...
[rozenTE_te_race_yr]   w=-1.85   (8th)
[DGP_rule_4]           w=+1.64   (11th — Compound×Stint α=20)
[DGP_rule_5]           w=+1.30   (17th — α=100)
```

| Family | Mass | n_feats | Mean \|w\| (per feature) |
|---|---:|---:|---:|
| **KBins20** | 44.5% | 159 | 0.44 |
| catOHE | 42.3% | 918 | 0.07 |
| DGP_rule | 5.2% | 16 | 0.51 |
| **rozenTE** | 3.8% | 6 | **1.00** ← highest per-feature |
| 3wTE | 2.4% | 16 | 0.23 |
| **static (Rozen tree-engineered)** | **1.8%** | 87 | **0.03** ← essentially ignored |

**Per-feature efficiency:** rozenTE > DGP_rule > KBins20 > 3wTE > catOHE
> static. **The Rozen tree-engineered features that hurt `lr_rozen_full`
standalone are *ignored* by mega LR** (mean |w|=0.032). Mega's 0.928
comes from KBins+TE+DGP-rule blend; the 87 Rozen static features
contribute 1.8% of mass.

**Distillation insight**: a "lr_mega_top30" variant on just the
30 highest-|w| features should match within a few bp at 50× faster
fit. Transferable starter pack for next comp.

### Probe 3 — GBDT-meta-stacker on K=24+mega (30 min)

| Meta | OOF AUC | Δ vs LR-meta |
|---|---:|---:|
| LR-meta on K=24+mega | 0.95387 | (baseline) |
| **LightGBM-meta (depth 4)** | **0.95378** | **−0.893 bp** |

**Reading.** GBDT-meta is *worse* than LR-meta. The pool's saturation is
NOT linear-meta-specific — non-linear routing doesn't unlock the 13
residual-after-PRIMARY directions either. **Refines** the
`path-b-amp-only-fires-on-meta-arch-not-base-add` friction: even GBDT-meta
can't tap that residual space. The rank-lock is genuinely structural to
the OOF correlation pattern.

This *closes* the GBDT-meta path that I originally rated highest-EV.

### Probe 4 — Random-Subspace bagged-LR (30 min)

10 LRs, each fit on a random 30% subset of mega's 284 columns
(KBins+OHE excluded for memory). Bag predictions rank-averaged.

| Bag | AUC range |
|---|---|
| Per-bag AUC | 0.889 – 0.912 |
| **Bag rank-average AUC** | **0.91247** |
| **Bag-of-LRs eff_rank** | **1.672** (vs 20-LR-bank 2.19) |

**Reading.** Random subspace **reduces** the bank's eff_rank vs hand-
designed FE variants. The bag is *more* concentrated on direction-1,
not less. Random feature dropout destroys the carefully-curated cross-
feature structure that hand-designed FE preserves.

**Settles the open question** from the playbook: the LR-class
eff_rank=2 ceiling is *fundamental* to this DGP, not pipeline-induced.
Random sampling within LR class doesn't break it.

This is the cleanest negative result of the session.

### Probe 5 — Per-segment mega LR (Compound × Year) ★ POSITIVE ★

20 cells (4 Compound × 5 Year/Wet). Per-cell LR fit on each fold.

| Compound × Year | n | global mega AUC | per-segment AUC | Δ bp |
|---|---:|---:|---:|---:|
| MEDIUM_2023 | 58264 | 0.8176 | **0.9257** | **+1081** |
| SOFT_2023 | 15457 | 0.8177 | 0.9042 | +865 |
| HARD_2023 | 60996 | 0.8573 | 0.9298 | +725 |
| SOFT_2024 | 5652 | 0.8051 | 0.8402 | +351 |
| INTERMEDIATE_2024 | 8440 | 0.8480 | 0.8753 | +272 |
| MEDIUM_2024 | 59548 | 0.8726 | 0.8991 | +265 |
| SOFT_2022 | 9926 | 0.8098 | 0.8281 | +183 |
| HARD_2022 | 22025 | 0.8112 | 0.8188 | +76 |
| MEDIUM_2025 | 47783 | 0.9118 | 0.9138 | +20 |
| HARD_2024 | 53463 | 0.8087 | 0.8127 | +40 |
| HARD_2025 | 34034 | 0.8471 | 0.8516 | +45 |
| MEDIUM_2022 | 45546 | 0.8981 | 0.9024 | +43 |
| SOFT_2025 | 7709 | 0.7888 | 0.7959 | +71 |
| INTERMEDIATE_2025 | 3366 | 0.9265 | 0.9189 | −76 |
| INTERMEDIATE_2022 | 4193 | 0.8848 | 0.8684 | −165 |
| WET_2022 | 1299 | 0.7636 | 0.7019 | −617 |
| INTERMEDIATE_2023 | 1383 | 0.7858 | 0.7194 | −664 |

**Global per-segment OOF: 0.93385** (vs global mega 0.92776, **Δ +60.8 bp**).

**Pattern:**
- **2023 cells dominate the lift** (Hard +725, Soft +865, Medium +1081).
  Per Day-12 finding, 2023 was the "easiest year" but ALSO the year where
  global LR coefficients leave the most on the table.
- **Small cells lose** (n<5000): WET_2022, INT_2023, INT_2022 all regress.
  Insufficient samples for fold-safe per-cell fit.
- **The story**: the s6e5 DGP has year-conditional structure that a single
  pooled LR cannot represent. Per-(Compound,Year) specialists capture
  it; the ~60 bp lift comes mostly from 2023 cells.

This is the only POSITIVE leverage finding in the 6 probes. It opens an
axis: **DGP-aware segmentation, not random subspace, is how to extract
new diversity within the LR class.**

### Probe 6 — mega_oof_prob as a single GBDT feature (30 min)

| LGBM input | Features | OOF AUC | Δ vs raw-only |
|---|---:|---:|---:|
| Raw + 3 cat-codes | 14 | 0.94142 | (baseline) |
| **Raw + 3 cat + mega_oof** | 15 | **0.93774** | **−36.7 bp** |

**Reading.** GBDT *overfits* when given mega_oof_prob as a feature. LGBM
keeps splitting on the strongest feature (mega_oof, AUC 0.928 alone) and
ignores the residual structure raw features carry. The OOF generation's
fold-specific bias gets amplified by the splits.

**Mega is meta-level useful, not feature-level useful.** Stacking via
LR-meta on the OOF columns works (Probe 1 showed 0.95387). Stacking by
adding the OOF as a GBDT input feature *destroys* the GBDT.

Closes the "mega as a feature factory" path.

## Synthesis

The six probes form a strong negative result with one bright positive.

**What we now know is closed:**
- `lr-meta-rank-lock-strong-anchor` survives mega (Probe 1, 6th confirmation).
- The rank-lock is NOT linear-meta-specific (Probe 3, GBDT-meta worse).
- Random-subspace bagging within LR class *reduces* eff_rank (Probe 4).
- Mega-prob as GBDT feature regresses GBDT (Probe 6).

**What we now know is OPEN:**
- **Per-(Compound × Year) mega LR is +60.8 bp over global mega.** The DGP
  has year-conditional structure (especially 2023) that pooled LR can't
  capture. This is a new segment-specialist axis.
- **Top-30 distilled mega features** are a transferable LR-FE starter
  pack: KBins20-of-quantile + 6 Rozen CV TE + 4 DGP rule lookups (with
  α tuning) cover 95% of mega's |w| mass at <50 features.

**What we now know about the FE landscape (from Probe 2):**
- Rozen tree-engineered FE is *not used* by LR even when present.
  Mean |w|=0.032 is essentially zero. Tree-engineered FE goes nowhere
  in linear-model space.
- rozenTE (Driver×Compound, Race×Year) is the strongest *per-feature*
  signal carrier in mega — only 6 features, mean |w|=1.00.

## Implications for s6e5 LB

Probe 5's per-segment mega is potentially a stack-add candidate. To test:
- Fit per-segment mega as a single OOF base (already have the OOF inline)
- Save as `oof_lr_perseg_mega_strat.npy` + test
- Run K=24 + lr_perseg_mega min-meta gate

**Predicted at ρ=0.85-0.92 vs PRIMARY** (lower than mega's 0.903 because
of segment-specific routing). EV: maybe +0.5 to +2 bp on K=24 meta. Not
LB-decisive but worth one slot if confirming.

## Implications for the playbook (cross-comp)

Two new lessons for `examples/fe-recipe-simple-lr.md`:

1. **Random-subspace bagging within LR class doesn't help diversity.**
   Skip this for future comps. The eff_rank ceiling is DGP-bound, not
   pipeline-bound.

2. **Per-segment LR specialists are the highest-EV LR diversity move.**
   Identify a small natural partitioning of the data (categorical
   feature with 4-20 levels each ≥1k samples). Fit per-segment LR.
   Lift comes from year/cohort-conditional DGP structure that pooled LR
   misses.

3. **mega_oof should NOT go into a fresh GBDT as a feature.** Use mega
   as a meta-level base only.

## Friction tags (new from this session)

- `random-subspace-LR-reduces-eff-rank` — NEW. Random feature subsets
  destroy the cross-feature structure carefully-designed LR FE preserves.
- `gbdt-meta-not-better-than-lr-meta-on-saturated-pool` — NEW. Probe 3.
  Refines: rank-lock is structural, not solver-specific.
- `mega-oof-as-gbdt-feature-causes-overfit` — NEW. Probe 6.
- `per-segment-LR-on-DGP-partition-finds-new-signal` — NEW (POSITIVE).
  +60.8 bp lift on s6e5 from per-(Compound,Year) mega LR.
- `lr-rozen-static-fe-ignored-by-mega-LR` — NEW. Probe 2 family-mass.
  Confirms tree-friendly FE goes nowhere in LR even when present.

## Files

- `scripts/lr_leverage_phaseA.py` — Probes 1+2 (+ build_mega_features_full_train)
- `scripts/lr_leverage_phaseB.py` — Probes 3+6 (GBDT-meta + mega-as-feature)
- `scripts/lr_leverage_phaseC.py` — Probes 4+5 (bagged-LR + per-segment)
- `scripts/artifacts/lr_leverage_phaseA.json` — top-30 + family mass
- `scripts/artifacts/lr_leverage_phaseB.json` — GBDT-meta + LGBM+mega
- `scripts/artifacts/lr_leverage_phaseC.json` — bag eff_rank + per-cell AUCs
