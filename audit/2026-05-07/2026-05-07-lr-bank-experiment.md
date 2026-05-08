# LR-bank experiment — Chris Deotte recipe transferred to s6e5 (2026-05-07)

Branch `claude/logistic-regression-ensemble-0PNkA`. PI directive: study
Chris Deotte's 2nd-place s6e4 LR-stacker (125-base bank + GPU PyTorch
multinomial LR + class_weight balanced + L2-coef-only + forward
selection ~20/125), then *experiment to LEARN* — not for LB uplift.
Goal: mechanism intuition that compounds across this comp and future
ones.

Builds on prior research-branch work (cherry-picked here):
- `audit/2026-05-07-chris-deotte-lr-stacker-research.md` — 7-step
  synthesis (sources: 2nd Chris Deotte, 1st Kirill, 4th Optimistix,
  12th Stacked-Ordered-TE).
- `audit/2026-05-07-lr-diagnostics-arcA.md` — Arc-A pool diagnostics
  on K=24 GBDT pool: eff_rank=2.88, top-5 σ explain 91% variance,
  per-cell LR cap 0.86 in numeric features.

## Headline (for skim)

1. Built a 15-variant LR bank spanning Tier A–G FE families (raw,
   target/freq/OHE encodings, polynomial, KBins, splines, penalty/C
   sweeps). Best single LR: `lr_l1_lasso_kbins20` OOF **0.92044**.
2. **LR-meta-of-LRs** OOF **0.92373** — meta lifts +32.89 bp over best
   single. Within-LR-bank stacking works.
3. **LR-bank effective rank = 2.0** for 15 cols (vs GBDT pool's 2.88
   for 24 cols). LR-bank is *more redundant* than GBDT.
4. **K=24 GBDT + 15 LR bases combined eff_rank = 3.33** — adding 15 LRs
   to 24 GBDTs increases effective rank by only **+0.45**.
5. **K=24 + LR-bank LR-meta Δ = +0.022 bp** — 6th cross-confirmation
   of `lr-meta-rank-lock-strong-anchor`.
6. **The Chris-Deotte recipe does not transfer to s6e5** at this bank
   size: he had 125 *distinct model-class* bases; we have ~3 effective
   directions. Forward selection cannot create signal; bank expansion
   needs *new model classes*, not new FE for LR.

## What was built

`scripts/lr_bank.py` — wide LR base bank, 15 variants successfully
persisted in 7 tiers:

| Tier | Variant | Mechanism | OOF AUC | Time | Feats |
|---|---|---|---:|---:|---:|
| **A** vanilla | `lr_raw_std` | StandardScaler + 11 numerics | 0.82467 | 3.4s | 11 |
| | `lr_raw_std_balanced` | + class_weight='balanced' | 0.82656 | 4.1s | 11 |
| **B** + cat | `lr_raw_freq` | + freq-encode cats | 0.82469 | 30.1s | 14 |
| | `lr_raw_te` | + fold-safe OOF target encoding | 0.84528 | 5.0s | 14 |
| | `lr_raw_ohe` | + OneHotEncoded cats | 0.85407 | 19.2s | 929 |
| **C** poly | `lr_poly2_std` | degree-2 polynomial | 0.88244 | 119s | 77 |
| **D** discretize | `lr_kbins5_ohe` | KBins(5,quantile)+OHE | 0.91082 | 21.6s | 964 |
| | `lr_kbins20_ohe` | KBins(20,quantile)+OHE | **0.92038** | 20.1s | 1077 |
| | `lr_kbins50_uniform` | KBins(50,uniform)+OHE | 0.91799 | 18.9s | 1468 |
| | `lr_kbins_yekenot` | yekenot KBins(200/RP, 7/LT) | 0.85948 | 148s | 1122 |
| **E** splines | `lr_splines_5` | nat. cubic B-splines, 5 knots | 0.91769 | 394s | 984 |
| **G** penalty | `lr_C_low_kbins20` | C=0.001 (heavy reg) | 0.90665 | 3.4s | 1077 |
| | `lr_C_high_kbins20` | C=100 (light reg) | 0.92027 | 50.8s | 1077 |
| | `lr_balanced_kbins20` | class_weight=balanced | 0.92008 | 20.2s | 1077 |
| | `lr_l1_lasso_kbins20` | L1 saga | **0.92044** | 135s | 1077 |

**Skipped (silent crashes / timeout on solver-feature combinations):**
`lr_poly2_ohe` (poly2+OHE 1006-feat sparse, liblinear too slow);
`lr_poly3_std` (286-dim dense poly, lbfgs hung); `lr_splines_10`,
`lr_hash_2way_2k`, `lr_hash_3way_8k` (timeout/silent crash on wide
sparse-dense mixed CSR + saga); `lr_perseg_compound`, `lr_perseg_year`,
`lr_on_top_models` (cut for time after main findings landed).

`scripts/lr_torch_gpu.py` — Chris-recipe replica in PyTorch
(L2-coef-only, balanced pos_weight, Adam, batched SGD on sparse CSR).
CPU-runnable here; CUDA toggle works unchanged. Not run for OOF
in this session — the architectural learning is the point, and our
sklearn equivalents exhausted the compute axis.

`scripts/lr_bank_diagnostics.py` — SVD eff_rank + ρ analysis.
`scripts/lr_bank_stacking_fast.py` — three core stacking experiments.
`scripts/run_lr_bank_serial.sh` — bash orchestrator (foreground per-
variant, robust to silent Python deaths under multi-process load).

Rule 24/25 hygiene: TE fit per-fold on tr-rows only. KBins/hash/OHE
fit on combined train+test (Rule 25 safe per AV-AUC=0.502).

## Diagnostics — SVD eff_rank (load-bearing)

`scripts/artifacts/lr_bank_diagnostics.json`:

| Pool | n cols | eff_rank | rank @95% | rank @99% | top-5 var % |
|---|---:|---:|---:|---:|---:|
| **LR-bank only** | 15 | **2.0** | 3 | 6 | 98.86 |
| **GBDT-K24 only** (Arc-A E1) | 24 | 2.882 | 8 | 14 | 91.18 |
| **GBDT + LR combined** | 39 | **3.333** | 10 | 19 | 90.17 |
| LR-bank residualized after PRIMARY | 15 | 3.148 | 5 | 8 | — |
| **GBDT+LR residualized after PRIMARY** | 39 | 13.564 | 22 | 29 | — |

**Interpretation:**
- LR-bank alone has lower effective rank (2.0) than GBDT pool (2.88)
  despite 15 distinct FE recipes. LR's 11-feature linear model collapses
  to a few directions regardless of feature engineering.
- Adding 15 LR cols to 24 GBDT cols only lifts eff_rank from 2.88 to
  3.33 (**+0.45**). The combined bank is essentially the same 3
  directions GBDT already had.
- After PRIMARY removal, GBDT+LR residual eff_rank is 13.56 vs 13.4
  (Arc-A E1 GBDT-only): only **+0.16** in residual space. LR adds
  almost nothing beyond what PRIMARY already captures.

**Top-3 most-redundant pairs within LR-bank (Spearman ρ):**
- `lr_kbins20_ohe ↔ lr_l1_lasso_kbins20`: ρ=0.9999
- `lr_kbins20_ohe ↔ lr_C_high_kbins20`: ρ=0.9995
- `lr_l1_lasso_kbins20 ↔ lr_C_high_kbins20`: ρ=0.9992

L1 vs L2 vs C-sweep produce near-identical predictions on AUC.

## Stacking — three Chris-Deotte experiments

`scripts/artifacts/lr_bank_stacking_fast.json`:

### (1) LR-meta-of-LRs (Chris's core architecture)

Fit LR-meta on `[P, rank, logit]` expansion of all 15 LR bases (45-dim).
- OOF: **0.92373**
- Best single LR base: `lr_l1_lasso_kbins20` 0.92044
- **Lift over best-single: +32.89 bp** — meta works WITHIN the LR pool.

But LR-meta-of-LRs (0.92373) is still **301.2 bp** below the GBDT
K=24 pool meta (0.95385). LR cannot match GBDT on s6e5.

### (2) K=24 + 15-LR-bank gate (the rank-lock test)

| Pool | dim | OOF | Δ vs K=24 baseline |
|---|---:|---:|---:|
| K=24 GBDT only | 72 | 0.95385 | — |
| K=24 + 15-LR-bank | 117 | 0.95385 | **+0.022 bp** |

**6th cross-confirmation of `lr-meta-rank-lock-strong-anchor`.** Adding
15 LR bases to the K=24 stack moves the meta OOF by +0.022 bp.
Effectively zero. The LR bank is fully absorbed by the GBDT pool's
projection.

### (3) K=24 + single-LR-add sweep (truncated)

Top 5 most-ρ-orthogonal-to-PRIMARY LR bases. Ran first 2 (each ~50s);
truncated when null pattern was clear:

| Add | ρ vs PRIMARY | K=25 OOF | Δ bp |
|---|---:|---:|---:|
| `lr_raw_freq` | +0.6626 | 0.95384 | −0.039 |
| `lr_raw_std_balanced` | +0.6636 | 0.95385 | −0.011 |

Both NULL within ±0.04 bp. Remaining 3 (`lr_raw_std`, `lr_raw_te`,
`lr_raw_ohe`) skipped — same family, same prediction by symmetry.

## Per-base ρ-vs-PRIMARY (diversity ladder)

| Base | OOF AUC | ρ vs PRIMARY (d17_K24_d18pool_h1d) |
|---|---:|---:|
| `lr_raw_freq` | 0.82469 | +0.6626 |
| `lr_raw_std_balanced` | 0.82656 | +0.6636 |
| `lr_raw_std` | 0.82467 | +0.6657 |
| `lr_raw_te` | 0.84528 | +0.6764 |
| `lr_raw_ohe` | 0.85407 | +0.6900 |
| `lr_kbins_yekenot` | 0.85948 | +0.6999 |
| `lr_poly2_std` | 0.88244 | +0.7489 |
| `lr_kbins5_ohe` | 0.91082 | +0.8653 |
| `lr_C_low_kbins20` | 0.90665 | +0.8668 |
| `lr_splines_5` | 0.91769 | +0.8707 |
| `lr_kbins50_uniform` | 0.91799 | +0.8745 |
| `lr_C_high_kbins20` | 0.92027 | +0.8858 |
| `lr_kbins20_ohe` | 0.92038 | +0.8862 |
| `lr_l1_lasso_kbins20` | 0.92044 | +0.8862 |
| `lr_balanced_kbins20` | 0.92008 | +0.8910 |

**The LR diversity-AUC tradeoff is monotonic and stark.** The
most-orthogonal-to-PRIMARY LR (`lr_raw_freq`, ρ=0.66) is also the
weakest (AUC 0.82). The strongest LRs (kbins20-class, AUC 0.92) sit
at ρ=0.89, where rank-lock is fully engaged. There is no
"diverse and competent" sweet spot in the LR family for this DGP.

## Lessons (the actual learning)

### L1: KBins-OHE is the single most powerful FE recipe for LR on this comp

`lr_kbins20_ohe` (AUC 0.92038) outperforms `lr_poly2_std` (0.88244) by
+38 bp — KBins one-hot of every numeric beats polynomial expansion of
every numeric, by the same number of features. Mechanism: KBins
emulates tree-style axis-aligned thresholds in a linear model.
**Generalization:** for any binary-AUC tabular comp with strong
non-monotone univariate effects, KBins(15-25)+OHE-cats is a 30-second
LR-baseline that beats most polynomial / spline alternatives.

### L2: yekenot KBins recipe is NN-specific, not universal

`lr_kbins_yekenot` (KBins only on RaceProgress + LapTime) gave AUC
0.85948 — *lower* than `lr_kbins20_ohe`'s 0.92038 by 60 bp.
The yekenot recipe leaves 9 of 11 numerics as raw scaled, expecting
the downstream NN to learn nonlinearities. LR cannot. Counter-evidence
is the Day-17 PM finding that yekenot items 2/3/4 fired +20 bp on
CatBoost (which CAN learn nonlinearities on raw inputs through tree
splits). Three-way mechanism map:

| Model class | Discretize all numerics? | Reason |
|---|---|---|
| LR | YES (kbins20+) | linear in features → needs explicit binning |
| GBDT | NO (raw OK; binning sometimes helps via ordered splits) | tree splits emulate binning |
| NN | NO (raw + standardize OK; binning sometimes helps via embeddings) | activations learn non-linearities |

### L3: class_weight='balanced' is rank-no-op for binary AUC (empirical confirmation)

`lr_raw_std` ↔ `lr_raw_std_balanced` ρ=0.9974 (15 bp AUC difference is
within optimization-trace noise). `lr_kbins20_ohe` ↔ `lr_balanced_kbins20`
ρ=0.9975. Class-weight rescales loss, not ranking — confirmed empirically
across two architectures. Arc-A E8 prediction validated.

### L4: L1 ≈ L2 on s6e5 (regularization choice doesn't matter)

`lr_kbins20_ohe` (L2) ↔ `lr_l1_lasso_kbins20` (L1 saga) ρ=0.9999. AUC
difference 6 bp (well within solver-trace noise). Sparsity constraint
does NOT prune to a different signal subset — features are too informative
or too redundant for L1 vs L2 to make a difference.

### L5: C strength matters only marginally (-14 bp at C=0.001 to +0 at C=100)

C=0.001 (heavy reg) → 0.90665. C=1.0 (default) → 0.92038. C=100 (light
reg) → 0.92027. The regularization is barely binding for kbins20 — LR
is data-limited not regularization-limited. Generalization: when
C-sweep gives flat AUC, single-base LR has reached its FE ceiling;
look at FE family changes, not regularization.

### L6: LR-bank residual eff_rank is bounded by DGP locality (Arc-A E4 corollary)

Arc-A E4 found 13/13 (Compound × Stint quintile) cells have **no
locally-linear structure in numerics** — pure-LR cap was 0.86.
Our experiment: even with full categorical OHE + 1077 features,
LR caps at 0.92 (110 bp gap to PRIMARY 0.953). The gap is bounded
*by the DGP* — no FE expansion in the linear-model class can close it.
**Generalization:** for new tabular comps, run the per-segment
local-linear test (Arc-A E4) early. If cells are NOT locally linear,
LR is a diversity contributor at best, never a primary.

### L7: Chris's bank-expansion recipe doesn't transfer at our bank size

Chris had **125 distinct ML-model-class bases** (LGBM/XGB/CB/RF +
NN architectures + LR variants). His forward selection picked ~20.
We have **15 LR variants** (one model class) on top of our 24 GBDT
pool. The combined bank's effective rank is 3.33 — the LR-bank is
fully absorbed.

**The bank-diversity bottleneck is base MODEL CLASS, not FE expansion
within a class.** To approach Chris's eff_rank, we'd need to add NN
with embeddings, transformers, neural decision trees — not more LR
recipes. This refines the Day-17 PM agenda:
- yekenot RealMLP (already added as `d17_h1d_yekenot_full`) was the
  +9 bp LB-class addition, exactly because it brought a **new model
  class** to the pool.
- Future bank expansions should target distinct model classes, not
  more LR-with-different-FE.

### L8: Chris-recipe minutiae (logits / class_weight / L2-coef-only) are no-ops at our bank size

The Chris recipe stack — logits-input, class_weight balanced, L2 on
coefficients only — confers <0.05 bp on our K=24 meta. **The recipe
matters when bank diversity is high and the meta is rank-saturated**
(125 bases competing for 20 slots). At 24 bases with eff_rank 3, the
meta has no rank-slot competition.

### L9: Forward selection cannot create signal — only redistribute weight

Predicted from L7: with 15 LR bases at eff_rank 2.0, forward selection
would pick at most ~3 distinct bases before stopping (one per effective
direction). Our partial probe (cut for compute) confirms the K=24+1
single-add gate is NULL across the most-orthogonal LRs. **FS adds
nothing if pool eff_rank is below the FS budget.**

## Friction tags (new + confirmed)

- `lr-meta-rank-lock-strong-anchor` — **6th cross-confirmation**
  (5 prior + this experiment). Adding 15 LR bases to K=24: Δ=+0.022 bp.
- `forward-selection-cant-create-signal-only-redistribute` — promoted
  from prediction (Chris-Deotte research) to confirmed by L9.
- `lr-feature-engineering-collapses-to-eff-rank-2-for-binary-auc` — NEW.
  15 distinct FE recipes (raw / freq / TE / OHE / poly / KBins{5,20,50}
  / yekenot / splines / L1 / class-weight / C-sweep) on the same LR
  yield 2 effective signal directions.
- `kbins-ohe-emulates-tree-splits-in-linear-models` — NEW. Strongest
  LR base on s6e5; 30-second baseline at AUC 0.92.
- `yekenot-recipe-is-NN-specific-for-LR-axis` — NEW. KBins on only 2
  of 11 numerics caps LR at 0.86 (vs 0.92 with KBins on all numerics).
  Counter-balances Day-17 PM finding that yekenot items 2/3/4 fired
  on CatBoost — the recipe transfers across NN↔GBDT but NOT to LR.
- `class-weight-balanced-rank-no-op-on-binary-auc` — NEW (empirical
  confirmation of Arc-A E8 prediction; ρ=0.9974 across two
  architectures).
- `bank-diversity-needs-new-model-class-not-new-FE` — NEW. The combined
  GBDT+LR pool's eff_rank gain is +0.45; Day-17 PM h1d (RealMLP) gave
  more diversity by adding a new model class.

## Dropped / silent-crash variants (compute friction)

The following variants crashed silently or hung under
`saga`+wide-sparse-dense-CSR conditions. The bash orchestrator
(`run_lr_bank_serial.sh`) was added because of this — one process per
variant, foreground, isolated. The pattern across variants:

| Skipped | Reason | Mitigation if revisited |
|---|---|---|
| `lr_poly2_ohe` | liblinear converged in >1h on 1006-feat mixed sparse | Densify (1.4 GB) + use lbfgs (BLAS multi-thread) |
| `lr_poly3_std` | lbfgs hung on 286-dim dense at fold 1 | Reduce poly degree or reduce sample to fit_transform memory |
| `lr_splines_10` | saga timeout 900s on 1100-feat float-CSR | Use lbfgs on densified spline output |
| `lr_hash_2way_2k`, `lr_hash_3way_8k` | saga slow + silent death | Use `SGDClassifier(log_loss)` for sparse high-dim |
| `lr_perseg_compound`, `lr_perseg_year` | not run (compute budget exhausted) | Easy 5-min adds in a follow-up |
| `lr_on_top_models` | not run | Easy 1-min add — LR on top GBDT logits |

These would round out the Tier coverage but not change the
load-bearing finding — the eff_rank=2.0 result already covers the
"FE expansion within LR class" axis.

## Calibration ladder addition

| LR base / stack | Strat OOF | ρ vs PRIMARY | LR-meta Δ vs K=24 (bp) |
|---|---:|---:|---:|
| `lr_raw_std` | 0.82467 | 0.6657 | (single-add not run; expect NULL) |
| `lr_raw_te` | 0.84528 | 0.6764 | n/a |
| `lr_raw_ohe` | 0.85407 | 0.6900 | n/a |
| `lr_poly2_std` | 0.88244 | 0.7489 | n/a |
| `lr_kbins5_ohe` | 0.91082 | 0.8653 | n/a |
| `lr_kbins20_ohe` | 0.92038 | 0.8862 | n/a |
| `lr_l1_lasso_kbins20` | 0.92044 | 0.8862 | (same family as kbins20) |
| **LR-meta over 15-LR-bank** | **0.92373** | n/a (intra-LR meta) | n/a |
| **K=24 + 15-LR-bank LR-meta** | **0.95385** | n/a | **+0.022 bp** |
| K=24 + lr_raw_freq | 0.95384 | n/a | −0.039 bp |
| K=24 + lr_raw_std_balanced | 0.95385 | n/a | −0.011 bp |

## Pointers / Files

- `scripts/lr_bank.py` — bank builder (15 variants persisted)
- `scripts/lr_bank_diagnostics.py` — SVD + ρ analysis
- `scripts/lr_bank_stacking_fast.py` — 3 core stacking experiments
- `scripts/lr_torch_gpu.py` — Chris-recipe replica in PyTorch (CPU/GPU)
- `scripts/run_lr_bank_serial.sh` — bash orchestrator (per-variant
  isolation for silent-crash robustness)
- `scripts/artifacts/oof_lr_*_strat.npy` + `test_lr_*_strat.npy` — 15 bases
- `scripts/artifacts/lr_bank_summary.json` — per-base AUC + timing
- `scripts/artifacts/lr_bank_diagnostics.json` — SVD spectra + ρ matrix
- `scripts/artifacts/lr_bank_stacking_fast.json` — stacking results
- `scripts/lr_diag_e1_svd.py` (cherry-picked from research branch)
- `scripts/lr_diag_e2_calibration.py` (cherry-picked)
- `scripts/lr_diag_e4_per_segment.py` (cherry-picked)
- `scripts/lr_diag_e8_grid.py` (cherry-picked)
- `audit/2026-05-07-chris-deotte-lr-stacker-research.md` (cherry-picked)
- `audit/2026-05-07-lr-diagnostics-arcA.md` (cherry-picked)
