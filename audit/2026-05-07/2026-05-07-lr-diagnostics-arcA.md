# LR-diagnostics Arc A — pool/meta archaeology (2026-05-07)

Branch `claude/ensemble-logistic-regression-research-MbLKu`. Reframe
from PI: criterion is now **knowledge build that compounds across
this comp and future ones**, not LB Δ. Arc A = 4 cheap diagnostics
on K=24 = K=21 + d16_cont_only + p1_cb_v3_gpu + d17_h1d_yekenot.

Scripts: `scripts/lr_diag_e{1,2,4,8}_*.py`. JSONs:
`scripts/artifacts/lr_diag_e{1,2,4,8}_*.json`.

## E1 — SVD effective rank of K=24 (load-bearing)

Standardize columns; SVD; entropy / cumulative-variance ranks.

| Representation | eff_rank (entropy) | rank @ 95% | rank @ 99% | top-5 σ var % |
|---|---:|---:|---:|---:|
| Probability | 1.78 | 4 | 10 | — |
| **Logit (LR-meta sees)** | **2.88** | **8** | **14** | **91.2** |
| Residualized after PRIMARY | 13.44 | 18 | 22 | — |

Stable rank logit = 1.28 (rank-1 dominated). Top-15 Spearman ρ pairs
on logits all >0.97; 7 of 15 are within-class clones (CatBoost↔CatBoost,
HGBC↔HGBC). Most-redundant: **`cb_year-cat` ↔ `cb_slow-wide-bag` ρ=0.9963**.

**Interpretation.** The K=22+ saturation has a quantitative root
cause: 24 nominal bases collapse to ≈3 effective signal directions;
top-5 of 24 explain 91% of variance. The LR-meta has 24 columns but
~3 independent things to allocate weight to. Residualizing PRIMARY's
prediction direction lifts eff_rank to 13.4, so there are 13 dimensions
of *residual* variation left — accessible only via meta-architecture
that re-allocates the dominant direction (Path-B-amp evidence).

This explains 4 prior frictions in one shot:
- `lr-meta-rank-lock-strong-anchor` — structural, not unlucky
- `path-b-amp-only-fires-on-meta-arch-not-base-add` — base-adds
  project onto direction-1; meta-arch redesign re-allocates the 13
  residual directions
- `rho-alone-insufficient-for-meta-utility` — high ρ to PRIMARY means
  high projection onto direction-1; the meta-utility lives in the
  residual 13
- `path-b-amp-needs-orthogonal-signal-not-meta-derivatives` — adding
  meta outputs increases direction-1 redundancy not residual rank

## E2 — Per-base LR calibration map

Per-fold LR(logit_b → y); record slope, intercept, Brier improvement.

- 21/24 bases have |slope − 1| < 0.1 (well-calibrated)
- median Brier-improvement-from-Platt = 0.10 bp → LR-meta is **not**
  spending coefficients on calibration
- 4 mis-calibrated bases:

| base | slope | intercept | Brier-Δ (bp) |
|---|---:|---:|---:|
| `cb_slow-wide-bag` | +1.95 | −3.14 | **+1182** |
| `a_horizon` | +1.21 | −1.97 | +714 |
| `b_lapsuntilpit` | +0.91 | −0.96 | +103 |
| `d16_orig_continuous_only` | +0.81 | −0.13 | +8 |

AUC is rank-invariant under Platt → these mis-calibrations don't hurt
our current LB. But: **`cb_slow-wide-bag` is BOTH the most-redundant
base (ρ=0.9963) AND the most mis-calibrated** — cleanest pool-surgery
candidate.

## E4 — Per-segment LR AUC

13 cells (Compound × Stint-quintile, n≥200, both classes). Within-cell
class-weighted LR on 11 numeric features (no Driver/Race/Compound).

- LR per-cell AUC range: 0.74–0.86
- PRIMARY per-cell AUC range: 0.84–0.96
- 13/13 cells show PRIMARY > LR by ≥559 bp; median gap 838 bp
- Largest gaps in rare cells (INTERMEDIATE|q2 +1638 bp)

Caveat: per-cell LR doesn't see Driver/Race; comparison conflates
within-cell-nonlinearity with categorical-information-gap. A clean
follow-up: global LR with Compound+Driver dummies, per-cell evaluation.

**Reading.** No (Compound × Stint) cell is locally-linear in numeric-
only features. A pure-LR base on raw numerics is capped ~0.86 cell-AUC.
Sets a **floor for new-base diversity tests** (A2/A3/A4 from the plan).

## E8 — class_weight × C × penalty × input grid (LR-meta)

Pruned to 20 fits (l2 lbfgs, 2 input modes × 2 cw × 5 C); original
saga+l1+tiny-C corner ran 1h+ before kill, salvaged 2 datapoints.

| Input mode | Δ bp range | min ρ vs anchor |
|---|---:|---:|
| **full (P + rank + logit, 72d)** | [−0.03, +0.08] | 0.9996 |
| **logit-only (24d)** | [−0.93, −0.87] | 0.9986 |
| salvaged L1 saga (C ∈ {0.01, 0.1}) | {−0.59, +0.08} | 0.9989 |

Anchor AUC 0.953845 = max of all 20 points within 0.08 bp.

**Three load-bearing findings:**

1. **Class-weight × C × L1/L2 are rank-no-op for binary-AUC LR-meta.**
   The 12th-place s6e4 "three axes must all be true" recipe applies to
   multinomial balanced-accuracy, NOT to binary AUC. Friction tag
   candidate: `s6e4-recipe-3-axes-do-not-transfer-binary-auc`.
2. **Logits-only is uniformly ~0.9 bp worse.** Dropping rank+P
   features removes real signal. The probability and rank
   transformations of the same OOF column carry distinct
   meta-information.
3. **Production config (full input, cw=None, C=1.0) is at ceiling**
   to within sub-bp. We are not leaving signal on the table from
   meta hyperparameters.

## Synthesis (Arc A)

**Four findings, four durable lessons:**

0. **Meta hyperparameter surface is flat — production is at ceiling**
   (E8). Class-weight × C × L1/L2 are rank-no-op for binary AUC. The
   s6e4 "three axes" recipe is metric-specific (balanced accuracy);
   does not transfer to AUC. Tweaking the LR-meta cannot break
   saturation. Logits-only input is −0.9 bp worse than P+rank+logit.

1. **Pool redundancy is the root cause of K=22+ saturation** (E1).
   Eff_rank ≈3 of 24. Pool surgery candidate identified (E2):
   `cb_slow-wide-bag` is redundant AND mis-calibrated. Future-comp
   reusable diagnostic: SVD+entropy on the OOF logit matrix as a
   Day-1 pool-health check.

2. **DGP has no locally-linear cells in numeric features** (E4).
   Pure-LR base will be a low-AUC diversity contributor (~0.86),
   not a competitive base. Sets honest expectations for A2/A3/A4.

3. **LR-meta is doing signal allocation, not calibration** (E2).
   21/24 bases well-calibrated. So coefficient changes from
   class_weight / penalty tweaks (E8 pending) reflect signal re-
   weighting, not implicit Platt fixes.

**Next moves under the new criterion:**
- Arc B (DGP archaeology): E5 bootstrap coefficient stability,
  E6 LR-residual-vs-feature-pair interaction map, E9 forward-
  selection trace. These produce the *FE shopping list* — concrete
  feature engineering items to add to the GBDT pool.
- Arc C (new-base diversity): A2 Bagged-LR / A3 Random-Subspace LR
  / A4 per-segment LR specialists, all framed as diagnostic
  experiments (coefficient distributions inform DGP) that *also*
  contribute as new bases.

**Reusable artifacts at Arc A end:**
- `scripts/lr_diag_e{1,2,4,8}_*.py` — drop-in for any tabular comp
- 4 JSON results for the local audit trail
- This audit note as a reusable diagnostic template
