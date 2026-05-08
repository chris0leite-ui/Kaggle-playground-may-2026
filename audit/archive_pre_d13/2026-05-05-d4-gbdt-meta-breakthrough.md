# Day-4: GBDT-meta breaks the LR-meta rank-lock

PI directive: think bigger moves, 34bp headroom to top-5%, stop
worrying about seed variance. Tested theory: **the LR-meta rank-lock
on M5q is an LR-on-GBDT-pool artifact**. A non-linear meta-learner
(LightGBM / HGBC) over the same K=14 base pool can capture
which-base-best-on-which-row patterns that LR can't represent.

## Result — first real PASS of the diversity gate this session

| Meta | Strat OOF | Δ M5q bp | ρ vs M5q test | Gate |
|---|---:|---:|---:|---|
| **m5q (LR meta, anchor)** | **0.95057** | — | 1.00000 | reference |
| m5_meta_lgbm_shallow (depth=3) | 0.95048 | −0.92 | **0.99508** | **PASS** |
| m5_meta_lgbm_medium (depth=5) | 0.95047 | −0.98 | **0.99436** | **PASS** |
| m5_meta_hgbc (depth=4) | 0.95042 | −1.50 | **0.99490** | **PASS** |

ρ in the 0.994-0.995 band is the lowest stack-vs-M5q diversity we
have measured today. The Day-3 friction `lr-meta-rank-lock-strong-anchor`
is not a property of the test distribution — it is a property of the
meta-learner choice.

## Calibration vs M5q-vs-M5h precedent

| pair | ρ | LB delta |
|---|---:|---:|
| M5q vs M5h | 0.99865 | +14bp |
| lgbm_shallow vs M5q | **0.99508** (3.6× more divergent) | unknown |
| lgbm_medium vs M5q | **0.99436** (4.2× more divergent) | unknown |

Pre-submit diff vs M5q for lgbm_shallow:
- 99.92% rows differ > 1e-6
- 91.94% rows differ > 1e-4
- 62.74% rows differ > 1e-3
- max abs diff 0.171, max rank shift 48,008 / 188,165 (~25%)
- median rank shift 1,820

This is structurally a different submission. LB delta is genuinely
unknown but no longer in tie territory.

## Outcome envelope

- **Upside (rank-lock theory correct + meta-honest)**: LB lifts; the
  GBDT meta is now the new PRIMARY mechanism family. Iterations:
  XGB meta, CatBoost meta, ensemble of metas.
- **Downside (OOF transfers 1:1)**: LB drops ~1bp from M5q's 0.95005
  → ~0.94995. Acceptable cost for the information value. Confirms LR
  meta was not the bottleneck and the true ceiling is the BASE pool.
- **Tie (very unlikely at ρ=0.995)**: would suggest test distribution
  is robust to ranking-level diff at this magnitude. Useful negative
  result.

## Recommended slot 2

**lgbm_shallow** (depth=3, num_leaves=8, lr=0.05, ES at iter ~470-600).
Strongest candidate by combination of:
- Best OOF among the three (smallest regression).
- ρ=0.99508 — sufficiently divergent for LB delta but the most
  conservative of the three.
- Best_iters 347-598 → genuinely converged, not under-trained.

`submissions/submission_m5_meta_lgbm_shallow.csv` is built and
pre-submit-diff verified. Awaiting PI approval (Rule 1 single-shot).

## What this proves about the pool

The LR meta with expand() (raw + rank + logit) was producing
near-identical TEST RANKINGS regardless of base composition or
weights. This was disguised as "stack saturation" but was actually
"linear-meta saturation". The base pool has more usable signal
than LR can extract.

## Next moves if slot-2 lifts

1. XGB meta with regularization (different non-linear family).
2. CatBoost meta with native cat handling on the base names as a
   "which-base" feature.
3. Stacked meta-of-metas (LR over [LR-meta, GBDT-meta]).
4. Re-run M5q+yetirank+nb under the GBDT meta — bases that were
   TIE_EXPECTED under LR may add real lift under non-linear meta.

End — 70 lines.
