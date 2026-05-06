# P1 single-model results — Day-16 PM (in progress)

Branch: `claude/read-kaggle-handover-rsi2Q`. Plan doc:
`audit/2026-05-06-p1-single-model-plan.md`. ISSUES leaf: 8a.

## TL;DR (filled at completion)

> _to be filled after feA_te completes._

## Phase 1 — replicate Rozen single LGBM (~118 features + 6 CV TE)

Recipe ported from `external/kernels/romanrozen/f1-pit-driver-race-year-encoding-0-95354.ipynb`:
- `make_features_A`: tyre/compound/race-progress/lag-rolling/within-stint
  + 3 combo categoricals (Race_Compound, Race_Year, Driver_Compound).
- `cv_target_encode`: 6 high-card combos with smoothing 15-30
  (Driver×Race×Year, Driver×Race, Race×Compound, Driver×Compound,
  Race×Year, Driver×Race×Compound).
- LGBM hparams (Rozen): lr=0.025, num_leaves=255, min_child_samples=25,
  feature_fraction=0.65, bagging_fraction=0.85, lambda_l1=1.2,
  lambda_l2=2.5, max_depth=10, path_smooth=0.1, n_estimators=6000.

| Variant | Hparams | OOF AUC | ρ vs PRIMARY | pred LB Δ | Wall |
|---|---|---:|---:|---:|---:|
| raw_only (3/5 folds) | Rozen | ~0.948 (partial) | — | — | ~70s/fold |
| feA_te (5 folds) | Rozen | _pending_ | _pending_ | _pending_ | _pending_ |
| feA_te_orig | Rozen | _planned_ | _planned_ | _planned_ | _planned_ |
| p1_single_cb feA_te | Rozen CB | _planned_ | _planned_ | _planned_ | _planned_ |

Reference (Rozen, T4×2 GPU):
- LGB: 0.95241 OOF
- XGB: 0.95232
- CB: 0.95127
- RealMLP-A: 0.95260
- RealMLP-B: 0.95259
- Stack v8: 0.95357
- Final blend: 0.95354 LB (LB **0.95372** in makimakiai re-run with extras).

## ρ inventory (PRIMARY vs external submissions)

```
ρ(PRIMARY 0.95059, makimakiai_v8_solo)        = 0.96952
ρ(PRIMARY 0.95059, makimakiai_blend LB95372)  = 0.98146  ← +313bp at ρ<0.999
ρ(PRIMARY 0.95059, pavlo baseline 0.942)      = 0.95470
ρ(PRIMARY 0.95059, gkanamoto tabM)            = 0.97860
ρ(makimakiai_v8, makimakiai_blend)            = 0.98649
```

**Implication**: makimakiai_blend's +313 bp at ρ=0.981 means the leader
pipeline captures genuine signal we miss; this is consistent with the
single-model recipe being responsible for ~+150 bp standalone, and
blending adding another ~+150 bp.

## Phase 2 — single CatBoost (8b, conditional on Phase 1)

> _to be filled if Phase 1 lands ≥ +30 bp standalone._

## Phase 3 — RealMLP via Kaggle GPU (8c)

> _deferred unless Phase 1 + Phase 2 land._

## Phase 4 — submit decision (deferred)

Decision rule:
- standalone OOF Δ > +30 bp AND ρ < 0.999 → submit candidate
  (R7-eligible if flip count > 200; PI sign-off required).
- standalone OOF Δ in [+5, +30] AND ρ < 0.995 → consider stack-add
  probe (8d) instead of direct submit.
- standalone OOF Δ < +5 → falsify P1 (single model can't close gap).

Submit slot: PRIMARY-replacement candidate. R5 final-3-day window
applies.

## Friction tags (candidate, to confirm)

- **single-model-recipe-OOF-LB-transferable** — if our single LGBM
  OOF lifts cleanly to LB, P1 thesis CONFIRMED. Adds to friction
  tag library.
- **target-encoded-driver-race-year-is-the-magic-feature** — if
  ablation shows TE feats account for most of the lift.
- **rozen-recipe-portable-across-ps-comp-instances** — for general
  Playground pattern reuse.

## Pointers

- `audit/2026-05-06-p1-single-model-plan.md` — 7-step plan.
- `external/kernels/romanrozen/...` — source notebook.
- `external/makimakiai_idsafe/submission.csv` — LB 0.95372 ceiling probe.
- `scripts/p1_features.py`, `scripts/p1_single_lgbm.py`, `scripts/p1_single_cb.py`.
- `scripts/p1_gate_all.py`, `scripts/p1_post.py` — post-run gate harnesses.
