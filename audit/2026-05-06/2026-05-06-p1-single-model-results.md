# P1 single-model thesis — final results (Day-17 AM)

Branch: `claude/read-kaggle-handover-rsi2Q`. Plan doc:
`audit/2026-05-06-p1-single-model-plan.md`. ISSUES leaf: 8a.

## TL;DR

**P1 thesis CONCLUSIVELY FALSIFIED on s6e5.** Under strict OOF
discipline (fold-safe label-conditional aggregates, fold-safe CV TE),
the best single-LGBM with kitchen-sink Rozen-style FE achieves
**OOF 0.94563**. Our K=22 + Path-B hier-meta PRIMARY (LB 0.95059) is
+52 bp ahead in honest terms. Stacking is necessary for our LB.

**4 LB submits this branch** (+ 1 from another branch):

| Submission | OOF | LB | Gap | Status |
|---|---:|---:|---:|---|
| K22_add_p1 LR-meta v1 | 0.95404 | 0.94933 | −471 bp | leaky |
| p1_single v1 | 0.94970 | **0.94107** | **−863 bp** | catastrophic |
| K2 LR(PRIMARY, v2) | 0.95398 | 0.94996 | −402 bp | 2-level + leaky |
| (other branch) d16_continuous_only | 0.95121 | **0.95089** | −3 bp | clean Path-B |
| **PRIMARY** (unchanged) | 0.95090 | **0.95059** | −3 bp | clean |

PRIMARY remains best validated. d16_continuous_only from another
branch is the +30 bp candidate via clean Path-B amplification.

## Phases run

### Phase 1 — v1 (leaky stint-count cluster)
- `make_features_A` v1: 50 engineered features + 6 CV TE; included
  `stint_size_far`, `stint_pct`, count-based `pit_imminent`/`pit_in_5`
  (per-split-count `groupby.cumcount()` / `transform('count')`).
- OOF 0.94970, K=22 LR-meta-add OOF 0.95404 (+33 bp).
- Submitted: K22_add LB **0.94933**, single LB **0.94107**.
- Diagnosis: train/test feature distribution shift via per-split
  counts (same physical stint, train sees count=7, test sees count=3).

### Phase 2 — v2 (FS_A leak)
- Removed leaky stint-count cluster; added Rozen FS_A merge aggregates
  (`race_avg_pit_lap`, `compound_avg_life`, etc.) + 1950-2022 priors.
- OOF 0.95128 (+158 bp vs v1, +38 bp over PRIMARY OOF — too good).
- Submitted: K=2 LR(PRIMARY, v2) LB **0.94996** (−63 bp vs PRIMARY).
- Diagnosis (via 80/20 holdout test, no slot burned):
  holdout AUC 0.94637 vs OOF 0.95128 = **−491 bp gap**. FS_A
  aggregates fit on full train (with PitNextLap labels) leaked val
  labels into val features.

### Phase 3 — v3 (fold-safe FS_A)
- Refactored: `make_features_static` (label-independent only) +
  `fit_fs_a` (per-fold, ti rows only) + `apply_fs_a` (merge +
  derivatives). 5-fold CV with ti-only FS_A and inner-CV TE per fold.
- **OOF 0.94563** (fold-std 0.00049, total wall 4 min — vs v2's 31
  min because LGBM early-stops at 450-715 iters instead of 5400-6000).
- Per-fold: 0.94654 / 0.94582 / 0.94544 / 0.94522 / 0.94525.
- Holdout AUC 0.94637 (with full-80% FS_A) — matches v3 OOF tightly,
  **confirming fold-safety**.

## Gate v3 vs PRIMARY

| Metric | Value |
|---|---:|
| Standalone OOF | 0.94563 |
| Δ OOF vs PRIMARY | −52.69 bp |
| ρ vs PRIMARY (TEST) | 0.953 (very diverse) |
| predicted LB band | LB ~0.94482 |
| G3 flips: +→− | 1415 |
| G3 flips: −→+ | 64 |
| K=2 LR-meta(PRIMARY, v3) OOF | 0.95124 (Δ +3.40 bp) |

**Genuine incremental value of v3 as 23rd base** = +3.40 bp at
ρ=0.953. Compare with v2's leaky +30.79 bp — the leak amplification
was 90% of the apparent K=2 lift. Not worth a slot for confident
LB submit.

## Lessons captured

Skill file `improvements.md`:
- G13 single-model-first / kitchen-sink FE before stacking
- G14 family falsification needs ≥3 variants
- G15 framework is scaffolding, not authorship
- **G16 fold-safe label-conditional aggregates** (NEW, Day-17)
- **G17 transductive features need AV check** (NEW, Day-17 PI lesson)
- pre-baseline gate items 8-11 (public-notebook scan, TE inventory,
  physics features, single-model OOF target)
- 80/20 holdout diagnostic mandatory before new-FE-family LB submit

Local `CLAUDE.md` R20-R23 updated.

`audit/friction.md` 2026-05-07 section: 4 new tags
- `target-construction-layer-leakage` (re-encountered at FS_A level)
- `2-level-stacking-with-meta-derivative` (re-encountered)
- `cv-te-stacking-base-leakage` (LR-meta over-credits CV-TE bases)
- `transductive-features-need-AV-check` (PI Day-17)
- `P1-single-model-thesis-falsified-on-s6e5`

## Files

- `scripts/p1_features.py` v3 — `make_features_static` +
  `fit_fs_a` + `apply_fs_a`; legacy `make_features_A` kept and flagged.
- `scripts/p1_single_lgbm_v3.py` — fold-safe trainer.
- `scripts/p1_holdout.py` — 80/20 honest holdout diagnostic.
- `scripts/p1_post.py`, `scripts/p1_gate_all.py` — gate harnesses.
- `scripts/artifacts/oof_p1_single_lgbm_v3_feA_te_strat.npy` (+test).
- `scripts/artifacts/p1_holdout_results.json` — holdout AUC 0.94637.
- `submissions/submission_K22_add_p1_feA_te.csv` — leaky v1 stack-add (LB 0.94933).
- `submissions/submission_p1_single_lgbm_feA_te.csv` — leaky v1 single (LB 0.94107).
- `submissions/submission_K2_PRIM_v2.csv` — leaky 2-level K=2 (LB 0.94996).
- `submissions/submission_p1_single_lgbm_v3_feA_te.csv` — fold-safe v3 (NOT submitted; predicted LB regression).
- `external/kernels/{romanrozen,...}/` — 8 reference notebooks.
- `external/{aadigupta_orig,f1_official_1950_2022,weather_woodshole}/` — external datasets.

## What's still open

- **d16_continuous_only_tau20000** (other branch, LB 0.95089) — clean
  +30 bp candidate via Path-B hier-meta on a fold-safe base. Future
  PRIMARY-replacement candidate.
- **Single-model HOLDOUT calibration as standard probe** — every
  new FE family should run the 80/20 holdout test before LB submit.
- **Rozen 0.95241 standalone may be similarly inflated** by FS_A leak
  in his pipeline. The blend LB 0.95354 is dominated by his external
  blend partners (5-source ensemble), not the single-LGB component.
