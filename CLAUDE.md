# CLAUDE.md — playground-series-s6e5

Running log + ⚠️ rules. Cap ≤50k tokens; archive when bloated.

## ⚠️ Reference branch

Truth lives on `origin/main` (now the GitHub default). At session start:
`git fetch origin && git log --oneline HEAD..origin/main` — if non-empty,
ff-merge before reading state below.

## ⚠️ Top-level rules (inherited from kaggle-comp framework, LOAD-BEARING)

1. **Ask-first / no-loop on submissions.** Every `kaggle competitions
   submit` is single-shot, explicitly approved. No retry / `until` /
   `while` / `for`. Polling monitors fine; writing monitors forbidden.
2. **Smoke + 1-fold time-probe + 1h GPU cap.** Smoke at 1 fold / 50k
   rows. If 5-fold projection ≥1h, shrink. If kernel preprocesses ≥30
   min with no fold output, kill it.
3. **4-gate leakage filter pre-LB-probe.** G1 standalone OOF clears
   anchor; G2 blend lift; G3 net rare-class-flip ratio ≥0.5; G4
   direction asymmetry. Plus minimal-input-meta sanity check.
4. **NEVER-GIVE-UP / saturation-is-bounded / never-lock-and-stop.**
   After every null, brainstorm 3 untried mechanisms. Locking is for
   the final 3-day window only.
5. **Keep CLAUDE.md fresh / archive-on-bloat.** Cap ≤50k tokens.
6. **Heuristics before heavy compute.** Closed-form rule / threshold
   / hand-coded baseline before Optuna / GPU / 5-fold-bagging.
7. **Research before saturation.** At 3 nulls / 5 sat at same LB / 2
   days no lift: web search + 2 prior-comp writeups + 5 untried
   mechanisms with citations BEFORE declaring ceiling.
8. **Settled-once facts** live in `comp-context.md`. Never re-ask.
9. **File-size cap ≤150 lines** for any committed doc.
10. **Pull-style updates.** No proactive minute-level chatter; on PI
    pull, 1-2 sentences with the latest fact.
11. **Model routing.** Haiku read-only; Sonnet default; Opus hard.
    10/day budget.
12. **Spend the full 5/day submission budget.** Submissions are
    calibration probes — measured OOF→LB gap per mechanism family is
    load-bearing data. Each submit single-shot + PI-approved (Rule 1).
13. **Kaggle GPU is part of compute budget.** Local CPU-only; Kaggle
    notebooks (P100/T4×2) are the GPU path. Port NN / deep-CB-depth≥8
    5-fold / any 5-fold > 1h-CPU before declaring "not cost-justified".
14. **Strategy-critic-loop fires automatically.** End-of-day audit, on
    OOF→LB gap drift ≥2bp on consecutive submits, before adding a new
    mechanism family, at 50% comp checkpoint, or at any plateau (before
    Research-loop). Output: `audit/YYYY-MM-DD-strategy-critique.md`.
15. **Handover protocol.** PI says **"handover"** → read `HANDOVER.md`
    and proceed per its instructions (skip the usual read-order;
    HANDOVER.md is the latest synthesis). PI says **"prepare handover"**
    → update `HANDOVER.md` with the next-session brief.

## ⚠️ Defaults baked in from prior-comp postmortem

- **R1 — Two-anchor OOF.** *s6e5: GroupKF dropped Day-3+ (U3 confirms
  i.i.d. test → Strat is LB proxy, gap +3.8bp).*
- **R2 — Final selection along public-LB axis.** PRIMARY = best public.
  HEDGE = best OOF that *regressed ≤30bp on public*.
- **R7 — Override-mechanism rules.** Flip count <200 → HEDGE only;
  >200 needs explicit PI sign-off.
- **R5 — Final OOF-best regression probe.** In final 3-day window,
  mandatory probe of OOF-best candidate rejected for public regression.
- **R8 — End-of-comp**: log final percentile to
  `~/.claude/skills/kaggle-comp/improvements.md`.

## Current state (Bookkeeper updates daily)

```yaml
day: 3                            # 2026-05-04 / Day-2 5/10 used; Day-3 fresh
lb_best_today: 0.95435            # leader (still); not refreshed
our_lb_best: 0.94991              # M5h L1coef-pruned 13-base stack, +28bp vs M5d
submissions_used_today: 6         # Day-2 5/10 + M5h Day-3-prep = 6/10
submissions_used_total: 6
saturation_count: 1
mechanism_families_explored:
  - baseline_lgbm_raw_features
  - oof_target_encoding
  - xgb_native_categorical
  - catboost_native_categorical
  - relative_state_fe
  - lr_meta_stacker_3view
  - dirichlet_random_search
  - hgbc_label_encoded_driver       # E3 -- BEST single-model pre-CB
  - row_subsample_catboost          # E1
  - l1_meta_sweep                   # E2 -- null
  - realmlp_cpu_singlefold          # E4 -- not pursued
  - lr_meta_stacker_expanded        # M5b -- LB 0.94891
  - reformulation_lgbm              # M5c -- A/B horizon-shift, laps-until-pit
  - hgbc_beta_variants              # M5d -- f1 deep + f2 shallow, LB 0.94963
  - catboost_year_in_cat_cols       # cb_year-cat -- +60bp Strat over base
  - catboost_lossguide_grow_policy  # cb_lossguide -- BEST CB on GroupKF
  - catboost_gpu_multi_seed_bag     # cb_slow-wide-bag -- BEST CB on Strat
  - corr_pool_prune                 # M5g (ρ≥0.97) -- TOO aggressive
  - l1coef_pool_prune               # M5h -- only prune that preserves OOF
plateau_days: 0
gate_status: cleared
headroom_to_top5pct: 0.00354      # 0.95345 − 0.94991 = 35.4bp
```

## Calibration ladder

| Mechanism | Strat OOF | GroupKF OOF | LB | Notes |
|---|---:|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.92059 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| m3_catboost | 0.94612 | 0.91645 | n/a | depth=6, default CTR; Race-overfit |
| m4_relstate | 0.94244 | 0.92195 | n/a | only B1 lifting both anchors |
| e3_hgbc | 0.94876 | 0.92785 | 0.94870 | BEST single pre-CB; gap −0.6bp |
| e5_optuna_lgbm | 0.94736 | 0.92585 | n/a | LGBM Optuna 30 trials |
| f1_hgbc_deep / f2_hgbc_shallow | 0.94870 / 0.94861 | 0.92739 / 0.92711 | n/a | β E3 clones (~99% corr) |
| **cb_year-cat** | **0.94679** | **0.91992** | n/a | Year ∈ CAT_COLS; +60bp/+34bp vs M3 |
| **cb_lossguide** | **0.94697** | **0.92377** | n/a | Lossguide; BEST CB GroupKF (+31.8bp) |
| **cb_slow-wide-bag** | **0.94790** | **0.92322** | n/a | GPU 3-seed bag; BEST CB Strat (+71.5bp) |
| m5b_lr_meta_expanded | 0.94926 | 0.92871 | 0.94891 | gap −3.5bp (anchor) |
| m5d_lr_meta_expanded | 0.95023 | 0.92994 | 0.94963 | D2 PRIMARY; gap −6.0bp (widened) |
| m5e (CB-only 13-base) | 0.95027 | 0.93084 | n/a | held; M5c + 3 CB winners |
| **m5f (combined 15-base)** | **0.95042** | **0.93105** | n/a | held; M5d + M5e new bases |
| m5g (corr ρ≥0.97 prune) | 0.94961 | 0.92915 | n/a | TOO aggressive; 5/15 surv |
| **m5h (L1coef top-13)** | **0.95043** | **0.93087** | **0.94991** | **NEW PRIMARY**; gap −5.2bp; drop m3+m4 dead weight |

## Hypothesis board (Day 3)

```
- DONE: H3 corr-prune sweep -- L1coef-13 (M5h) is the only prune that
        preserves M5f OOF; submit candidate alongside M5f.
- ACTIVE: 4 cheap diagnostics from strategy-critique
        (per-Race/Stint/Year OOF; reliability; agreement matrix; sequence-FE)
- H1: pseudo-labeling guarded by multi-base agreement (≥10/13 of M5h)
- H4: HGBC multi-seed bagging (echo cb_slow-wide-bag pattern)
- 2-way TE (Driver×Race, Driver×Compound) — Day-1 lever, never executed
- Kaggle-GPU port of RealMLP (per yekenot 56-vote notebook)
```

## Pointers

- `HANDOVER.md` — next-session brief (Rule 15).
- `comp-context.md` — settled-once facts.
- `audit/2026-05-04-strategy-critique.md` — Rule 14 origin.
- `audit/2026-05-04-catboost-research.md` — CatBoost lever map.
- `audit/2026-05-04-m5h-l1coef-prune.md` — Day-3 submit candidate.
- `audit/friction.md` — friction one-liners.
