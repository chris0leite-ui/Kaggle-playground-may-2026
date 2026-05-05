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
12. **Spend the full 10/day submission budget.** Submissions are
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
day: 5                            # 2026-05-06 / Day-5 morning; Path B Phase 1+2 PASS, slot-1 candidate held
lb_best_today: 0.95435            # leader; not refreshed
our_lb_best: 0.95005              # M5q (M5h + RealMLP-TD, K=14) — PRIMARY pending Path B slot
submissions_used_today: 0         # 0/10 today; Path B partial-pseudo K=14 candidate ready
submissions_used_total: 12
saturation_count: 0               # Path B Phase 2 broke the meta-add ceiling; +2.54bp OOF on partial-pseudo K=14
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
  - unified_te_2way_keys            # d3a -- +2.2bp Strat std-alone, +0.1bp stacked (null)
  - sequence_fe_race_driver         # d3b -- +18bp Strat std-alone, +0.2bp stacked (null)
  - tier_break_l1_prune             # M5h2 v1 -- drop a_horizon, K=12, LB 0.94991 (tied; gap unchanged)
  - catboost_yetirank_pairwise      # d4 -- 0.90508 std, ρ=0.666 vs M5q (most diverse), TIE_EXPECTED
  - gaussian_naive_bayes_mixed      # d4 -- 0.87984 std, ρ=0.853 vs M5q, TIE_EXPECTED stack
  - gbdt_meta_lr_alternative        # d4 slot 2 -- LGBM/HGBC meta over M5q pool; LB 0.95001 (-4bp)
  - recursive_gbdt_m5q_feature      # d5 path-c -- 0.94994 std (+92bp); K=15 stacks NULL (3rd rank-lock)
  - gbdt_meta_k15_recursive         # d5 -- LGBM/HGBC meta over K=15 NULL (-1bp vs d4 K=14)
  - tabnet_smoke_default_config     # d5 -- 0.93532 fold0, FAIL gate; under-trained, parked
  - pseudo_label_e3_mvp             # d5 path-b phase1 -- +4.1bp e3, ρ=0.996 PASS both gates
  - pseudo_label_5_base_phase2      # d5 path-b phase2 -- 5 fast bases all lift +2-19bp
  - partial_pseudo_m5q_k14          # d5 -- 6 pseudo + 8 orig; OOF 0.95082 (+2.54bp); ρ=0.99836 REAL_DELTA
plateau_days: 0
gate_status: cleared              # Path B both gates PASS (e3 +4.1bp/ρ0.996; partial-pseudo +2.5bp/ρ0.998)
headroom_to_top5pct: 0.00340      # 0.95345 − 0.95005 = 34.0bp
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
| **m5h (L1coef top-13)** | **0.95043** | **0.93087** | **0.94991** | **CURRENT PRIMARY**; gap −5.2bp; drop m3+m4 dead weight |
| d3a_te_unified | 0.93692 | 0.91284 | n/a | +2.2bp Strat vs d2a; std-alone redundant w/ d2a |
| m5i (M5h+d3a, 14) | 0.95043 | 0.93096 | n/a | d3a L1=0.079 last; tie M5h Strat |
| m5j (d3a swaps d2a, 13) | 0.95044 | 0.93092 | n/a | d3a L1=1.065 (3rd); +0.1bp Strat tie |
| d3b_seqfe | 0.94254 | 0.92136 | n/a | +18bp Strat over baseline; FAIL gate by 35bp |
| m5k (M5h+d3a+d3b, 15) | 0.95045 | 0.93102 | n/a | d3b L1=0.316 (mid-tier); +0.2bp Strat tie |
| m5h2_v1 (drop a_horizon, 12) | 0.95044 | n/a | **0.94991** | **TIED M5h LB**; gap-vs-pool-size hypothesis falsified for 13→12 |
| m5j (d3a swaps d2a, 13) | 0.95044 | n/a | **0.94991** | **TIED M5h LB**; TE-key swap is LB-neutral (quantization-limit) |
| m5p (minimal+LR-FE+EBM, 6) | 0.94839 | n/a | **0.94754** | -237bp; orthogonal-mech thesis FAILED |
| m5n_3b (minimal-basis, 4) | 0.94808 | n/a | **0.94700** | -291bp; minimal-basis thesis FAILED — clones earn slot |
| **m5q (M5h + RealMLP, 14)** | **0.95057** | n/a | **0.95005** | **PRIMARY**; +14bp LB; +1.4bp OOF → 10× LB amplification |
| d4_cb_yetirank | 0.90508 | n/a | n/a | std weak; ρ=0.666 vs M5q (most-diverse base); +0.0bp stack TIE |
| d4_nb (mixed Gaussian + TE) | 0.87984 | n/a | n/a | std weak; ρ=0.853 vs M5q; +0.24bp stack TIE |
| m5x (M5q + yetirank, K=15) | 0.95057 | n/a | n/a | held; ρ=0.99966 vs M5q TIE_EXPECTED |
| m5z (M5q + yetirank + nb, K=16) | 0.95060 | n/a | n/a | held; ρ=0.99957 vs M5q TIE_EXPECTED |
| m5_meta_lgbm_shallow (LGBM d=3) | 0.95048 | n/a | **0.95001** | **slot 2**; -4bp LB; meta-switch bounded; ρ=0.995→4bp |
| m5_meta_lgbm_medium (LGBM d=5) | 0.95047 | n/a | n/a | held; ρ=0.99436 vs M5q; OOF -1bp |
| m5_meta_hgbc | 0.95042 | n/a | n/a | held; ρ=0.99490 vs M5q; OOF -1.5bp |
| d5_recursive_m5q (HGBC + M5q feat) | 0.94994 | n/a | n/a | std-alone +92bp baseline; ρ=0.99159 vs M5q |
| d5_M5_K15a (M5q + recursive, LR) | 0.95056 | n/a | n/a | NULL (-0.06bp); rec L1=0.84 but ρ=0.99991 TIE_EXPECTED |
| d5_meta_k15_lgbm_shallow (GBDT meta) | 0.95038 | n/a | n/a | NULL (-1.0bp vs d4 K=14); GBDT-meta ceiling fixed |
| **d5_partial_pseudo_m5q (K=14)** | **0.95082** | n/a | n/a | **slot-1 candidate**; +2.54bp OOF; ρ=0.99836 REAL_DELTA |

## Hypothesis board (Day 5 morning)

```
- DONE: Path B Phase 1+2 cleared both gates. e3_hgbc rebuilt on
        train+pseudo gives +4.1bp OOF, ρ=0.996 vs orig. K=14 partial-
        pseudo M5q (6 pseudo + 8 orig): Strat 0.95082 (+2.54bp);
        ρ=0.99836 vs M5q REAL_DELTA. FIRST non-null Day-5 meta-level
        result. L1 reshuffles away from cb_slow-wide-bag/a_horizon.
- DONE: Recursive GBDT (M5q_oof_proba as feature) — std-alone +92bp
        baseline (best single GBDT to date, 0.94994), but K=15 LR
        stack and K=15 GBDT-meta both NULL (-0.06bp / -1.0bp). 3rd
        confirmation lr-meta-rank-lock-strong-anchor.
- DONE: TabNet smoke at default config (n_d=32, cat_emb_dim=4)
        FAIL gate: fold-0 0.93532, model under-trained at 120 epochs.
        Park; tuned retry only after Path B succeeds or fails.
- ACTIVE: Path B Phase 3 — CatBoost CPU rebuilds (4 bases, ~1-2h),
        d2a_te TE-aware rebuild (~10min), RealMLP Kaggle GPU
        rebuild (~6h overnight). Decision rule cleared: partial-
        pseudo OOF > M5q+1bp → expand. Compounding lift expected.
- LATER: NN-family multiplication (RealMLP seed bag, FT-Transformer)
        only after Path B's full ceiling is measured.
```

## Pointers

- `HANDOVER.md` — next-session brief (Rule 15).
- `comp-context.md` — settled-once facts.
- `audit/2026-05-04-strategy-critique.md` — Rule 14 origin.
- `audit/2026-05-04-catboost-research.md` — CatBoost lever map.
- `audit/2026-05-04-m5h-l1coef-prune.md` — Day-3 submit candidate.
- `audit/2026-05-04-d3a-te-unified.md` — Step 1 result + M5i/M5j.
- `audit/2026-05-04-d3b-seqfe.md` — Step 2 result + M5k.
- `audit/2026-05-05-d4-yetirank-nb-results.md` — Day-4 base-add probes.
- `audit/2026-05-05-d4-gbdt-meta-breakthrough.md` — Day-4 slot-2 envelope.
- `audit/2026-05-05-nn-stack-priorities.md` — bigger-move ordering.
- `audit/friction.md` — friction one-liners.
