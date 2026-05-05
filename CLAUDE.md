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
16. **New-candidate pre-flight (5-question check).** Before committing
    CPU/GPU compute on any new base or meta variant, answer:
    (1) Is the underlying mechanism in `mechanism_families_explored`?
    (2) Does the candidate fall in {meta-only, rule_residual-on-raw,
    GBDT-on-binary-target, formulation-already-in-pool}? If yes,
    rank-lock-vulnerable. (3) Predict standalone OOF (cite precedent).
    (4) Predict ρ vs PRIMARY (cite closest base). (5) At that ρ,
    cite the closest gate-PASS/FAIL precedent. If 1–5 don't return
    a coherent answer, downgrade EV midpoint by 0.3× before ranking.
    Origin: `tag: menu-overcrediting-redundant-mechanism` (Day-8
    falsified T1.5/T1.3/T1.2 all of which passed research-agent EV
    ranking but failed the 5-question check retroactively).

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
day: 6                            # 2026-05-07 / Day-6: F1.2 multi-rule LANDED at LB 0.95026 (+2.1bp)
lb_best_today: 0.95435            # leader; not refreshed
our_lb_best: 0.95026              # d6_k18_multi_rule (M5q + 4 rule-residual bases) — NEW PRIMARY
submissions_used_today: 1         # 1/10 today; d6_k18_multi_rule cleared LB at 0.95026
submissions_used_total: 14
saturation_count: 0               # F1.2 multi-rule cleared the +0.5bp predicted, +2.1bp actual (+1.3bp upside)
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
  - aux_feature_gbdt_meta           # d6 F5 -- +0.12bp over no-aux LGBM; FALSIFIED
  - 2base_recursive_blend           # d6 B -- 4 variants; tie or regress; FALSIFIED
  - rule_residual_l1_base           # d6 C/F1.1 -- residual GBDT on rule_proba; min-meta PASS
  - multi_rule_residual_k18         # d6 F1.2 -- 4 rules; LB 0.95026 (+2.1bp PRIMARY)
plateau_days: 0
gate_status: cleared              # F1.2 multi-rule LB +2.1bp; gap narrowed -5.2 -> -3.9bp
headroom_to_top5pct: 0.00319      # 0.95345 − 0.95026 = 31.9bp
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
| **d5_partial_pseudo_m5q (K=14)** | **0.95082** | n/a | **0.94963** | **slot-1 −4.2bp LB**; gap WIDENED −5.2→−12bp; pseudo over-amp falsified |
| d6_aux_meta_with_aux | 0.95049 | n/a | n/a | F5 falsified; +0.12bp over no-aux; held |
| d6_2base_v1_lr_expand | 0.95055 | n/a | n/a | Move B falsified; ρ=0.99996 (tie); held |
| d6_rule_residual (standalone) | 0.94593 | n/a | n/a | Δe3 −28bp; ρ vs M5q test 0.92887 (most diverse since RealMLP) |
| d6_k15_rule_residual | 0.95062 | n/a | n/a | held; +0.51bp; ρ=0.99971; superseded by K=18 |
| d6_rule_compound_stint (std) | 0.94604 | n/a | n/a | F1.2 R2; min-meta +0.30bp PASS |
| d6_rule_driver_compound (std) | 0.94457 | n/a | n/a | F1.2 R3; ρ=0.89144 (most diverse); min-meta +0.45bp PASS |
| d6_rule_year_race (std) | 0.94586 | n/a | n/a | F1.2 R4; min-meta +0.37bp PASS |
| **d6_k18_multi_rule** | **0.95065** | n/a | **0.95026** | **NEW PRIMARY**; +2.1bp LB; gap −3.9bp (NARROWED from −5.2); 1.3bp upside on +0.8bp prediction |

## Hypothesis board (Day 6 evening)

```
- DONE: F5 aux-feature GBDT-meta — FALSIFIED (+0.12bp, OOF -0.78bp
        vs M5q). Third rank-lock confirmation that base-pool signal
        is the binding constraint, not meta expressiveness.
- DONE: Move B 2-base [M5q, recursive] — FALSIFIED across 4 variants.
        K=2 LR-expand tie-locks (ρ=0.99996); V2-V4 OOF regression.
        Recursive structurally redundant with M5q (recursive trained
        on m5q_oof_proba feature).
- DONE: F1.1 rule_residual single base — REAL but quantum-bounded.
        Std OOF 0.94593, ρ=0.93 vs M5q test, K=15 +0.51bp, minimal-
        meta PASS. First non-tie minimal-meta lift in 5 days.
- DONE: F1.2 multi-rule strengthening — LANDED at LB 0.95026
        (+2.1bp PRIMARY). 4 rule_residual bases (Compound x TyreLife,
        Compound x Stint, Driver x Compound, Year x Race) all pass
        minimal-meta. K=18 LR-stack: OOF 0.95065 (+0.78bp), ρ=0.99902
        (4x safety margin vs tie threshold). Gap narrowed -5.2→-3.9bp.
- ACTIVE: Move F multi-seed RealMLP bag — kernel realmlp-bag-gpu v1
        running on Kaggle T4 (seeds 123+456). ~6h ETA. Pull when
        complete; rank-average with seed-42; rebuild K=19 stack
        (M5q_bag + 4 rules). Predicted +1-3bp on top of K=18 anchor.
- NEXT: F1.3 classifier-residual variant (sample_weight inverse to
        rule confidence) + F1.4 rule_proba as meta-feature. Both
        cheap CPU; build alongside RealMLP bag pull.
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
- `audit/2026-05-07-d6-critic-loop.md` — Rule 14 audit; 5 untried mechanisms.
- `audit/2026-05-07-d6-f5-aux-meta-result.md` — F5 falsified.
- `audit/2026-05-07-d6-move-b-2base-recursive.md` — Move B falsified.
- `audit/2026-05-07-d6-move-c-rule-residual.md` — F1.1 single rule.
- `audit/2026-05-07-d6-f1-2-multi-rule.md` — F1.2 K=18 LB-landed +2.1bp.
- `audit/friction.md` — friction one-liners.
