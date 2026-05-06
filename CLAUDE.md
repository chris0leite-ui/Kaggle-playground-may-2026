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
day: 10                           # 2026-05-11 / Day-10: d9h+d9i BOTH LB 0.95034 (+3bp NEW PRIMARY); FM-class OOF→LB miscalibrated
lb_best_today: 0.95435            # leader; not refreshed
our_lb_best: 0.95034              # d9h_K22_add_aug12 / d9i_S1_K21_swap_aug2way (TIED); gap -1.7bp NARROWED from -2.4
submissions_used_today: 5         # d9b TIE; d9c +3; d9f +2; d9h +3 (300× OOF upside); d9i +3 (OOF predicted -0.19, actual +3)
submissions_used_total: 19
saturation_count: 0               # FM model class transferred LB; Day-10 BREAKTHROUGH; Sd pred +0.53 actual +3.0
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
  - simple_math_rule_residual_pool  # d9 -- 9 closed-form / Bayesian rule_residuals; ALL FAIL min-meta vs PRIMARY
  - hash_lr_3way_baseline           # d9 R14 -- sparse-LR 3-way interactions; std 0.794, ρ=0.444 most-diverse
  - hash_lr_strength_ladder         # d9b R14 L0-L5 -- L2/L3/L4 PASS at +0.01bp; L4 K=20 swap LB 0.95025 TIE
  - factorization_machine_cpu       # d9c FM -- std 0.921, ρ=0.899, min-meta +0.18bp PASS; K=20 swap LB 0.95029 (+3bp)
  - hash_lr_3way_strength_ladder    # d9b R14 L0-L5 -- L2/L3/L4 PASS; K=20 swap+L4 LB 0.95025 TIE
  - factorization_machine_partition # d9f FM_A driver-dynamics + FM_B race-context -- K=21 swap LB 0.95031 (+2bp NEW PRIMARY)
  - groupkf_stack_rebuild_audit     # d10b/c -- FM-class lift +2.01bp under Race-only GKF vs +0.87bp Strat (2.3× AMPLIFIED); FM_B is #1 L1 component under GKF; PRIMARY private-LB robust
plateau_days: 0
gate_status: cleared              # d9h K=22 add + d9i S1 K=21 swap aug 2-way BOTH LB 0.95034 (+3bp each); gap -1.7bp NARROWED from -2.4
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
| **d6_k18_multi_rule** | **0.95065** | n/a | **0.95026** | **PRIMARY**; +2.1bp LB; gap −3.9bp (NARROWED from −5.2); 1.3bp upside on +0.8bp prediction |
| d9_R5_weibull_compound | 0.94600 | n/a | n/a | d9 -- ρ vs PRIMARY 0.943; min-meta -0.09bp FAIL |
| d9_R6_next_compound | 0.94443 | n/a | n/a | d9 -- ρ 0.908 (P5 1-step lookup); min-meta -0.12bp FAIL |
| d9_R7_prev_compound | 0.94481 | n/a | n/a | d9 -- ρ 0.914; min-meta -0.10bp FAIL |
| d9_R10_driver_eb | 0.94463 | n/a | n/a | d9 -- ρ 0.912 Beta-Binom; min-meta -0.10bp FAIL |
| d9_R14_hash_lr_3way (L0) | 0.79377 | n/a | n/a | d9 -- ρ=0.444 most diverse; min-meta -0.02bp FAIL by hair |
| d9b_R14_L2 (binned numerics) | 0.91449 | n/a | n/a | d9b -- ρ 0.874; min-meta +0.01bp PASS; sweet spot |
| d9b_R14_L3 (+ Compound × num) | 0.91626 | n/a | n/a | d9b -- ρ 0.875; min-meta +0.01bp PASS; best ladder rung |
| d9b_R14_L4 (+ Driver × num) | 0.91369 | n/a | n/a | d9b -- ρ 0.869; min-meta +0.01bp PASS; K=20 swap chosen |
| d9b_k20_swap_l4 | 0.95067 | n/a | **0.95025** | d9b SUBMITTED -- pred +0.19bp, actual −0.01bp TIE (LB quantization) |
| **d9c_FM (Factorization Machine)** | **0.92069** | n/a | n/a | **d9c -- ρ 0.899, min-meta +0.18bp PASS, 18× R14 lift; new model class** |
| d9c_Sd_K20_swap_FM | 0.95070 | n/a | 0.95029 | hedge; +3bp LB (5.7× upside on +0.53bp pred); demoted by d9f |
| d9f_FM_A_driver_dynamics | 0.82505 | n/a | n/a | d9f -- D/C/S/T_q5; ρ vs PRIMARY 0.487 (most-diverse since R14) |
| d9f_FM_B_race_context | 0.88438 | n/a | n/a | d9f -- R/Y/Rp_q5/P_q5; ρ 0.861; min-meta +0.04bp PASS |
| d9f_K21_swap_partA_partB | 0.95073 | n/a | 0.95031 | demoted by d9h/d9i; was PRIMARY |
| d9h_FM_aug12 (12-field unified) | 0.92540 | n/a | n/a | strongest single FM ever (+4.7bp std OOF over d9c FM); ρ=0.917 vs d9f |
| **d9h_K22_add_aug12** | **0.95073** | n/a | **0.95034** | **NEW PRIMARY (TIED)**; +3bp LB (300× upside on +0.01bp pred) |
| d9i_FM_A_aug (D/C/S/T/Cd/Ld) | 0.88123 | n/a | n/a | aug FM_A; ρ vs d9f PRIMARY 0.720 |
| d9i_FM_B_aug (R/Y/Rp/P/Nx/Pv) | 0.88561 | n/a | n/a | aug FM_B; ρ 0.863 |
| **d9i_S1_K21_swap_aug2way** | **0.95071** | n/a | **0.95034** | **NEW PRIMARY (TIED)**; +3bp LB; OOF predicted -0.19bp (regression!), actual +3bp lift; OOF direction-flipped |
| d10b_K13_baseline (Strat / GKF-Race) | 0.95043 | 0.92744 | n/a | 13 GBDT/baseline; gap −229.92bp (leakage signature) |
| d10b_K15_+FMA+FMB (Strat / GKF-Race) | 0.95052 | 0.92764 | n/a | FM-class lift +0.87bp Strat → **+2.01bp GKF (2.3× amplified)**; FM_B L1 #1 under GKF |
| d10d_leak_corrected_meta | n/a | 0.92764 | n/a | held; G3 FAIL (rare-class flip 0.001); rebalances FM_B L1=6.96 but smooths away GBDT row-extremes; pred-LB 0.95001 |

## Hypothesis board (Day 9 evening)

```
- DONE: d9 simple-math rule_residual cohort — 9 of 10 FALSIFIED at
        min-meta vs PRIMARY (-0.09 to -0.12bp band regardless of
        lookup key / smoothing). 5th confirmation of P10: rule_residual
        family is saturated within PRIMARY's 4-rule cohort.
- DONE: d9 R14 hash_lr_3way — std 0.794 but ρ=0.444 (most-diverse
        single base since RealMLP). Min-meta -0.02bp FAIL by a hair;
        new model-class signal flagged.
- DONE: d9b R14 strength ladder L0-L5 — L2/L3/L4 PASS at +0.01bp;
        L1 (+Race/Year) and L5 (kitchen sink) FAIL. Sweet spot is
        adding 5-quintile bins of TyreLife/RaceProgress/Position to
        the LR. K=20 swap+L4 SUBMITTED at LB 0.95025 (TIE -0.01bp;
        pred +0.19bp; quantization-bounded).
- DONE: d9c Factorization Machine — std OOF 0.92069, ρ=0.899,
        min-meta +0.18bp PASS (18× R14_L3's lift). FM auto-learns
        cross-feature interactions in low-rank space; replaces R14
        ladder entirely. Sd K=20 swap with FM (no R14): pred LB
        +0.53bp, ρ=0.99973. ABOVE +0.5bp slot threshold.
- DONE: d9c Sd K=20 swap + FM SUBMITTED at 18:56 UTC. **LB 0.95029,
        +3bp lift, NEW PRIMARY.** 5.7× upside on +0.53bp prediction.
        Gap narrowed -3.9 → -2.6bp. FM is the first genuinely new
        model class to land LB lift since RealMLP joined M5q (Day-3).
- DONE: d9d FM hparam sweep + 3-seed bag — FLAT. k=8 is sweet spot;
        bagging HURTS K=20 stack (smooths predictions toward shared
        bias, removes routing diversity LR meta consumes).
- DONE: d9e FFM (field-aware FM) — STRICTLY WORSE than FM. 4× more
        params overfits 351k train rows; 8 fields too few for FFM's
        per-field-pair specialization to add value.
- DONE: d9f multi-FM with disjoint feature partitions —
        FM_A driver-dynamics (D/C/S/T_q5) + FM_B race-context
        (R/Y/Rp_q5/P_q5). ρ FM_A vs FM_B = 0.406 (≈ orthogonal).
        K=21 swap (drop d9c FM, add FM_A + FM_B): OOF 0.95073
        (+0.29bp), ρ=0.99965. d9c FM demoted out of L1 top-15 in
        K=22 add — partition replaces unified FM cleanly.
- DONE: d9f K=21 swap SUBMITTED at 20:25 UTC. **LB 0.95031, +2bp lift,
        NEW PRIMARY.** 6.25× upside on +0.32bp prediction (mirrors
        d9c's 5.7× pattern — FM-class LB amplification is real).
        Gap narrowed -2.6 → -2.4bp.
- DONE: d9g 3-way multi-FM (3+2+3 partition) — REGRESSED at all 3
        stack configs (-0.46bp K=22 swap; -0.09bp K=24 add). Per-FM
        too weak; LR meta demotes them. d9f 2-way is partition
        sweet-spot.
- DONE: d9h FM_aug12 unified 12-feat — std OOF 0.92540 (strongest
        single FM ever, +4.7bp over d9c). K=22 add OOF +0.01bp (TIE
        expected). **SUBMITTED 21:19 UTC: LB 0.95034, +3bp lift,
        NEW PRIMARY (300× upside on OOF prediction).** Calibration
        win — challenged "OOF tie → LB tie" assumption.
- DONE: d9i augmented 2-way (D/C/S/T/Cd/Ld + R/Y/Rp/P/Nx/Pv) — std
        FM_A_aug 0.881 (+5.6bp), FM_B_aug 0.886. K=21 swap predicted
        OOF -0.19bp REGRESSION. **SUBMITTED 21:20 UTC: LB 0.95034,
        +3bp lift (TIED with d9h)**. OOF *predicted regression* but
        LB *amplified positive* — 16× direction-flip plus magnitude.
- INSIGHT: FM-class OOF on Strat-fold underestimates LB lift due to
        StratifiedKFold's 80% within-group leakage (P6) inflating
        GBDT-pool OOF. Three consecutive FM-class submits (d9c +3bp,
        d9f +2bp, d9h +3bp, d9i +3bp) confirm. **OOF Δ is a LOWER
        BOUND on LB Δ for FM-class candidates at ρ ≈ 0.9997.**
- DONE: d10 GroupKF audit — under strict (Race,Driver,Year,Stint)
        GKF, FM bases drop only 2.5–54bp vs GBDTs dropping 209–247bp
        under Race-only GKF. FM bases are leakage-robust at the
        standalone level.
- DONE: d10b/c GroupKF stack rebuild — built K=13/K=15 stacks under
        BOTH Strat and Race-only GKF (apples-to-apples). FM-class
        lift: **+0.87bp Strat → +2.01bp GKF (2.3× AMPLIFIED)**.
        L1 inversion: under Strat FMs are mid-pack (FM_B L1=0.138,
        rank 13/15); under GKF FM_B is **#1 dominant** (L1=6.96,
        2× the next base). When LR meta can't piggyback on within-
        group leakage from GBDTs, it routes hard through FM. PRIMARY
        d9f K=21 swap (LB 0.95031) is private-LB robust.
- DONE: d10d leak-corrected LR meta (refit LR on GKF OOFs, apply to
        GKF test preds). G3 rare-class flip ratio 0.001 (1751 rows
        drop out of top-1%, 2 added). FM_B L1=6.96 dominates as
        designed but smooths away GBDT row-specific extremes that
        ARE genuine (i.i.d. test → those rows really do pit). HELD,
        pred-LB 0.95001. Insight: GKF OOFs cannot see test-row-
        specific signals, so they over-credit FM. Bayesian
        hierarchical stacker (Path B) is the correct synthesis.
- LATER: External-data Pirelli pit-window scrape (Tier-2 highest
        absolute EV), EmbMLP CPU (different model class), hazard NN
        (GPU; d9 hazard_nn_stack regressed 315bp — implementation
        matters; main-branch agent's leakfree hazard NN at OOF 0.92013
        confirmed DEAD).
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
- `audit/2026-05-09-d9-math-heuristics.md` — d9 10-approach cohort, all min-meta FAIL vs PRIMARY.
- `audit/2026-05-09-d9b-r14-ladder.md` — d9b R14 ladder L0-L5; K=20 swap+L4 SUBMITTED LB 0.95025 TIE.
- `audit/2026-05-09-d9c-fm.md` — d9c FM passes min-meta +0.18bp; Sd K=20 swap+FM LB 0.95029 (+3bp).
- `audit/2026-05-10-d9d-fm-sweep-bag.md` — FM hparam sweep + bag NULL; bag HURTS stack.
- `audit/2026-05-10-d9e-ffm.md` — FFM strictly worse than FM (overfit + redundant).
- `audit/2026-05-10-d9f-multi-fm.md` — multi-FM partition K=21 swap LB 0.95031 (+2bp prior PRIMARY).
- `audit/2026-05-10-d9g-3way-multi-fm.md` — 3-way partition REGRESSED.
- `audit/2026-05-10-d9h-fm-augmented.md` — FM_aug12 standalone strongest; K=22 add LB 0.95034 (+3bp NEW PRIMARY tied).
- `audit/2026-05-10-d9i-augmented-2way.md` — aug 2-way K=21 swap LB 0.95034 (+3bp NEW PRIMARY tied; OOF was -0.19bp regression).
- `audit/2026-05-10-d10-groupkf-audit-fm-real.md` — strict GKF FM bases drop 2.5–54bp vs GBDTs −210bp.
- `audit/2026-05-10-d10b-groupkf-stack-rebuild.md` — FM-class lift +2.01bp GKF vs +0.87bp Strat (2.3× AMPLIFIED); FM_B is #1 L1 under GKF.
- `audit/2026-05-10-d10d-leak-corrected-meta.md` — leak-corrected LR meta gate-FAILs (G3 flip ratio 0.001) but informative; Bayesian hierarchical is correct synthesis.
- `audit/friction.md` — friction one-liners.
