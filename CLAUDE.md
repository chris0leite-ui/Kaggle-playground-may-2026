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
    → follow `WRAPUP.md` section B exactly. On a non-`main` branch,
    write today's notes to a `## Day-N PM <branch-slug>` H2 section
    inside HANDOVER.md (slug = part after `claude/`); never edit other
    branches' sections. Scribe consolidates at handover time.
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
17. **Wrap-up + handover triggers.** PI says **"wrap up"** → follow
    `WRAPUP.md` section A. PI says **"prepare handover"** → follow
    `WRAPUP.md` section A then section B. Both end with a push to the
    current branch. No manual PI transcription required.
18. **Issue-tree claim before compute (BPS step 4).** Before any new
    probe consuming >10 min CPU/GPU, read `ISSUES.md` and claim an
    unclaimed `open` leaf by editing its `[owner: ...]` to your branch
    slug (part after `claude/`). One open leaf per branch. Update
    status (`wip` → `done`/`null`/`parked`) at wrap-up. Re-decomposition
    of the tree fires on the same triggers as the strategy-critic-loop
    (plateau, saturation, kickoff, 50% checkpoint, "redecompose").
19. **Experimentation harness (BOTE-first / gate-after).** Embed
    BOTE in problem-solving. Workflow:
    (a) Before any candidate ≥10 min CPU/GPU, run
        `python scripts/probe.py bote NAME --family X --cost_min N`.
        SKIP verdict → don't run; DEFER → only if cycles permit;
        PURSUE → ok. Family priors are calibrated to empirical hit rate
        (~17% for s6e5 base population; family-conditional adjustments
        in `FAMILY_PRIORS` dict).
    (b) After artifacts exist, run
        `python scripts/probe.py gate NAME --oof PATH --test PATH`
        for the uniform structured report (standalone OOF Δ, ρ vs
        PRIMARY, predicted LB Δ, G3 flip ratio, verdict). DON'T write
        bespoke gate logic per script.
    (c) For K=21+N stack-add probes use
        `python scripts/probe_min_meta.py --candidates ...`.
    (d) Rule-out is a valid result. Cheap null findings get audit
        notes too (`audit/YYYY-MM-DD-*.md`); don't only document wins.
    (e) "Many small things" beats "one big bet": prefer 5×30-min
        probes over 1×3-h NN unless EV/cost-min strongly favors the
        big bet under the harness's BOTE.
    PI corollary: the calendar/budget belongs to PI; agents do
    not propose timelines or "today/tomorrow" framings — execute
    until PI says stop.

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
day: 16                           # 2026-05-16 PM. **PRIMARY UNCHANGED at LB 0.95059 (d15b_path_b_K22_dae_only_tau20000)**. Branch `claude/read-handover-lA8Nr` ran the virgin-axes complement to HANDOVER T1-T4 (axes α/β/δ/ε/ζ/η from d13 problem-decomposition tree, untouched by other branches). 8 probes / 4 NULL / 3 KILLED / 1 marginal. Load-bearing finding: **5th cross-confirmation of `lr-meta-rank-lock-strong-anchor`** — even GRU sequence model (causal RNN over (Driver,Race) lap windows) at ρ=0.919 standalone diversity (most-diverse base of session) is fully absorbed by K=22 LR-meta with [raw,rank,logit] expand at K=22+1 gate (Δ=-0.043bp NULL). H9 transductive pseudo +0.631bp at LR-meta(K=22) but -0.30bp vs PRIMARY hier-meta (Path-B-amp doesn't fire on base-add per friction). H9+H2 / H9+GRU multi-add ≈ same as H9 alone (+0.671 / +0.629 bp). No new PRIMARY; 0 submits today; meta-arch redesign (HANDOVER T4, owned by other branches) is the only amp-eligible axis remaining.
lb_best_today: 0.95435            # leader; not refreshed
our_lb_best: 0.95059              # d15b_path_b_K22_dae_only_tau20000 (unchanged Day-16); gap to top-5% -2.86bp
submissions_used_today: 0         # Day-16: probe-only night, all OOF/min-meta gates, no LB submits
submissions_used_total: 28
saturation_count: 1               # Day-16 +1: K=22 rank-locked across all virgin base-add axes (5th cross-confirmation including α4 sequence)
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
  - factorization_machine_aug12     # d9h unified 12-field FM + K=22 add LB 0.95034 (+3bp tied PRIMARY, 300× upside)
  - factorization_machine_aug2way   # d9i FM_A_aug + FM_B_aug 2-way partition K=21 swap LB 0.95034 (+3bp tied)
  - factorization_machine_aug15     # d13 EDA H1 -- 12 d9h fields + CRT(Compound×TL_q5×RP_q5) + Cdpl(cumdeg/lap q5) + Ldz(LapTime_Delta race-z q5); std OOF 0.92711 (strongest FM ever, +1.7bp vs aug12), ρ=0.909 (most diverse), K=23 add LB 0.95032 (-2bp regress); FM-FIELD-AUGMENTATION LEVER SATURATED at 12 fields
  - empirical_bayes_hierarchical_meta # d13 Path B -- per-segment LR meta over K=21 PRIMARY pool with τ-shrinkage to global. Compound (5 seg) τ=100k LB 0.95033 (+2bp, 6.7× upside); Stint (5 seg) τ=100k LB 0.95041 (+7bp, 11.6× upside; mechanism leakage-robust per d13d GKF probe 2.9× amplified); **Compound×Stint (24 seg) τ=20k LB 0.95049 NEW PRIMARY (+8bp, 8× upside)**; Year (4 seg)/Race (26 seg)/Year×Compound/Year×Stint UNTESTED
  - groupkf_stack_rebuild_audit     # d10b/c -- FM-class lift +2.01bp under Race-only GKF vs +0.87bp Strat (2.3× AMPLIFIED); FM_B is #1 L1 component under GKF; PRIMARY private-LB robust
  - leak_corrected_lr_meta          # d10d -- refit LR on GKF OOFs; G3 fail (flip ratio 0.001); FM dominance over-credits, smooths GBDT row-extremes; held
  - empirical_bayes_hier_lr_meta    # d13 Stint τ=100000 -- LB 0.95041 NEW PRIMARY (+7bp; 11.6× OOF upside); GKF lift +2.59bp 2.9× AMPLIFIED per d13d probe -- mechanism leakage-robust, private-LB-likely-real
  - t12_censored_regression         # d12 -- LGBM weighted-regression on log(laps_until_pit); std 0.544, FAIL min-meta
  - t12_ratio_target                # d12 -- LGBM regression on pits/stints + heuristic; std 0.674, FAIL min-meta
  - t12_stintlevel_survival         # d12 -- stint-level LGBM duration → row hazard; std 0.601, FAIL min-meta
  - year_segmented_specialist       # d12 -- M_active/M_2023 split FALSIFIED; AV-AUC 0.502 (no shift); 2023 is EASIEST segment
  - adversarial_validation_reweight # d12 -- e3+adv-weight FALSIFIED -4.92bp min-meta; train/test i.i.d.
  - lambdarank_race_meta            # d12 -- LambdaMART Race-grouped REGRESSED -86bp; LR-meta-stays-best
  - aucpairwise_xgb_base            # d12 -- XGB rank:pairwise smoke -451bp fold-0; FAIL gate
  - single_bag_e3_5seed             # d12 -- standalone bag -19bp OOF; K=21 complexity JUSTIFIED (not OOF-noise)
  - groupkf_full_pool_meta          # d12 -- KEY FINDING: rank-lock partial dissolves; ρ(Strat-vs-GKF meta)=0.9914
  - fm_partition_5_3_d13a           # d13a -- FM_A_53 (D,C,S,T,Cd) + FM_B_53 (R,Y,Rp); Strat S3 K=24 +0.20bp pred ρ 0.99976; LB 0.95032 TIE; GKF Δ -41.6/-2.9bp BOTH leakage-robust
  - fm_partition_4_4_ct_axis_d13d   # d13d V2 -- FM_A_CT (C,T,Cd,Ld) + FM_B_DR (D,R,S,Y); K=25 add REGRESS -0.05bp; wheel-physics axis redundant w/ d9f+d13a
  - fm_partition_6_6_alt_d13d       # d13d V3 -- FM_A_DH + FM_B_RT (T moved to B, Nx/Pv to A); K=25 add +0.03bp noise-floor; partition-shape SATURATED across 6 shapes
  - gkf_full_22_stack_d13b          # d13b -- 4-FM (d9c+d9f A/B+d13a A_53/B_53) GKF stack +3.20bp; SWAP_21 (drop d9c) -0.01bp = REDUNDANT; Move C minimal validated under GKF
  - move_c_strat_pool_refactor      # d13c -- T1 drop_d9c K=23 = T0 K=24 (no regress) ✓; T2/T3 drop GBDT leak-eaters -2.5/-2.6bp Strat FALSIFIED — leak-eaters carry public-LB row-iid signal
  - within_stint_lgbm_fe            # d13 G1 -- 6 γ-pack feat (laps_into_stint etc); std 0.94194, ρ 0.965, min-meta -0.38bp NULL
  - cross_driver_intra_race_lgbm_fe # d13 G2' -- 9 γ4 feat (block_tyrelife_std +0.29 row-corr); std 0.94250, ρ 0.957, min-meta +0.03bp NULL
  - stintgrouped_lambdamart         # d13 G3 -- pairwise loss; smoke fold-0 0.74585, killed (63% all-zero stints from probe Q1)
  - fm_aug13_3way_concat_field      # d14 H1 -- CTRq Compound×TL_q5×RP_q5 (114/125 levels); std 0.92639 (+9.9bp vs aug12), ρ **0.917** (most diverse), min-meta -0.13bp NULL
  - path_b_cohort_sweep_d14         # d14 -- Year(4)/Year×Stint(24)/Race(26) × τ∈{5k,20k,100k}; 9 variants, NONE beats current PRIMARY (Compound×Stint τ=20k) on OOF; best Year×Stint τ=20k OOF 0.95080 (-0.30bp). Cohort lever Compound axis dominates Year axis (2023 flat-rate generator defeats per-Year specialization).
  - two_level_stacking_meta_as_base # 2026-05-06 -- K=21 + d12_lr_meta (= K=21 LR-meta-OOF itself) +1.348 bp OOF, but Path B Compound×Stint hier-meta on K=22 SUBMITTED LB 0.95045 (-4 bp REGRESS, predicted +5-11 bp via Path B amp). FALSIFIED: Path B amp requires orthogonal pool signal, NOT meta-derivatives whose info is already convex combo of pool. Friction tag `path-b-amp-needs-orthogonal-signal-not-meta-derivatives`. Harness: `two_level_stacking_meta_as_base` family added (P=0.10, bp band -2/0/1).
  - tabpfn_finetune_v25_v26         # Day-14 -- v2.5 @ 50k rows fold-0 0.94439; v2.5 @ 150k rows fold-0 0.94446 (IDENTICAL — flat training loss; fine-tuning not learning); v2.6 OOM P100 at any row count (model weights ≈15.37GB > 16GB). AUC ceiling 0.944 = -64bp vs PRIMARY. ρ=0.960 (diverse) but gap too large for pool. DEAD.
  - fm_new_input_features           # Day-14 Move D -- F1_PitWindow_q5, F2_HazardDecay_q5, F3_CompoundPress_q5, F4_RaceStage. FM_aug16 (16-field): std 0.92741 (+20.1bp vs aug12 0.92540), ρ=0.919, min-meta -0.07bp FAIL. Confirms FM-field-augmentation saturated at 12 fields; new input types add standalone OOF but zero pool increment.
  - masked_column_self_prediction   # Day-14 -- 4 LGBM regressors predict LapTime_Delta/Cumulative_Degradation/Position/LapNumber from rest of row; OOF z-residuals + L1 anomaly as 5 new features for LGBM target model. std 0.94200 (-88bp), ρ=0.9599 (diverse), K=2 min-meta -0.025bp NULL, K=22 add +0.172bp at ρ=0.9958 (pred LB -1.3bp under harness band). Load-bearing diag: across all 4 targets OOF RMSE ≈ marginal σ within 3 sig figs — synthetic NN-DGP added near-independent per-feature noise within rows. 5th NULL on per-row-FE axis; jointly explains FM-aug12 saturation, Move D NULL, Day-13/14 alt-axis 4-of-4, TabPFN 0.944 ceiling. Friction tag `synthetic-dgp-conditionally-near-independent`. Family closed.
  - kd_distillation_lgbm            # 2026-05-06 -- small LGBM mimicking K=21 LR-meta-OOF logits; std 0.94212; K=21+1 +0.526 bp meta-derivative-class artifact (HELD; LB-regress predicted by meta-derivative pattern).
  - nn_with_embedding_layers        # 2026-05-06 -- MLP w/ Driver/Race/Compound/Year/Stint embeds; std 0.92362 (-272 bp), ρ=0.918 most-diverse measured but K=21+1 -0.025 bp NULL. Confirms ρ alone insufficient predictor of meta-utility.
  - lap_mod_features_lgbm           # 2026-05-06 -- LapNumber_mod_{3,5,7,10}+id_mod_{5,7,13,100,1000}; id-audit found LapNumber_mod_10 marginal target span 566 bp; std 0.94076, K=21+1 +0.002 bp NULL. The 566 bp span absorbed by GBDT feature interactions.
  - pseudo_label_confidence_extreme # 2026-05-06 -- top/bot 5% test by PRIMARY confidence as half-weight pseudo; std 0.94083, K=21+1 +0.019 bp NULL.
  - within_race_lt_quantile         # 2026-05-06 -- LapTime_Delta within-(Race,Year) q5 as LGBM feat; std 0.94008, K=21+1 +0.20 bp NULL/marginal. The +922 bp single-feat leak signal absorbed.
  - year_stint_sparse_lr            # 2026-05-06 -- one-hots for Year/Stint/Compound + Driver-hash + Year×Stint and Y×S×C; std 0.88164 (very weak), ρ=0.844 most-diverse, K=21+1 +0.05 bp NULL.
  - blend_aggregators_K21           # 2026-05-06 -- mean/gmean/rank_mean/trimmed of K=21 standalone OOFs; ALL std OOF -19 to -32 bp vs PRIMARY. RULED OUT: LR meta is doing real work, simple blends never match it.
  - driver_cluster_path_b_cohort    # 2026-05-06 -- k-means k=4 on per-Driver stats → cluster cohort; Compound×cluster (5×4=20 segs) Path B; -0.4 to -0.9 bp NULL across τ. Cohort axis exhausted.
  - alpha_calibrated_tau_resweep    # 2026-05-06 / d15 Branch A -- two independent confirmations. main-branch agent: Path B α computed with full-train counts at OOF; τ=20k UNCHANGED (ρ=1.0, Δ -0.02 bp). d15 Branch A re-ran the same hypothesis end-to-end: ρ=1.000000 vs d13e at τ=20000 (literally identical predictions). At segments ≥1000 rows, α=n/(n+τ)≈1 in both regimes; smaller τ values regress (τ=2000 -0.99bp). PRIMARY's τ=20k empirically optimal — calibration is not the binding constraint.
  - id_order_synth_artifact         # 2026-05-06 -- LapNumber_mod_10 marginal span 566 bp, id_mod_1000 568 bp. Marginal span DOES NOT translate to predictive lift when GBDT has feature interactions.
  - target_reformulation_invlaps    # 2026-05-06 -- LGBM regression on 1/(1+laps_until_pit), target-derived (NOT meta-derivative). Std OOF 0.94053 (-103 bp), ρ=0.924. K=21+1 +1.899 bp OOF. **POSITIVE single-add finding** (largest non-meta-derivative single-add lift ever measured). HELD candidate.
  - target_reformulation_stintprog  # 2026-05-06 -- LGBM regression on TyreLife/max(stint), std 0.64851, ρ=0.252 (most-diverse single base ever); K=21+1 alone NULL.
  - multi_target_nn_pit_aux_invlaps # 2026-05-06 -- shared-trunk NN with pit_next_lap (BCE) + inv_laps (MSE 0.3) heads; std 0.92295, K=21+1 +0.086 bp NULL.
  - path_b_K22_invlaps_compound_stint # 2026-05-06 -- Path B Compound×Stint hier-meta on K=22 = K=21 + inv_laps_until_pit; OOF 0.95110 (+2.75 bp vs PRIMARY) at τ=20k, ρ=0.99753. **LARGEST OOF ADVANCE OF SESSION**, target-derived. Held pending submission decision.
  - extra_trees_5fold_d15c          # d15 Branch C -- ExtraTreesClassifier(4000, max_features='sqrt') 5-fold; std OOF 0.92967 (underfits row-iid leakage GBDTs eat, as predicted); min-meta +0.059bp at ρ 0.99599 = noise-floor band. R5 HEDGE only.
  - knn_distance_lgbm_d15d          # d15 Branch D -- per-Compound + per-Driver k=5 NN distances (10 features) + LGBM; std OOF 0.94166; min-meta +0.056bp at ρ 0.99586. C+D K=23 add additive +0.095bp; ρ between C&D raw 0.9325 (diverse) but LR-meta routes both to ρ≈0.996 vs PRIMARY. Rank-lock pattern reasserted. R5 HEDGE only.
  - dae_swap_noise_lgbm_d15b        # d15 Branch B -- Jahrer Porto-Seguro recipe ported to GPU. DAE 256-512-256 swap-noise frac=0.15, 20 epochs batch=4096 on (train+test 627k); 768d latent (h2+h3 concat); LGBM 5-fold on raw+latent vs latent-only. **`dae_only` std OOF 0.94007, ρ_test 0.9477 (most-diverse since FM_A_53), min-meta +0.793bp at ρ 0.99547.** K=22 Path B Compound×Stint τ=20000 OOF 0.95090 +0.715bp vs d13e, ρ 0.99973, flips 59/53 R7-eligible. **SUBMITTED 2026-05-06 15:38: LB 0.95059 (+1.0bp NEW PRIMARY).** Realised amp 1.4× (above ρ-band baseline +0.22bp, well below Path-B-amp 6-11.6× central). New friction `path-b-amp-only-fires-on-meta-arch-not-base-add`: amp transfers on meta-architecture redesign (segmentation lifts) NOT on K_pool→K_pool+1 base additions, even when new base is orthogonal-class.
  - d15_orig_transfer               # 2026-05-06 branch decode-synthetic-data-uoPIn -- LGBM trained on aadigupta1601 original (99k rows, source verified) → predicts synth. Std-alone synth AUC 0.85138 (-99bp vs PRIMARY), ρ vs OLD PRIMARY 0.565 (most-diverse single base since d9f FM_A 0.487). At LR-meta(K=22) +0.778bp OOF; LR-meta SUBMITTED LB 0.95039 (-10bp REGRESS, meta-arch confound). Hier-meta(K=22, Compound×Stint τ=20k): +1.127bp OOF, ρ=0.99844, flips 180 (R7 ✓), SUBMITTED LB 0.95049 TIE → HEDGE-tier candidate. Mechanism: synthesizer corrupted joint structure but kept marginals; orig-trained model carries un-corrupted DGP signal orthogonal to GBDT pool.
  - d15_orig_multi_arch_bag         # 2026-05-06 -- d15_orig_cb (CB) + d15_orig_xgb (XGB) + d15_orig_lgbm_t (tuned LGBM) trained on same original. Inter-arch ρ 0.94-0.99 (high), ρ vs PRIMARY 0.57-0.64. Hier-meta K=23(+cb) Δ +0.005bp NULL; K=24(+cb+xgb) Δ +0.33bp but flips 293>R7-200 cap. Multi-arch on shared training-data is REDUNDANT; vary training-data subset, not architecture. Friction tag `external-data-arch-bag-redundant-when-shared-training-data`.
  - d15_decode_normalized_tyrelife  # 2026-05-06 -- direct lookup of host-removed Normalized_TyreLife from original (5.5% match rate) + stint-fraction estimate fallback. Formula recovered: NTL = TyreLife / D(Driver,Race,Year,Stint), D ≈ stint length. Standalone OOF 0.94162; min-meta(K=21) Δ -0.008bp NULL (absorbed by TyreLife+RaceProgress+Stint already in pool). Cheap rule-out.
  - d15_physics_residual            # 2026-05-06 -- Ridge LapTime ~ Driver+Race+Year+Compound+TyreLife (5-fold OOF residual) + per-Race-Compound z-score on Cumulative_Degradation. Std OOF 0.94228, min-meta Δ -0.036bp NULL. Physics already in GBDT pool.
  - d15_leak_lookup                 # 2026-05-06 -- 16 EB-smoothed lookup features from original (P(PitNextLap | LapTime), univariate/bivariate/trivariate). 97.55% of synth LapTime values exist in original (CTGAN/CopulaGAN signature). Std OOF 0.94203, min-meta(K=21) Δ +0.270bp soft-pass (smaller than orig_transfer). Hier-meta K=22(leak-only) -0.90bp vs orig; K=23(leak+orig) +0.19bp incremental over orig — best OOF on branch 0.95096 but ρ=0.9986 → predicted LB tie, NOT submitted.
  - d16_year_2023_hard_mask         # Day-16 H4 -- post-process zero-mask on (Year=2023 ∩ rare-Driver). Best K=5 +0.004bp ceiling NULL. PRIMARY hier-meta already routes 2023 rare rows to near-zero (pos rate 0.96%). d13 G4 axis falsified.
  - d16_conformal_isotonic          # Day-16 H7 -- per-bin isotonic recalibration of PRIMARY OOF, inner-CV-validated, 4 schemes (Compound / Year×Compound / Year×Compound×Stint / RaceProgress_q5×Compound). All schemes regress -2.5 to -9.6 bp. PRIMARY hier-meta is globally well-calibrated. δ2/3 axis NULL.
  - d16_two_stage_stint_logistic    # Day-16 H10 -- α5 axis (d13 tree). Stage-1 LGBM regression on E[T_stint] + stage-2 1-D logistic on remaining-laps. Std OOF 0.625 NULL — stage-2 too restrictive (1 feature). Methodological miss; α5 axis not falsified.
  - d16_twin_pool_2_meta_blend      # Day-16 H2 -- ε2 axis. Pool A (6 GBDT) + Pool B (5 model-class diverse) with ρ(metaA,metaB)=0.967 real disagreement. Top-level LR over [metaA, metaB] OOF 0.95010 vs single LR-meta(K=11) 0.95028 -- FALSIFIED Δ -1.79bp. Friction `twin-pool-2-meta-collapses-rank-info`. 2-feature top-level LR collapses 33-dim rank info that K=11 LR captures.
  - d16_deepgbm_leaf_encoding       # Day-16 ε4/ε4b -- ε4 axis. Stage-1 LGBM leaf-indices → stage-2 (cat-LGBM 627 features KILLED 16min over-engineered; sparse-LR head fold-0 0.92507 weak, sparse-LR ~20min/fold). KILLED both. ε4 axis NULL within tested impl.
  - d16_av_sample_weight_lgbm       # Day-16 H11 -- ε axis. AV classifier (train vs test) → AV-prob as sample weight in base LGBM. KILLED at 12min on AV stage under contention. EV bounded by AV-AUC=0.502 (no global shift).
  - d16_transductive_pseudo_full    # Day-16 H9 -- ζ6 axis. LGBM trained on (synth_train + half-weight PRIMARY-pseudo-test) 627k rows. Std OOF 0.93433, ρ=0.872. K=22+1 LR-meta Δ +0.631 bp PASS at LR-meta-K22 baseline; meta gives negative-direction routing weight (raw -0.295, logit -0.222). BUT vs PRIMARY hier-meta Δ -0.30 bp regress -- Path-B-amp doesn't fire on base-add per friction. MARGINAL HEDGE candidate.
  - d16_gru_sequence_alpha4         # Day-16 H1 -- α4 axis (d13 tree, virgin). Causal GRU 1-layer hidden=96, embeds (Driver 16 + Compound 4 + Race 4 + Year 4) over (Driver,Race) lap windows; Kaggle T4×2, 12 epochs × 5-fold, 58 min wall. Std OOF 0.93066, ρ_test 0.919 (most-diverse single base of session). K=22+1 LR-meta Δ -0.043 bp NULL. **5th cross-confirmation of `lr-meta-rank-lock-strong-anchor`** -- even prediction-unit-distinct sequence model fully absorbed by [raw,rank,logit] LR-meta. New friction `temporal-axis-also-rank-locked-at-K22`.
plateau_days: 1                   # Day-16 no advance: 4 NULL + 3 KILLED + 1 marginal H9. K=22 + Path-B-hier-meta architecture rank-saturated against EVERY base-add axis (per-row FE / calibration / α4 sequence / α5 two-stage / β rank loss / ε twin-pool / ε4 leaf-encoding / ε AV-weight / ζ6 transductive / η1 mask). Day-17 priority: META-ARCH REDESIGN (HANDOVER T4 -- non-Gaussian shrinkage, nested hierarchy, Yao/Vehtari covariance-BMA, alt segmentation crosses) -- owned by `claude/ml-handover-alignment-xvUN0`. If T4 doesn't land: (a) external second-source data (Ergast/FastF1) never tested; (b) Pirelli scrape (HANDOVER A4); (c) structured pool-replace (drop 5 weakest GBDT clones + add 5 fresh diverse-architecture bases).
gate_status: cleared              # d15b_path_b_K22_dae_only_tau20000 LB 0.95059 PRIMARY (unchanged Day-16)
headroom_to_top5pct: 0.00286      # 0.95345 − 0.95059 = 28.6bp (unchanged Day-16; no submits)
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
| d13_path_b_stint_tau100000 | 0.95082 | 0.94600 | 0.95041 | demoted by d13e; +7bp LB; 11.6× OOF upside; GKF lift +2.59bp = 2.9× AMPLIFIED vs Strat +0.90bp (leakage-robust per d13d) |
| d13e_compound_stint_tau20000 | 0.95083 | n/a | 0.95049 | demoted by d15b_K22; +8bp LB over Path B Stint; 8× OOF upside (+1.00bp OOF); ρ=0.9958 vs d9f; 24 seg cross |
| d15b_lgbm_dae_full (DAE 768d + raw → LGBM) | 0.94325 | n/a | n/a | min-meta +0.221bp at ρ 0.99537; raw+latent redundant LGBM-extractable |
| **d15b_lgbm_dae_only (DAE 768d → LGBM)** | **0.94007** | n/a | n/a | std-alone, ρ_test 0.9477 (most-diverse since FM_A_53); min-meta +0.793bp at ρ 0.99547; clean orthogonal axis |
| **d15b_path_b_K22_dae_only_tau20000** | **0.95090** | n/a | **0.95059** | **NEW PRIMARY** (+1.0bp); K=22 Path B = K=21 + d15b_dae_only; +0.715bp OOF, ρ=0.99973, flips 59/53; realised amp 1.4× (Path-B-amp does NOT fire on base-add) |
| d13e_compound_stint_tau100000 | 0.95081 | n/a | n/a | held; +0.82bp OOF; ρ=0.9996 vs Stint winner (TIE band); 55/98 flips (under R7 200); HEDGE-eligible if τ=20000 lands |
| d13b_path_b_stint_tau20000 | **0.95082** | n/a | n/a | held; +0.88bp OOF; ρ=0.996; flip ratio 0.220; tau=100000 superseded by submit |
| d13_path_b_compound_tau100000 | 0.95076 | n/a | **0.95033** | calibration probe; LB +2bp on +0.30bp OOF (6.7× upside); ρ=0.9990; demoted by Stint variant |
| d13_g1_within_stint (LGBM, +6 γ FE) | 0.94194 | n/a | n/a | NULL; ρ=0.9651 vs PRIMARY (0.95073 anchor); min-meta -0.38bp; LGBM-class feature add dead |
| d13_g2_cross_driver (LGBM, +9 γ4 FE) | 0.94250 | n/a | n/a | NULL; ρ=0.9572; min-meta +0.03bp; cross-driver intra-race signal already in pool |
| d14_h1_fm_aug13_3way (FM, +CTRq) | **0.92639** | n/a | n/a | NULL vs Path B PRIMARY; ρ=**0.9169** (most diverse single base); min-meta -0.13bp; +9.9bp standalone over d9h_aug12 but no incremental signal at meta |
| d14_fm_aug16 (FM, 12+4 Move-F) | **0.92741** | n/a | n/a | NULL; +20.1bp standalone vs aug12; ρ=0.919 diverse; min-meta **-0.07bp FAIL**; FM-field-augmentation SATURATED at 12 fields (new input types F1-F4 confirm) |
| d14_tabpfn_v25_150k (fold-0 only) | 0.94446 | n/a | n/a | DEAD; identical to 50k-row result 0.94439; flat training loss; fine-tuning ceiling ≈0.944 at any row count; v2.6 OOM P100; -64bp vs PRIMARY |
| d12_groupkf_meta (K=21 GKF) | 0.95069 / **GKF 0.94776** | n/a | n/a | **Day-12 STRUCTURAL FINDING**: ρ(Strat-vs-GKF meta-test)=0.9914 — rank-lock partial dissolves; FM ΔAUC −9bp vs GBDT −200 to −343bp |
| d12_groupkf_meta_no_realmlp K=20 | 0.95056 / **GKF 0.94577** | n/a | n/a | clean K=20 (no realmlp Strat anchor); ρ vs Strat-meta 0.9856; GroupKF-meta candidate HEDGE for R5 |
| d12 single bags (e3 5seed / cb 3seed) | 0.94876 / 0.94790 | n/a | n/a | calibration probe -- regress -19/-28bp every segment vs PRIMARY; K=21 complexity JUSTIFIED, NOT OOF-noise overfit |

## Hypothesis board (Day 12 evening)

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
- DONE: d13/d13b Path B empirical-Bayes hierarchical LR meta.
        Per-segment partial-pooled LR with shrinkage τ. Sweep:
        Compound (5 seg) τ=100000 → +0.30bp OOF, ρ=0.9990 PASS;
        **Stint (5/6 seg) τ=100000 → +0.86bp OOF, ρ=0.9984**, flip
        ratio 0.211 FAIL G3 (but 4-5× better balanced than d10d).
        Compound×Stint segmentation killed at fold 2; Year×Compound
        not run. Stint variants held for R5 final-window OOF-best.
- DONE: d13c Path B Compound τ=100000 SUBMITTED at 05:31 UTC.
        **LB 0.95033, +2bp lift over d9f K=21 swap PRIMARY** (-1bp
        from d9h/d9i tied PRIMARY at 0.95034). 6.7× LB upside on
        +0.30bp OOF prediction — FM-class amplification pattern
        TRANSFERS to hier-meta architecture.
- DONE: d13 Path B Stint τ=100000 SUBMITTED at 05:34 UTC.
        **LB 0.95041, +7bp lift over d9h/d9i tied PRIMARY → NEW
        PRIMARY**. 11.6× LB upside on +0.86bp OOF prediction.
- DONE: d13d GroupKF probe of hier-meta. K=20 GKF pool (no realmlp).
        Global LR meta GKF OOF 0.94574; Stint hier τ=100000 GKF OOF
        **0.94600 (+2.59bp)**. Strat lift +0.90bp → GKF lift +2.59bp
        = **2.9× AMPLIFIED** (stronger than FM-class 2.3× in d10b/c).
        Hier-meta mechanism is leakage-robust. **Public-LB +7bp lift
        is mechanism-driven, not sample variance.** Revised private-LB
        estimate: median +4 to +6bp over HEDGE (conservative +2bp,
        bull +7bp). Three independent leak-blocking probes agree.
- LATER: External-data Pirelli pit-window scrape (Tier-2 highest
        absolute EV), EmbMLP CPU (different model class), hazard NN
        (GPU; d9 hazard_nn_stack regressed 315bp — implementation
        matters; main-branch agent's leakfree hazard NN at OOF 0.92013
        confirmed DEAD).
- DONE Day-12: 6 wider-step options run as parallel subagents
        overnight. 5 FALSIFIED. **Option 1 produced load-bearing
        finding.**
- DONE: Option 1 GroupKF full rebuild — K=21 LR-meta on GroupKFold
        OOFs produces test predictions with ρ=0.9914 vs Strat-meta
        (K=20 clean: 0.9856). **Rank-lock partially dissolves under
        leakage-blocked OOF.** Per-base ΔAUC: GBDT bases drop −200 to
        −343bp under GKF; FM/rule/sparse-LR drop −9 to −43bp. **FM is
        23–37× more leakage-robust than every GBDT.** L1 ranking
        shifts: cb_slow-wide-bag −17 ranks, e5_optuna_lgbm −13; FM
        +15 (#20→#5). The Strat-OOF "all-bases-tie" is GBDTs eating
        shared fold-mate signal; FM/rules generalize.
- DONE: Option 9 single-bag probe — falsifies "OOF-noise overfit"
        thesis. Single 5-seed/3-seed bags regress -19 to -28bp every
        Year/Stint/Compound segment vs PRIMARY. K=21 stack complexity
        is JUSTIFIED — it routes between leakage-eaters and
        leakage-robust bases.
- DONE: Option 3 T1.2 multi-formulation 3-of-3 (censored / ratio /
        survival LGBMs) — ALL FAIL min-meta. Standalone OOF 0.54-0.67
        confirms time-to-event LGBMs are GBDT-class (leakage-eaters
        category). 4-of-4 T1.2 cohort dead-listed.
- DONE: Option 4 Year-specialist + AV-reweight — BOTH FAIL min-meta
        -4.5 to -5.0bp. Two crisp findings: (a) AV-classifier AUC =
        0.502 (train/test i.i.d.; no shift to exploit, refutes
        external-info-as-lever thesis), (b) Year=2023 is the EASIEST
        segment for the pool (AUC 0.94602, highest of any Year);
        cohort splitting strips cross-Year regularization (specialist
        2023-AUC -105bp). P3's "-45bp 2023 lift" interpretation
        INVERTED.
- DONE: Option 5 LambdaRank meta — REGRESSES -86bp under Race
        grouping; AUC-pairwise XGB base smoke -451bp fold-0. LR-meta
        on [raw, rank, logit] is metric-aligned for global AUC when
        bases are well-calibrated; pairwise-rank objectives offer no
        lift here. Dead-list.
- DONE: Option 2 TabPFN-2.5 fine-tuned — Kaggle GPU kernel ready at
        `kernels/d12-tabpfn-finetune-gpu/`. CPU smoke blocked on
        license. T4×2 wall 5-7h. EV +5-15bp std-alone, +1-3bp stack
        median, tail +3-9bp. **Only live "10bp shot" remaining.**
- INSIGHT (Day-12 unifying frame): K=21 stack works because LR-meta
        routes between two base populations — leakage-eating GBDTs
        (high Strat AUC, real LB AUC much lower) and leakage-robust
        FM/rules (Strat AUC ≈ GroupKF AUC). Public LB is row-iid (U3)
        so PRIMARY survives, but the diversification we need is
        WITHIN the leakage-robust population. **Pivot:** more FM-class
        bases (5/3 multi-FM, 4/4 multi-FM, DeepFM-lite, regularised
        FFM re-attempt). Replace 3 most-leakage-eating GBDTs.
- NEXT (Day-13+): (a) push TabPFN kernel to Kaggle (PI-approved);
        (b) build 5/3 + 4/4 multi-FM partitions (cheap CPU); (c)
        DeepFM-lite (FM + 2-layer MLP head); (d) GroupKF-meta as R5
        final-3-day HEDGE candidate (don't submit as PRIMARY — public
        is row-iid).
```

## Pointers

- `HANDOVER.md` — next-session brief (Rule 15).
- `WRAPUP.md` — wrap-up + prepare-handover procedure (Rule 17).
- `ISSUES.md` — live problem decomposition / claim board (Rule 18).
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
- `audit/2026-05-12-d12-master-synthesis.md` — Day-12 6-option overnight synthesis.
- `audit/2026-05-12-d12-groupkf-rebuild.md` — Option 1 STRUCTURAL FINDING.
- `audit/2026-05-12-d12-tabpfn-finetune-prep.md` — Option 2 Kaggle GPU kernel ready.
- `audit/2026-05-12-d12-t12-multi-formulation.md` — Option 3 T1.2 3-of-3 falsified.
- `audit/2026-05-12-d12-year-specialist-advweight.md` — Option 4 falsified; AV-AUC 0.502.
- `audit/2026-05-12-d12-lambdarank-meta.md` — Option 5 -86bp regression.
- `audit/2026-05-12-d12-monolithic-bag-probe.md` — Option 9 K=21 complexity justified.
- `audit/2026-05-10-d9h-fm-augmented.md` — FM_aug12 standalone strongest; K=22 add LB 0.95034 (+3bp NEW PRIMARY tied).
- `audit/2026-05-10-d9i-augmented-2way.md` — aug 2-way K=21 swap LB 0.95034 (+3bp NEW PRIMARY tied; OOF was -0.19bp regression).
- `audit/2026-05-10-d10-groupkf-audit-fm-real.md` — strict GKF FM bases drop 2.5–54bp vs GBDTs −210bp.
- `audit/2026-05-10-d10b-groupkf-stack-rebuild.md` — FM-class lift +2.01bp GKF vs +0.87bp Strat (2.3× AMPLIFIED); FM_B is #1 L1 under GKF.
- `audit/2026-05-10-d10d-leak-corrected-meta.md` — leak-corrected LR meta gate-FAILs (G3 flip ratio 0.001) but informative; Bayesian hierarchical is correct synthesis.
- `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B empirical-Bayes; Stint τ=100000 +0.86bp OOF, ρ=0.998, G3 0.211 FAIL; R5 candidate held.
- `audit/2026-05-13-d13d-path-b-gkf-probe.md` — Path B GKF probe; Stint hier-meta lift +0.90bp Strat → +2.59bp GKF (2.9× AMPLIFIED) — mechanism leakage-robust.
- `audit/friction.md` — friction one-liners.
