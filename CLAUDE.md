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
16. **New-candidate pre-flight (6-question check).** Before committing
    CPU/GPU compute on any new base or meta variant, answer:
    (1) Is the underlying mechanism in `mechanism_families_explored`?
    (2) Does the candidate fall in {meta-only, rule_residual-on-raw,
    GBDT-on-binary-target, formulation-already-in-pool}? If yes,
    rank-lock-vulnerable. (3) Predict standalone OOF (cite precedent).
    (4) Predict ρ vs PRIMARY (cite closest base). (5) At that ρ,
    cite the closest gate-PASS/FAIL precedent. **(6) Does the training
    objective match the row-level AUC metric?** Pairwise-rank /
    group-rank / multi-task-aux objectives ≠ row-AUC and trigger a
    one-tier verdict downgrade in BOTE (`bote(metric_aligned=True/
    False)` or `--metric-aligned`; unanswered = forced SKIP). Q6
    origin: d12 LambdaRank meta -86bp + AUC-pairwise XGB -451bp
    fold-0 + MAST FM-2.6 reasoning-action-mismatch (arXiv 2503.13657).
    If 1–6 don't return a coherent answer, downgrade EV midpoint by
    0.3× before ranking. Origin: `tag: menu-overcrediting-redundant-
    mechanism` (Day-8 falsified T1.5/T1.3/T1.2 all of which passed
    research-agent EV ranking but failed the 5-question check
    retroactively).
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
    (f) **Calibration loop (added 2026-05-06).** Every BOTE call
        accepts `--metric-aligned true/false` (Q6, mandatory) and
        `--pi-predicted-lb-bp X` (PI's own LB-Δ prediction tracked
        next to the agent's `expected_lb_bp`). Records append to
        `audit/decisions.jsonl` (decision-time log; locks framework_sha
        + agent_branch at decision-time). After a submit lands, run
        `python scripts/probe.py record-outcome NAME --actual-lb-bp X`
        to close the loop; `python scripts/probe.py calibration`
        emits PI vs agent vs actual error per family.
    (g) **Family kickoff seed (added 2026-05-06).** When opening a
        NEW mechanism family (one not in `mechanism_families_explored`),
        run `python scripts/research_seed.py FAMILY` to generate a
        web-retrieval stub at `audit/research-seed-<slug>-DATE.md`,
        then fill it via WebSearch (≥5 sources, ≥2 prior-comp, ≥1
        paper, ≥1 GM blog). Origin: MLE-STAR's web-retrieval seed
        accounts for most of its +47bp over AIDE on MLE-bench-Lite
        (arXiv 2506.15692).
    PI corollary: the calendar/budget belongs to PI; agents do
    not propose timelines or "today/tomorrow" framings — execute
    until PI says stop.
20. **Single-model-first / kitchen-sink FE before stacking.** Before
    adding a 2nd base or LR-meta, build the kitchen-sink feature
    factory (≥30 engineered features + CV target encoding on every
    high-card combo, **including 3-way**) and a SINGLE model first.
    That OOF is the floor; stacking adds on top, it does NOT replace
    it. Origin: s6e5 Day-16 PI question after we ran 16 days of
    stack-mechanism work without ever asking what's the best single
    model. Single LGBM with Rozen's recipe matched our K=22+Path-B
    PRIMARY OOF on Fold-1 alone.
21. **Family falsification requires ≥3 variants.** A mechanism
    family (TE, FM, lag, target-reform, pseudo, calibration) is
    only "dead" after ≥3 distinct configs of the key hyperparameter
    (smoothing, polynomial order, field count, key cardinality,
    regularisation). Single-variant nulls update the prior on that
    VARIANT, not on the family. Origin: s6e5 d3a closed TE family
    on Day-3 single 2-way × single smoothing variant; the 3-way
    (Driver, Race, Year) at smoothing 20 was the comp's +200 bp
    standalone trick that sat unused for 13 days.
22. **Public-notebook scan at every plateau.** Triggered on 3
    consecutive nulls, 5 saturations at same LB, 50% checkpoint, or
    "redecompose": pull top 5 Kaggle public notebooks for the comp
    slug (≥10 votes), list their features + OOF AUCs + model classes.
    Question: which features are NOT in our pool? Build that gap as
    the next experiment. Strengthens Rule 7 with comp-specific recipe
    intelligence. Origin: s6e5 Day-16 — Rozen's 0.95354 recipe sat
    at 19-72 votes the entire comp without us pulling it.
23. **Framework is scaffolding, not authorship.** The framework
    (BOTE → gate → submit, 7-step, ISSUES tree) optimises HOW to
    evaluate. It does NOT generate WHAT to evaluate. Reserve ≥1 slot
    per 3-day cycle for free-form FE creativity uncoupled from
    existing pool. Triggered when 3+ days pass without a probe whose
    source idea is NOT a 1-step variant of an existing experiment.
    Origin: s6e5 16-day plateau where every probe was a rank-locked
    stack-add variant; discipline is necessary but not sufficient.
24. **Fold-safe label-conditional aggregates.** ANY feature derived
    from labels via groupby aggregation (target encoding, mean of
    positives per group, target-conditional ratios) MUST be re-fit
    inside each CV fold using ti rows only. Never include val/holdout
    labels in the aggregate. For test prediction, either refit on full
    train + apply, OR 5-fold-average models each with own ti-fitted
    aggregate. **Diagnostic:** strict 80/20 holdout test (independent
    seed, FE on 80% only, eval on 20%) detects this in <10 min CPU
    without burning a slot. If holdout_AUC ≪ OOF_AUC by >10 bp, leak
    present — debug before submit. Origin: Day-17 P1 v2 — FS_A merges
    (`compound_avg_life`, `race_avg_pit_lap`, `dc_avg_stint_life`)
    fit on full train inflated OOF by 491 bp (0.95128 vs honest
    holdout 0.94637); v1 single LB 0.94107 (−863 bp gap); K=2 LR-meta
    LB 0.94996 (−63 bp).
25. **Transductive features need AV check.** Even using test FEATURE
    values (not labels) at training time can be unsafe if train/test
    distributions differ. Frequency encoding, quantile binning,
    factorize maps, PCA/AE fit on combined train+test can encode
    distributional structure that differs between train/test or
    public/private LB. Rule: before any combined-set transform, run
    adversarial validation (train-vs-test classifier AUC). If
    AV-AUC ≈ 0.5, combined-set FE is safe. If AV-AUC > ~0.55, fit on
    train only. (s6e5 AV-AUC = 0.502 per U3, so combined-FE was
    safe here.) Companion to Rule 24 — Rule 24 covers label-derived
    features; Rule 25 covers feature-value transforms. Origin:
    PI Day-17 lesson on cross-comp generalisation discipline.

26. **PI interaction protocol (non-coding PI; added 2026-05-06).**
    PI is read+strategy, not keyboard. Agent runs all Python; PI
    ratifies and calibrates. Anti-rubber-stamp rituals:
    (a) **Sealed-prediction order.** Before agent reveals its BOTE
        `expected_lb_bp` for any candidate, agent FIRST asks:
        "what's your LB Δ prediction in bp?" PI commits a number +
        one-line rationale to chat. THEN agent reveals its number
        and they go to `audit/decisions.jsonl` together via
        `--pi-predicted-lb-bp`. Agent revealing first poisons the
        calibration loop via anchoring.
    (b) **Three required questions on every BOTE.** Agent asks PI:
        (i) PI-predicted LB Δ (sealed per (a));
        (ii) Q6 — does training objective match row-AUC?;
        (iii) which precedent is PI pricing this against? (cite a
        calibration-ladder row). Skipping any → agent runs without
        `--pi-predicted-lb-bp` and flags the omission in chat.
    (c) **Devil's-advocate ritual.** Once per session, agent picks
        its strongest current recommendation and argues *against* it
        in 3 bullets. PI accepts the counter or rebuts it. Surfaces
        blind spots cheaply; replaces "hand-run a probe" for non-
        coding PI.
    (d) **Daily deep-read.** PI picks one audit note end-to-end and
        writes back: "load-bearing finding is X; the part I don't
        follow is Y". Agent must explain Y. Builds mechanism
        intuition without code.
    (e) **Override-rate** is captured by the postmortem skill
        (`.claude/skills/postmortem/SKILL.md` step 2 "PI-overrides").
        0/M overrides across 2 consecutive postmortems → flag stamp
        risk in HANDOVER.md `## Where we are`. Origin: rubber-stamp
        anti-pattern (Sethserver; MLE-bench HITL literature).
    Origin: `knowledge-base/concepts/agentic-kaggle-systems-comparison.md`
    HITL section + non-coding-PI reframe (2026-05-06 chat).

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
day: 17                           # 2026-05-07 PM-late. **🎯 NEW PRIMARY: LB 0.95368 (d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000)** via branch `claude/reverse-engineer-data-generation-Hu8EK`. K=27 = K=21 + v4 (CB yekenot pulled from main) + h1d (RealMLP yekenot pulled from main) + d16 + d18 + E2 + F2. Path-B Compound×Stint τ=100k OOF 0.95432. PI directive "combine main findings with our DGP-class" → solo-marginal probe over K=23 v4+h1d showed d16 +0.79 / E2 +0.42 / d18 +0.33 / F2 +0.25 add real (DAE NULLed +0.16, leak/d18b/Rozen/orig-transfer all <+0.1). K=27 Path-B τ=100k Δ vs main K=23 v4+h1d PRIMARY +1.7 bp OOF; ρ=0.999 borderline tie; flips 129/154 R7 OK. Submission #39: LB 0.95368 (+1.4 bp over previous PRIMARY 0.95354). PI band 0-2 bp; realised +1.4 bp dead-center. Top-5% gap −3.7 bp (was −5.1 bp; boundary 0.95405); leader 0.95476 (-10.8 bp). 14 probes this session in DGP-reverse-engineering arc (E1-E5 + F1-F6 + G/H/I/J/K).
lb_best_today: 0.95476            # MILANFX leader (refreshed Day-17 PM-late)
our_lb_best: 0.95368              # d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000 (Day-17 PM-late NEW PRIMARY); gap to top-5% -3.7 bp
submissions_used_today: 8         # Day-17: K22_add_p1_feA_te, p1_v1, K2_PRIM_v2, d16_cont_only, d18_K23_d16_d18 LB 0.95149, d17_K22_v3 0.95143, d17_K24 0.95345, d17_K23_v4_h1d 0.95354, **d18_K27_v4h1d_d16_d18_e2_f2 LB 0.95368**
submissions_used_total: 39
saturation_count: 0               # reset on PRIMARY advance
mechanism_families_explored:  # Q1 reference. Detail in audit/. Compressed 2026-05-06.
  - baseline_lgbm_raw_features
  - oof_target_encoding
  - xgb_native_categorical
  - catboost_native_categorical
  - relative_state_fe
  - lr_meta_stacker_3view
  - dirichlet_random_search
  - hgbc_label_encoded_driver       # E3 — BEST single-model pre-CB
  - row_subsample_catboost          # E1
  - l1_meta_sweep                   # E2 — null
  - realmlp_cpu_singlefold          # E4 — not pursued
  - lr_meta_stacker_expanded        # M5b LB 0.94891
  - reformulation_lgbm              # M5c A/B horizon-shift, laps-until-pit
  - hgbc_beta_variants              # M5d LB 0.94963
  - catboost_year-cat / lossguide / slow-wide-bag  # 3 CB winners
  - corr_pool_prune / l1coef_pool_prune  # M5g/M5h prune; only L1coef preserves OOF
  - unified_te_2way_keys            # d3a +2.2bp Strat std, +0.1bp stacked NULL
  - sequence_fe_race_driver         # d3b +18bp Strat, +0.2bp stacked NULL
  - tier_break_l1_prune             # M5h2 v1 LB 0.94991 TIE
  - catboost_yetirank_pairwise / gaussian_naive_bayes_mixed  # d4 std weak; stack TIE
  - gbdt_meta_lr_alternative        # d4 slot 2 LB 0.95001 (-4bp)
  - recursive_gbdt_m5q_feature      # d5 +92bp std, K=15 stacks NULL
  - gbdt_meta_k15_recursive         # d5 NULL
  - tabnet_smoke_default_config     # d5 FAIL gate, parked
  - pseudo_label_e3_mvp / 5_base_phase2 / partial_pseudo_m5q_k14  # d5 PASS; K=14 LB regress
  - aux_feature_gbdt_meta           # d6 F5 falsified
  - 2base_recursive_blend           # d6 B falsified
  - rule_residual_l1_base / multi_rule_residual_k18  # d6 F1.1/F1.2; K=18 LB 0.95026 (+2.1bp)
  - simple_math_rule_residual_pool  # d9 9 closed-form; ALL FAIL min-meta
  - hash_lr_3way_baseline / strength_ladder  # d9/d9b L4 K=20 swap LB 0.95025 TIE
  - factorization_machine_cpu       # d9c FM std 0.92069, ρ 0.899; K=20 swap LB 0.95029
  - factorization_machine_partition # d9f K=21 swap LB 0.95031 (+2bp; FM_A ρ 0.487 most-diverse)
  - factorization_machine_aug12     # d9h K=22 add LB 0.95034 (+3bp; 300× upside)
  - factorization_machine_aug2way   # d9i K=21 swap LB 0.95034 (+3bp; OOF predicted regression)
  - factorization_machine_aug15     # d13 H1 std 0.92711 strongest FM ever; K=23 add -2bp REGRESS
  - empirical_bayes_hierarchical_meta  # d13 Path B; Compound×Stint τ=20k LB 0.95049 (+8bp 8× upside)
  - groupkf_stack_rebuild_audit     # d10b/c FM-class +2.01bp GKF vs +0.87bp Strat (2.3× AMP)
  - leak_corrected_lr_meta          # d10d G3 fail flip 0.001; held
  - empirical_bayes_hier_lr_meta    # d13 Stint τ=100k LB 0.95041 (+7bp; 11.6× upside; GKF 2.9× AMP)
  - t12_censored_regression / ratio_target / stintlevel_survival  # d12 4-of-4 FAIL min-meta
  - year_segmented_specialist       # d12 falsified; AV-AUC 0.502 (i.i.d.); 2023 EASIEST
  - adversarial_validation_reweight # d12 -4.92bp min-meta; train/test i.i.d.
  - lambdarank_race_meta            # d12 -86bp REGRESSED (Q6 origin)
  - aucpairwise_xgb_base            # d12 -451bp fold-0 (Q6 origin)
  - single_bag_e3_5seed             # d12 -19bp every segment; K=21 complexity JUSTIFIED
  - groupkf_full_pool_meta          # d12 KEY: ρ(Strat-vs-GKF meta)=0.9914; rank-lock partial dissolves
  - fm_partition_5_3_d13a / 4_4_ct_axis / 6_6_alt  # d13a/d13d K=23-25 noise-floor; partition SATURATED
  - gkf_full_22_stack_d13b          # d13b SWAP_21 redundant; Move C minimal validated under GKF
  - move_c_strat_pool_refactor      # d13c T2/T3 -2.5/-2.6bp Strat FALSIFIED — leak-eaters carry signal
  - within_stint_lgbm_fe / cross_driver_intra_race_lgbm_fe  # d13 G1/G2' min-meta NULL
  - stintgrouped_lambdamart         # d13 G3 killed (63% all-zero stints)
  - fm_aug13_3way_concat_field      # d14 H1 ρ 0.917 most-diverse; min-meta -0.13bp NULL
  - path_b_cohort_sweep_d14         # d14 Year/YxStint/Race × τ; 9 variants ALL <PRIMARY OOF
  - two_level_stacking_meta_as_base # K=22 path B LB 0.95045 -4bp REGRESS (meta-derivative class FALSIFIED)
  - tabpfn_finetune_v25_v26         # Day-14 DEAD; AUC ceiling 0.944; v2.6 OOM P100
  - fm_new_input_features           # Day-14 Move D aug16 std +20bp; min-meta -0.07bp FAIL
  - masked_column_self_prediction   # Day-14 5th per-row-FE NULL; DGP near-independent (load-bearing diag)
  - kd_distillation_lgbm            # 2026-05-06 +0.526bp meta-derivative-class artifact (HELD)
  - nn_with_embedding_layers        # 2026-05-06 ρ=0.918 most-diverse; K=21+1 -0.025 NULL (ρ-alone falsified)
  - lap_mod_features_lgbm           # 2026-05-06 +0.002 NULL; 566bp marginal absorbed by GBDT interactions
  - pseudo_label_confidence_extreme # 2026-05-06 +0.019 NULL
  - within_race_lt_quantile         # 2026-05-06 +0.20 NULL/marginal
  - year_stint_sparse_lr            # 2026-05-06 ρ=0.844; +0.05 NULL
  - blend_aggregators_K21           # 2026-05-06 mean/gmean/rank/trimmed -19 to -32bp; LR meta does real work
  - driver_cluster_path_b_cohort    # 2026-05-06 -0.4 to -0.9bp NULL across τ; cohort axis exhausted
  - alpha_calibrated_tau_resweep    # 2026-05-06 / d15A ρ=1.0 vs d13e; τ=20k empirically optimal
  - id_order_synth_artifact         # 2026-05-06 marginal span ≠ predictive lift (rule of thumb)
  - target_reformulation_invlaps    # 2026-05-06 K=21+1 +1.899bp OOF orig → strict +0.234bp (88% collapse, target-construction-layer-leakage); held submission INVALIDATED
  - target_reformulation_stintprog  # 2026-05-06 ρ=0.252 most-diverse base ever; K=21+1 NULL; same target-construction-layer-leakage
  - multi_target_nn_pit_aux_invlaps # 2026-05-06 +0.086bp NULL
  - path_b_K22_invlaps_compound_stint  # 2026-05-06 OOF +2.75bp orig → INVALIDATED via strict-OOF audit (inv_laps base was leaky); DO NOT submit
  - extra_trees_5fold_d15c          # d15C +0.059bp at ρ 0.99599 noise-floor; R5 HEDGE only
  - knn_distance_lgbm_d15d          # d15D +0.056bp; ρ between C/D 0.9325 raw but LR routes to ρ=0.996; HEDGE only
  - dae_swap_noise_lgbm_d15b        # d15B Jahrer DAE GPU; LB 0.95059 NEW PRIMARY (+1bp; amp 1.4× — friction `path-b-amp-only-fires-on-meta-arch-not-base-add`)
  - d15_orig_transfer               # 2026-05-06 LGBM on aadigupta orig; hier-meta K=22 LB 0.95049 TIE; HEDGE-tier
  - d15_orig_multi_arch_bag         # 2026-05-06 multi-arch on shared training-data REDUNDANT (friction tag)
  - d15_decode_normalized_tyrelife / physics_residual / leak_lookup  # 2026-05-06 3 lookup probes; cheap rule-outs
  - target_reformulation_pit_horizon  # 2026-05-06 4-class horizon; +3.191bp orig → strict +0.302bp (90% collapse, same leakage)
  - target_reformulation_reverse_cum  # 2026-05-06 # remaining pits; +4.867bp orig → strict -0.005bp (100% collapse, BIGGEST leak)
  - target_construction_layer_leakage_audit  # 2026-05-06 friction `target-construction-layer-leakage`; ALL target-reform 88-100% inflated; DO NOT submit any path_b_*invlaps* / *megapool* / *K23_dae_invlaps*
  - target_reform_strict_oof_audit  # 2026-05-06 strict per-fold target construction; 3-of-3 collapse to ≤+0.3bp K=21+1; PRIMARY remains LB 0.95059
  - path_b_multilevel_4tier        # 2026-05-06 4-tier hier-meta on K=22+DAE; 5 (τ_0,τ_1,τ_2) configs ALL NULL -0.16 to -0.79bp; T4a meta-arch FALSIFIED
  - d16_path_b_K22_continuous_only  # Day-16 K=22 Path B continuous_only τ=20k LB 0.95089 (+3.0bp; PRIMARY-replace candidate Day-17+)
  - d16_year_2023_hard_mask        # Day-16 H4 zero-mask 2023×rare-Driver; K=5 ceiling +0.004bp NULL (PRIMARY routes 2023 already)
  - d16_conformal_isotonic         # Day-16 H7 per-bin isotonic; 4 schemes -2.5 to -9.6bp; PRIMARY hier-meta globally calibrated; δ2/δ3 NULL
  - d16_two_stage_stint_logistic   # Day-16 H10 α5 stage-1 E[T_stint] + stage-2 1-D logistic; std OOF 0.625 NULL methodology miss
  - d16_twin_pool_2_meta_blend     # Day-16 H2 ε2 ρ(metaA,metaB)=0.967; -1.79bp FALSIFIED `twin-pool-2-meta-collapses-rank-info`
  - d16_deepgbm_leaf_encoding      # Day-16 ε4/ε4b leaf-indices→head; KILLED both (>16min over-engineered + sparse-LR weak)
  - d16_av_sample_weight_lgbm      # Day-16 H11 ε AV-prob as weight; KILLED 12min on AV stage; EV bounded by AV-AUC=0.502
  - d16_transductive_pseudo_full   # Day-16 H9 ζ6 627k+pseudo half-weight; +0.631bp K=22 LR-meta but -0.30bp vs hier-meta; MARGINAL HEDGE
  - d16_gru_sequence_alpha4        # Day-16 H1 α4 GRU; ρ_test 0.919 most-diverse; -0.043bp NULL; 5th `lr-meta-rank-lock-strong-anchor` confirmation
  - p1_single_lgbm_kitchen_sink    # Day-17 P1 thesis FALSIFIED; v1 LB 0.94107 leaky; v3 fold-safe OOF 0.94563 honest; single-LGBM ceiling ~0.946 = -52bp from PRIMARY OOF (origin Rule R20/R24/R25)
  - chain_decomposition_orig_likelihood  # Day-17 PM d18 v1 (causal+gauss): K=21+1 +7.365 bp (largest single-base of session); v2 (causal+q10) +1.43 bp (modeling-axis 5x); v3 reverse PARKED. Per-step orig-DGP log-likelihood on causal chain (Year→Race→Compound→Stint→LapNumber→TyreLife→RP→Position→LapTime→Delta→CumDeg→PosCh→PitStop). LANDED in K=23 PRIMARY LB 0.95149.
  - dgp_preimage_join_knn          # Day-17 PM E2: kNN(K=10) per-Compound in orig over 7 KS-low feats; 7 aggregate features (preimage_y_mean/std/dist/ntl/match). Std OOF 0.94829, K=21+1 +1.88 bp at ρ=0.9944. PASS.
  - dgp_chain_ll_q5_pathb_cohort   # Day-17 PM E5 c1: Path-B Compound×chain_LL_q5 cohort REGRESS -1.91 to -0.04 bp vs K=22 LR-meta. Friction `chain-ll-q5-cohort-weaker-than-compound-stint`. Disambiguates Phase-5 K=14-pool caveat.
  - d18_path_b_K23_d16_d18_tau20000  # Day-17 PM 🎯 NEW PRIMARY LB 0.95149 (+6.0 bp). K=21+d16+d18 Path-B Compound×Stint τ=20k OOF 0.95184 (highest of session). Realised at optimistic end of band; PI sealed pred +3 bp, agent +5 bp, actual +6 bp.
plateau_days: 0                   # Day-17 PM PRIMARY advance +6 bp via d18 chain-decomp + d16 cont_only stack. Next priorities: meta-arch redesign on K=23 (Student-t shrinkage / Yao-Vehtari Σ-BMA), E4 class-conditional chain (predict-batch optim), E3 CTGAN-replay on Kaggle GPU, v3 reverse-causal chain rerun, Pirelli external scrape.
gate_status: cleared              # d18_path_b_K27_v4h1d_d16_d18_e2_f2_tau100000 LB 0.95368 NEW PRIMARY (Day-17 PM-late); +1.4 bp over main's K=23 v4+h1d 0.95354
headroom_to_top5pct: 0.00037      # 0.95405 − 0.95368 = 3.7 bp on K=27 combined PRIMARY
```

## Calibration ladder

Pre-Day-13 rows archived → `audit/archive-2026-05-06-claude-md-compression.md`.
Active anchors below; cite these for 6-Q questions 3-5 ("predict ρ band /
OOF / closest precedent").

| Mechanism | Strat OOF | GroupKF OOF | LB | Notes |
|---|---:|---:|---:|---|
| baseline_two_anchor | 0.94075 | 0.92059 | 0.94113 | LB-proxy ✓ gap +3.8bp |
| **m5q (M5h + RealMLP, K=14)** | 0.95057 | n/a | 0.95005 | OLD PRIMARY (Day-3); +14bp LB; +1.4bp OOF → 10× amp |
| **d6_k18_multi_rule** | 0.95065 | n/a | 0.95026 | OLD PRIMARY (Day-7); +2.1bp LB |
| **d9c_FM** | 0.92069 | n/a | n/a | ρ 0.899; min-meta +0.18bp PASS; first FM-class anchor |
| d9c_Sd_K20_swap_FM | 0.95070 | n/a | 0.95029 | +3bp LB (5.7× upside); FM-class amp first-fired |
| **d9f_FM_A_driver_dynamics** | 0.82505 | n/a | n/a | ρ vs PRIMARY 0.487 (most-diverse since R14) |
| d9f_K21_swap_partA_partB | 0.95073 | n/a | 0.95031 | +2bp LB (6.25× upside) |
| **d9h_K22_add_aug12** | 0.95073 | n/a | 0.95034 | +3bp LB; 300× upside on +0.01bp pred (TIE→LB-lift) |
| **d9i_S1_K21_swap_aug2way** | 0.95071 | n/a | 0.95034 | +3bp LB; OOF predicted -0.19bp REGRESSION; LB amplified positive |
| d10b_K13_baseline | 0.95043 | 0.92744 | n/a | gap −229.92bp (leakage signature) |
| d10b_K15_+FMA+FMB | 0.95052 | 0.92764 | n/a | FM-class +0.87bp Strat → +2.01bp GKF (2.3× AMP) |
| d10d_leak_corrected_meta | n/a | 0.92764 | n/a | G3 FAIL (flip 0.001); held; pred-LB 0.95001 |
| d13_path_b_compound_tau100000 | 0.95076 | n/a | 0.95033 | +2bp LB on +0.30bp OOF (6.7× upside; first hier-meta) |
| d13_path_b_stint_tau100000 | 0.95082 | 0.94600 | 0.95041 | +7bp LB (11.6× upside); GKF +2.59bp = 2.9× AMPLIFIED |
| **d13e_compound_stint_tau20000** | 0.95083 | n/a | 0.95049 | +8bp LB over Stint; 8× upside; 24-seg cross; OLD PRIMARY |
| d13e_compound_stint_tau100000 | 0.95081 | n/a | n/a | held; +0.82bp OOF; HEDGE-eligible |
| d13b_path_b_stint_tau20000 | 0.95082 | n/a | n/a | held; +0.88bp OOF |
| d13_g1_within_stint / g2_cross_driver | 0.94194 / 0.94250 | n/a | n/a | LGBM-class FE NULL at min-meta |
| d14_h1_fm_aug13_3way | 0.92639 | n/a | n/a | ρ=0.9169 most-diverse; min-meta -0.13bp NULL |
| d14_fm_aug16 (Move D) | 0.92741 | n/a | n/a | +20.1bp std vs aug12; min-meta -0.07bp FAIL (FM-aug saturated 12 fields) |
| d14_tabpfn_v25_150k | 0.94446 | n/a | n/a | DEAD; ceiling 0.944; v2.6 OOM P100 |
| **d15b_lgbm_dae_only (DAE 768d → LGBM)** | 0.94007 | n/a | n/a | ρ_test 0.9477 (most-diverse since FM_A); min-meta +0.793bp |
| **d15b_path_b_K22_dae_only_tau20000** | 0.95090 | n/a | **0.95059** | PRIMARY (Day-15); +1.0bp; flips 59/53; realised amp 1.4× |
| d15c_extra_trees / d15d_knn_lgbm | 0.92967 / 0.94166 | n/a | n/a | min-meta +0.05-0.06bp; rank-lock at ρ≈0.996; R5 HEDGE only |
| **d16_path_b_K22_continuous_only_tau20000** | n/a | n/a | **0.95089** | OLD-PRIMARY (Day-16, +3.0bp); clean Path-B base-add; superseded by d18 K=23 |
| target_reform_strict_oof_audit | n/a | n/a | n/a | Day-17 friction `target-construction-layer-leakage`: ALL invlaps/pit_horizon/reverse_cum collapse 88-100% under strict OOF |
| d18_chain_decomp (v1 causal+gauss) | 0.94954 | 0.9914 | n/a | Day-17 PM K=21+1 **+7.365 bp** (largest single-base of session); per-step orig-DGP log-likelihood on causal chain |
| **d18_path_b_K23_d16_d18_tau20000** | 0.95184 | 0.9923 | **0.95149** | **NEW PRIMARY (Day-17 PM, +6.0 bp)**; K=21+d16+d18 Path-B Compound×Stint τ=20k; PI sealed pred +3 bp, agent +5 bp, actual +6 bp |
| p1_single_lgbm_v3_fold_safe | 0.94563 | n/a | n/a | Day-17 honest single-LGBM ceiling; -52bp from PRIMARY OOF; stacking justified |

## Hypothesis board (Day-17 AM)

Day-9→Day-12 DONE history archived → `audit/archive-2026-05-06-claude-md-compression.md`.

```
- INSIGHT (Day-16): friction `path-b-amp-only-fires-on-meta-arch-not-base-add`
  PARTIALLY INVALIDATED. d16 K=22 Path B continuous_only τ=20k LB 0.95089
  (+30bp) is a clean BASE-ADD (Rule 24 fold-safe) that DID fire amp. The
  Day-15 friction tag was a leakage artifact, not a mechanism truth.
- INSIGHT (Day-17): Rule 24/25 leakage-audit pass invalidated 3 target-reform
  "wins" (invlaps/pit_horizon/reverse_cum) — 88-100% of OOF lift was
  target-construction-layer leakage. Genuine signal ≤+0.3bp K=21+1.
  All path_b_*_invlaps_* candidates DO NOT submit.
- INSIGHT (Day-12 unifying frame): K=21 stack works because LR-meta
  routes between leakage-eating GBDTs (high Strat AUC, real LB AUC much
  lower) and leakage-robust FM/rules (Strat AUC ≈ GKF AUC). Public LB
  row-iid (U3) so PRIMARY survives. Diversification needed is WITHIN
  the leakage-robust population.
- INVALIDATED (Day-17): `path_b_K22_invlaps_tau20000` and the entire
  `path_b_K23_dae_invlaps_*` / `path_b_K25_megapool_*` family. All built
  on target-construction-layer-leaky bases. DO NOT submit.
- LATER:
  (a) Meta-arch redesign — non-Gaussian shrinkage prior (Beta-Binomial /
      Student-t); Yao/Vehtari covariance-modelled BMA; alternative seg
      crosses (Year×Compound, Compound×TyreLife_q5, Driver_clustered×Stint).
      Highest tail EV; only Path-B-amp-eligible axis per friction tag.
  (b) External-data Pirelli pit-window scrape (Tier-2; ISSUES.md leaf 4a).
  (c) Tier-2 target reformulations (pit_horizon_multiclass; reverse
      cumcount of pits; stint_index_within_race) — `probe_target_reform.py`.
  (d) Pool-composition surgery — STRUCTURED replace (drop 2 leak-eaters
      AND add 2 target-derived). d13c falsified naive drop.
  (e) GroupKF-meta as R5 HEDGE (don't submit as PRIMARY; public is row-iid).
- TabPFN v2.5/v2.6 — DEAD (Day-14; AUC ceiling 0.944; v2.6 OOM P100).
- DAE base — LANDED (LB 0.95059 +1bp NEW PRIMARY). DAE variants in
  Tier 3 (mask-noise / stacked-2-layer / CatBoost-on-latent / DAE-on-OOFs).
- HEDGE ladder accumulating: d13e τ=100k, d15c ExtraTrees, d15d KNN-LGBM,
  d15c+d15d K=23, path_b_K22_invlaps τ=100k.
```

## Pointers

Pre-Day-12 pointers archived → `audit/archive-2026-05-06-claude-md-compression.md`.

- `HANDOVER.md` — next-session brief (Rule 15).
- `WRAPUP.md` — wrap-up + prepare-handover procedure (Rule 17).
- `ISSUES.md` — live problem decomposition / claim board (Rule 18).
- `comp-context.md` — settled-once facts.
- `audit/friction.md` — friction one-liners (read top of file each session).
- `knowledge-base/README.md` — KB scaffold (PI second-brain; concepts/thoughts/friction/questions/flags).
- `knowledge-base/concepts/agentic-kaggle-systems-comparison.md` — agentic-loop research synthesis + tips for PI (origin of Rule 26).
- `knowledge-base/concepts/decision-time-logging.md` — `audit/decisions.jsonl` rationale.
- `knowledge-base/concepts/decision-quality-vs-outcome-quality.md` — postmortem framing.
- `.claude/skills/postmortem/SKILL.md` — wrap-up postmortem skill (Rule 17 step 4b).
- `audit/2026-05-06-target-reform-leakage-audit.md` — Day-17 strict-OOF leakage finding (origin of Rule 24).
- `audit/2026-05-12-d12-master-synthesis.md` — Day-12 6-option overnight synthesis (4 falsified, 1 structural finding).
- `audit/2026-05-12-d12-groupkf-rebuild.md` — Option 1 STRUCTURAL FINDING (rank-lock dissolution under GKF).
- `audit/2026-05-13-d13-path-b-hier-meta.md` — Path B empirical-Bayes Stint τ=100k +0.86bp OOF; held.
- `audit/2026-05-13-d13d-path-b-gkf-probe.md` — Path B GKF; Stint lift +0.90bp Strat → +2.59bp GKF (2.9× AMP).
- `audit/2026-05-15-d15-4branch-results.md` — Day-15 PM 4-branch deep-dive + DAE + new PRIMARY 0.95059.
- `audit/2026-05-06-d14-dgp-residuals.md` — masked-column self-pred NULL + DGP near-independent diagnostic.
- `audit/2026-05-06-d15-decode-synthesizer.md` — orig_transfer K=22 hier-meta LB 0.95049 TIE (HEDGE-tier).
- `scripts/hypothesis_view.py` — graph view of mechanism_families_explored
  grouped by axis; `--axis target_reform` / `--status alive` filters; `--json
  data/hypothesis_graph.json` for JSON dump (gitignored, regenerable).
- `scripts/dispatch_branches.py` — parallel-branch dispatcher; reads ISSUES.md,
  emits 6-Q-templated brief for N picks (Rule 18 / 19e).
- `scripts/probe.py` / `probe_min_meta.py` / `research_seed.py` — harness (Rule 19).
