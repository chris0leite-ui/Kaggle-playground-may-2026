# Friction log

One-liners. Distilled weekly per `~/.claude/skills/kaggle-comp/self-improvement.md`.

## 2026-05-07

- `tag: feature-subset-orig-transfer-passes-where-arch-bag-fails` —
  d16 Phase 4: 4 variants of orig-trained LGBM on different feature
  subsets. K=21+1 gates: continuous_only +3.33bp (LARGEST single-base
  K=21+1 of session, beating inv_laps +1.90 by 1.75×), no_laptime
  +1.87, no_tyrelife_rp +0.86, categorical_only PASS via meta-stack.
  Mechanism: orig-LGBM restricted to features the synthesizer left
  marginal-aligned (TyreLife KS=0.017, Position KS=0.019, Position_Change
  KS=0.015) generalises to synth far better than full-feature orig
  (which uses heavily-corrupted LapNumber KS=0.188, Stint KS=0.175,
  RaceProgress KS=0.186). **Refines** `external-data-arch-bag-redundant-when-shared-training-data`:
  arch variation IS redundant; FEATURE-SUBSET variation is NOT, because
  each subset emphasises a different region of the orig DGP. Pre-flight
  for new orig-data probes: vary FEATURES, not architecture. Phase-1
  KS-divergence diagnostic literally guided the discovery: the marginal-
  aligned features the synth preserved are the lever for transfer.

- `tag: density-ratio-routes-or-weights-but-fails-as-feature` — d16
  Phase 2: r̂(x)=p_synth/p_orig used three ways. As single feature K=21+1
  NULL (-0.07 bp). As sample weight in mixed-source training (P2.3)
  K=21+1 +0.78 bp PASS. As cohort router for segment-calibrated orig
  (P2.4) K=21+1 +1.32 bp PASS. **Lesson**: density ratio carries
  across-distribution information but it's not pointwise predictive;
  the productive uses are (a) re-weighting and (b) routing. CRITICAL
  prerequisite: exclude high-cardinality cats (Driver, Race) from the
  classifier. v1 with Driver included hit AUC 0.9985 from 856 ghost-
  Driver tells; r̂(x) saturated at clip ceiling and was useless.

- `tag: rho-alone-insufficient-for-meta-utility` — 4th independent
  confirmation. d16 Phase 3 GMM single-feat: ρ vs PRIMARY 0.503 (most-
  diverse single base ever measured, beating d9f FM_A 0.487 and
  d15_orig_transfer 0.5653) but K=2 gate -0.10 bp NULL. Joins
  nn_embeddings (ρ=0.918 NULL), year_stint_sparse_lr (ρ=0.844 NULL),
  stint_progress (ρ=0.252 NULL). The K=21 LR-meta with [raw,rank,logit]
  expand absorbs high-diversity-low-information bases as convex combos.
  Already codified in `scripts/probe.py FAMILY_PRIORS`; this just adds
  a fourth instance.

- `tag: bgmm-default-oversmooths-at-reg-covar-1` — sklearn
  BayesianGaussianMixture with reg_covar=1.0 (set after v1 crashed at
  reg_covar=1e-3 with ill-defined empirical covariance) over-smoothed
  the orig joint. BGMM single-feat AUC 0.55 (near-random) vs GMM 0.76
  on the same data. ρ(GMM, BGMM) on synth_train = 0.81 — they correlate
  but BGMM has lost most of the predictive structure. **Fix**: don't
  use sklearn BGMM as a drop-in for GMM on this joint without a careful
  reg_covar sweep; or use a proper VI implementation (numpyro / pyro).

- `tag: path-b-on-pool-subset-conflates-cohort-axis-with-pool-size` —
  d16 Phase 5 ran Path B on a K=14 sub-pool (only 14 of 21 named bases
  existed under exact filenames). All r̂_q5 / logp_q5 cohort axes
  regressed -3 to -4 bp vs PRIMARY (which uses K=21). Cannot cleanly
  attribute the regression to cohort-axis failure vs missing-bases
  artifact. **Fix**: when probing alternative cohort axes, the pool MUST
  match PRIMARY exactly — load OOFs by file-glob with matching shapes,
  not by named list. Re-test Phase 5 cohort-axis on full K=21 to
  disambiguate.

## 2026-05-06

- `tag: synthetic-dgp-conditionally-near-independent` — Day-14 PM:
  d14 DGP-residuals probe (masked-column self-prediction; SAINT/
  TabNet/VIME class). Trained 4 LGBM regressors to predict
  LapTime_Delta / Cumulative_Degradation / Position / LapNumber
  from the rest of the row. **Across all 4 targets, OOF RMSE ≈
  marginal σ within 3 sig figs** (LapTime_Delta 41.05/41.06;
  CumDeg 34.94/34.97; Position 3.491/3.491; LapNumber 1.559/1.559).
  Conditional-given-rest variance ≈ marginal variance — the synthetic
  NN-DGP added near-independent per-feature noise within rows.
  Stack outcome: standalone OOF 0.94200 (Δ −88bp vs PRIMARY); K=2
  min-meta −0.025bp NULL; K=22 add +0.172bp at ρ=0.996 (pred LB
  −1.3bp under harness band). Family closed.
  **Lesson**: this is the 5th independent NULL of the same axis
  (Day-13 G1/G2'/G3, Day-14 H1/Move-D, now d14 DGP-residuals): the
  K=21 + Path-B-hier-meta has fully absorbed every cross-feature
  signal extractable from the synthetic DGP within a single row.
  Per-row feature engineering / self-supervised pretraining cannot
  break the ceiling — only meta-layer / model-class / external-data
  innovations can. Joint-explains FM-aug12 saturation, Move D NULL,
  Day-13/14 alt-axis 4-of-4, and TabPFN's 0.944 ceiling.

- `tag: rho-alone-not-sufficient-for-meta-utility` — measured 13+
  single-base candidates this session. Three with very low ρ vs
  PRIMARY (extreme diversity) all NULL at meta gate: nn_embeddings
  (ρ=0.918, +0.025 bp NULL), year_stint_sparse_lr (ρ=0.844, +0.05 bp),
  stint_progress alone (ρ=0.252, NULL). The K=21 LR meta with
  expand([raw,rank,logit]) reproduces high-diversity bases as
  convex combinations of pool when the test-time-distinct signal
  is linearly recoverable. **Fix:** when BOTE-ing single-base
  additions, do NOT credit ρ < 0.95 as predictive of meta lift.
  Instead require evidence that the candidate's signal is sourced
  from *outside* the pool's prediction span (e.g., target-derived
  reformulation like `inv_laps_until_pit`, NOT model-class diversity
  alone). Codify this in `scripts/probe.py FAMILY_PRIORS` —
  `nn_or_fe_diversity_alone` family with P=0.10, bp band (0, 0, 1).
- `tag: marginal-bin-span-not-predictive-lift` — id-order audit found
  LapNumber_mod_10 marginal target span 566 bp; lap_mod_features
  LGBM with explicit mod features got K=21+1 +0.002 bp NULL.
  The 566 bp marginal pattern was fully captured by existing GBDT
  feature interactions (LapNumber × other features). **Fix:**
  marginal-bin-span findings are NOT a reliable EV proxy for
  predictive lift; need a "joint-model holdout" check (refit LGBM
  WITHOUT the candidate feature; measure ΔAUC; if ΔAUC < 0.5 bp
  the candidate adds nothing).
- `tag: target-derived-vs-meta-derived-orthogonality` — empirical
  separation now established: `d12_lr_meta` (= K=21 LR-meta-OOF)
  produced +1.348 bp OOF but LB regress -4 bp; `inv_laps_until_pit`
  (LGBM regression on 1/(1+laps_until_pit), target-derived) produced
  +1.899 bp OOF. Both have similar ρ (~0.99) and superficially
  similar OOF lift, but the orthogonal-signal criterion separates
  them mechanistically. The target-derived candidate has not yet
  been LB-tested but should NOT be subject to the meta-derivative
  failure mode. **Pre-flight rule:** before treating any K=K_pool+1
  candidate as Path-B-amp-eligible, classify the candidate's signal
  source: (a) target-derived (PASS), (b) feature-engineered from
  raw inputs (PASS conditional), (c) meta-derivative / convex-combo
  of existing pool predictions (FAIL — discount Path B amp by 10×).
- `tag: cpu-contention-multi-probe-batch` — running 7 LightGBM/NN
  probes simultaneously made each ~4× slower than alone. KD probe
  reached `max_iter=400` without early-stop on any fold; FE-combo
  was killed after fold 0 took 62 min. **Fix:** Cap to 3 concurrent
  CPU-heavy probes max in future batches. Schedule cheap probes
  (<30 s each) ahead of slow ones to free CPU for the slow batch.

- `tag: path-b-amp-only-fires-on-meta-arch-not-base-add` — Day-15
  d15b_path_b_K22_dae_only_tau20000 SUBMITTED at LB 0.95059 (+1.0bp
  NEW PRIMARY) on +0.715bp OOF — realised amp 1.4×, well below Path B
  amp 6-11.6× central. Cross-confirmed by parallel main-branch agent
  same day: K=22 + orig_transfer base-add LB 0.95049 (TIE at +1.127bp
  OOF). The amp pattern (d13 Compound 6.7×, Compound×Stint 8×, Stint
  11.6×) is conditional on the LIFT being a meta-architecture redesign
  (e.g. segmentation refinement Stint→Compound×Stint), NOT on
  K_pool→K_pool+1 base additions even when the new base is genuinely
  orthogonal-class (DAE ρ_test 0.9477 standalone, Jahrer Porto-Seguro
  precedent). Base-additions get standard ρ-band treatment per
  probe.py predicted_lb_delta_bp, with a small positive deviation from
  the diverse new-class signal. Refines `path-b-amp-needs-orthogonal-
  signal-not-meta-derivatives` (which excluded meta-derivatives from amp
  eligibility): even a true orthogonal base-add does not fire amp.
  Lesson: Day-16 priority should be META-ARCH REDESIGN candidates
  (non-Gaussian shrinkage prior on hier-meta, Yao/Vehtari covariance-
  modelled BMA, alternative segmentation cross like Year×Compound or
  Compound×TyreLife_q5), not more orthogonal base-add candidates.
  Pre-flight rule: classify candidate as `meta_arch_redesign` (amp-
  eligible) or `pool_addition` (ρ-band only); discount EV for the latter
  by ~3-5× regardless of standalone OOF diversity.

- `tag: kaggle-p100-fallback-reproduced-day15` — 8 days after first
  encountered (Day-3 RealMLP), pushed Day-15 d15b-dae-lgbm-gpu kernel
  with `machine_shape: GpuT4x2` per the prior friction lesson; v1 ran
  on P100 (sm_60) anyway, ERRORed at first GPU op
  (`cudaErrorNoKernelImageForDevice`). The friction tag was internalised
  but the FIX was not pre-applied — I should have copy-pasted the
  torch 2.4 force-reinstall pattern from realmlp_gpu.py into the new
  kernel BEFORE pushing v1. Cost: one wasted Kaggle queue + ~5 min
  diagnose + push v2. Skill amendment: when creating any new torch-
  based GPU kernel, START from `kernels/hazard-nn-smoke-gpu/` boilerplate
  (force-reinstall already wired in), don't fork from a kernel that
  doesn't have the fix. Add to do-and-dont.md GPU template list.

- `tag: subagent-friction-4-of-4-recurrence` — Day-15 4-branch parallel
  probe: ALL FOUR dispatched general-purpose subagents fired the same
  pre-existing frictions `subagent-monitor-truncation` /
  `subagent-non-execution`. Subagent A wrote a script and exited;
  Subagent B ran smoke for 30+ min then exited mid-LGBM; Subagent C
  ran fold 0 then exited; Subagent D wrote KNN feature script then
  exited mid-LGBM-fold-0. Main thread had to relaunch Branches A and
  D and supervise C/B's tail. Lesson: the friction tag is well-known
  but I keep dispatching general-purpose agents for long-running
  python jobs. Cost: 2× wall (subagents wasted ~30-60 min each before
  main-thread takeover). **Permanent fix: do NOT dispatch
  general-purpose subagents for `python script.py >log` jobs that
  exceed ~5 min wall.** Spawn the python directly from the main thread
  via Bash run_in_background=True. Document in skill.

- `tag: path-b-amp-needs-orthogonal-signal-not-meta-derivatives` —
  Path B family-conditional amplification (prior precedents: Stint
  +0.86 bp OOF → +7 bp LB at 11.6×; Compound×Stint +1.0 bp OOF →
  +8 bp LB at 8×) DID NOT FIRE on a K=22 add of `d12_lr_meta`
  (the K=21 LR-meta-OOF itself). +0.99 bp OOF predicted, actual LB
  −4 bp (LB 0.95045 vs PRIMARY 0.95049). Cause: `d12_lr_meta` is a
  convex combination of the same 21 base predictions already in
  the K=21 pool — adding it as a 22nd "base" creates a 2-level
  stack but no orthogonal signal flow into the segment routers.
  The hier-meta gains OOF AUC by exploiting fold-structure of the
  inner-meta-OOF that does NOT survive at test-time (where the
  inner meta is fit on full-train, not per-fold).
  **Lesson:** Path B amplification is conditional on the K_pool
  addition carrying new/orthogonal SIGNAL (FM-class diverse vs
  GBDT-class; sparse-LR diverse vs dense-LR; etc.), NOT on adding
  a meta-derivative whose information is already in the pool.
  Pre-flight rule: before treating a K=K_pool+1 candidate as
  Path-B-amp-eligible, classify the candidate signal axis vs the
  existing K_pool axes. Meta-derivative / convex-combo additions
  → discount Path B amp by 10× (treat as `tuning_existing` not
  `meta_arch_redesign`). Add a `two_level_stacking` family in
  `scripts/probe.py FAMILY_PRIORS` with P(useful)≈0.10 and
  bp band (0, 0, 1) so BOTE catches this in the future.

- `tag: external-data-arch-bag-redundant-when-shared-training-data` —
  branch `decode-synthetic-data-uoPIn` 2026-05-06: trained 4 model
  classes (LGBM-default, CB, XGB, LGBM-tuned) on the same
  aadigupta1601 original 99k rows. Inter-arch ρ on synth test 0.94-0.99
  (high overlap); ρ vs PRIMARY single-row 0.57-0.64. Hier-meta K=23
  adding CB Δ +0.005bp NULL; K=24 adding XGB Δ +0.33bp but flips 293
  > R7 200-cap. **Root cause**: same training data → architectures
  share most of the underlying DGP signal; LR-meta absorbs
  architectures redundantly. **Fix**: vary training-data subset
  (e.g. mixed-source with weights) or target-engineering (e.g.
  laps-until-pit regression instead of next-lap classification),
  not just architecture. Add a `bag_same_training_data` discount
  in `scripts/probe.py` FAMILY_PRIORS (-50% on per-arch increments
  beyond the first).

- `tag: meta-arch-required-for-orthogonal-base-eval` —
  branch `decode-synthetic-data-uoPIn` 2026-05-06: submitted K=22
  LR-meta + d15_orig_transfer → LB 0.95039 (-10 bp regress vs PRIMARY
  hier-meta 0.95049). The OOF lift +0.778 bp under LR-meta(K=22) was
  REAL but landed below PRIMARY's hier-meta architecture floor.
  **Root cause**: pre-submit BOTE quoted "+0.778 bp" but didn't
  specify which meta architecture would be used to evaluate; LR-meta
  vs hier-meta = ~14 bp delta on this comp, dominating any +0.5-1 bp
  base-add gain. The follow-up hier-meta(K=22) probe lifted +1.127 bp
  OOF and landed LB 0.95049 TIE, confirming the mechanism but
  validating the meta-arch confound. **Fix**: any new-base BOTE must
  specify both `family` AND `eval_meta_arch` (LR or hier-meta-Cmpd-Stint).
  Add `eval_meta_arch` to `scripts/probe.py bote()` signature.

- `tag: lb-quantization-floor-defeats-decoded-data` —
  branch `decode-synthetic-data-uoPIn` 2026-05-06: d15_orig_transfer
  hier-meta(K=22) lifted +1.127 bp OOF at ρ=0.998 vs PRIMARY → LB tie
  (5-decimal display 0.95049 == 0.95049). At ρ ≥ 0.998 vs the on-LB
  reference, even +1 bp OOF lifts land within Kaggle's ~5 bp public-LB
  resolution. **Root cause**: synth public LB is row-iid (U3 confirmed)
  with 188k test × 20% public split = 37k scoring rows; AUC resolution
  ≈ 1/(N_pos × N_neg)^0.5 ≈ 5e-5 floor. Decoded-data signals on this
  comp are real but bounded by public-LB granularity. **Fix**: don't
  consume LB submit slots on candidates predicting <2 bp at ρ ≥ 0.998
  — they're calibration probes, not lift candidates. Update Rule 12
  to "spend slots on predicted ≥3 bp candidates only". For sub-2-bp
  candidates, hold as HEDGE/R5 final-window pool.

## 2026-05-13/14

- `tag: single-base-fe-additions-noise-wall` — Day-13/14 alternative-axis
  branch ran 4 candidates (G1 within-stint LGBM FE, G2' cross-driver
  LGBM FE, G3 stint-grouped LambdaMART, H1 FM aug13 CTRq 3-way) chosen
  from probe + EDA findings to span (LGBM, FM) × (relative FE,
  cross-row FE, 3-way concat). All 4 hit min-meta zero/negative
  despite 0.92-0.97 ρ vs PRIMARY (very high disagreement). H1 was
  the closing data point — it ALSO failed the gate even with +9.9bp
  standalone OOF lift over d9h_aug12. **Lesson: Path B's K=21 +
  hier-meta has absorbed signal from any single new base built on
  existing-class new features.** Future +bp axes are (1) genuinely
  new model classes (TabPFN), (2) further meta-layer innovations,
  (3) target reformulation upstream of the K=21 pool, (4) external
  data revisit. **Process fix:** before running ≥30 min compute on a
  new candidate, run a 10-min standalone-vs-PRIMARY ρ-and-min-meta
  spot check on a SUBSAMPLE (e.g. 100k rows, 1 epoch FM / 200 LGBM
  rounds). All 4 candidates would have shown low EV at the spot
  check; would have saved ~2 wall-hours over 4 candidates.
- `tag: torch-not-in-requirements` — bootstrap.sh installs from
  requirements.txt which doesn't pin torch, but FM scripts (d9c, d9f,
  d9h, d9i, d13_move_b, d14_h1) all `import torch`. First run of
  d14_h1 errored on `ModuleNotFoundError: No module named 'torch'`.
  Cost: one extra round-trip + `pip install torch` (~30s wall, ~3GB
  download). **Fix:** add `torch` to `requirements.txt` so future
  bootstraps don't require manual install.

## 2026-05-12

- `tag: multi-agent-handover-collision` — Day-12 session on
  `claude/nn-design-options-NjDZ0` ran in parallel with
  `claude/math-heuristics-ml-62fpM`. While I worked on TabM v3 +
  strategy critique + HANDOVER updates, the parallel agent landed
  d9c → d9d → d9e → d9f → d9g → d9h on origin/main (≈ one
  falsification or PASS per ~30 min). **My HANDOVER updates were
  stale before every push.** Rewrote HANDOVER.md 5+ times in one
  session; 4 merge conflicts on HANDOVER.md, all manually resolved.
  Conflict surface: HANDOVER.md and CLAUDE.md (state block); audit
  + script files do NOT conflict because of unique
  `<day>-<probe-letter>-*.md` naming. Cost: ~45 min repeated
  HANDOVER rewrites + 5 force-rebuilds + push-rejected loops.
  Three coordination options, by infrastructure cost:
  (a) **HANDOVER ownership / scribe-of-the-day** — PI designates
      ONE agent per session as HANDOVER scribe; other agents commit
      audits/scripts only and let the scribe consolidate at EOD,
      not mid-flight. Lowest cost; relies on social contract.
  (b) **Append-only log** — HANDOVER becomes a chronological feed
      where each agent appends `## Day-N PM <agent> — <summary>`
      sections; daily scribe pass folds them into the structured
      "## Day-N+1 morning" brief. No conflicts on disjoint append
      sections; needs scribe role anyway.
  (c) **Lock file** — `touch HANDOVER.lock` before edit; agents
      poll. Higher infra cost; race conditions on the lock itself.
  Recommendation for the remaining 15 days: (a) — PI designates
  scribe per session; non-scribe agents commit audits + scripts
  but skip HANDOVER edits. End-of-day reconciliation is one merge
  per agent, not five. Friction file itself is 358 lines (over 150
  cap from CLAUDE.md Rule 9) — separate issue; needs weekly
  distillation per the skill.

## 2026-05-08

- `tag: menu-overcrediting-redundant-mechanism` — Strategic-menu
  research synthesis (`audit/2026-05-08-strategic-menu-wider-steps.md`)
  recommended T1.5 (Deotte L2 stacking) and T1.2 (multi-formulation
  L1) as Tier-1 candidates. Both were predictably redundant: T1.5 is
  a meta-only change (Day-3 endgame says "LR with [raw,rank,logit]
  is genuinely the right stacker"), and T1.2 IS already in the pool
  via `a_horizon` (horizon-shift) + `b_lapsuntilpit` (laps-until-pit).
  T1.3 (Q12 single-rule rule_residual) was rank-lock-vulnerable per
  the 4× prior confirmation pattern. Cause: research agents proposed
  candidates from general SOTA / Deotte writeups without cross-
  checking against `mechanism_families_explored` ledger or load-
  bearing day-N-endgame audits. Fix: pre-flight 5-question check
  (see CLAUDE.md Rule 16) — match against the ladder, classify
  mechanism vulnerability, predict standalone OOF + ρ vs PRIMARY,
  cite the closest gate-PASS/FAIL precedent. Apply 0.3× EV
  downgrade for {meta-only, rule_residual-on-raw, formulation-
  already-in-pool}. Total cost of today's friction: ~14 min CPU
  + 3 menu items demoted.

## 2026-05-04

- `tag: stats-error` — Pre-baseline gate audit reported "PitStop ↔
  PitNextLap match rate 0.724 → strong structural relationship".
  Wrong: independent-baseline match rate at priors 0.136 and 0.199
  is 0.719. Observed 0.724 ≈ chance. U2 single-feature OOF AUC for
  `lead_PitStop` is 0.512 (basically random). Correction: don't
  flag a "match rate" as a structural finding without comparing
  against the independent-baseline expectation. Add to
  pre-baseline-gate.md item 2 ("schema check") a step:
  "for any binary-vs-binary correlation claim, report observed vs.
  independent-baseline match rate; the EXCESS is the signal."

- `tag: cv-anchor-context` — Auto R1 verdict ("gap >50bp ⇒ leakage")
  fired on baseline_two_anchor (gap 200bp), but that conclusion was
  wrong given U3 (test is i.i.d. row split). R1's rule needs a
  qualifier: "leakage" interpretation requires that the test set's
  generalisation regime matches anchor B; if test is i.i.d. row
  split (verifiable via U3-style alt-ratio probe), anchor A is the
  LB proxy and the gap is in-stratum signal, not leakage. Fix:
  update metric_notes default in pre-baseline-gate.md to require
  U3-equivalent split-structure check before interpreting R1 gap.

- `tag: subagent-monitor-truncation` — Subagents that launch python
  via the Monitor tool and rely on completion notifications
  return prematurely with truncated messages ("Monitor armed",
  "I'll wait for completion notification"). The agent's completion
  event fires before its child process finishes; artifacts are
  half-written or absent. Fix: subagent contract must specify
  "run python > log 2>&1; wait for exit; read log; summarize".
  Forbid Monitor + early-exit pattern in agent prompts.

- `tag: tail-pipe-buffering` — `python script.py | tail -40` buffers
  ALL output until the pipe closes. On timeout-kill nothing reaches
  the log; only "Terminated" emerges. Fix: for any long-running
  job, redirect to file directly (`python script.py > log 2>&1 &`)
  and tail the file separately. Never pipe-tail a script you might
  need to debug mid-run.

- `tag: pandas-groupby-rolling-lambda` — `df.groupby(K).transform(
  lambda s: s.rolling(window=W).mean())` HANGS on ~14k+ groups
  (M4 RelState probe sat 17 min before being killed). pandas
  invokes the lambda once per group; per-group rolling
  materialisation is O(n_groups × group_op). Fix: use
  `groupby(K).rolling(W).mean().reset_index(level=K, drop=True)
  .reindex(df.index)` directly — single vectorised pass. Add to
  do-and-dont.md as anti-pattern.

- `tag: probe-extrapolation-drift` — M2 XGB 1-fold probe estimated
  48s/fold; full 5-fold both-anchor took >1200s and timed out
  (~5× projection error). Likely cause: high-cardinality native
  categorical (Driver=887) interacts non-linearly with depth=8.
  Fix: when high-card cats present, multiply 5-fold projection by
  2-3× safety factor before deciding "fits in 1h". Better: add a
  "1-fold-actual" gate (per new PI rule) — decide based on what
  one full-data fold actually does, not a downsampled probe's
  extrapolation.

- `tag: schema-grep-before-FE` — M4 RelState scoped 6 features
  from cross-comp research; 4 of 6 (Position_Change, LapTime_Delta,
  RaceProgress, Cumulative_Degradation) ALREADY existed in the
  dataset. Re-deriving them was no-op; net-new lift came from only
  2 features. Fix: every FE candidate must first be checked
  against `train.columns` and the data dictionary in `brief.md`
  / `comp-context.md`. Add step to do-and-dont.md.

- `tag: kaggle-kernel-metadata-bools` — `kaggle kernels init`
  template emits string-quoted booleans (`"is_private": "true"`,
  `"enable_gpu": "false"`). The CLI silently accepts these but
  Kaggle treats string `"true"` as `false`, so `enable_gpu: "true"`
  → GPU never allocated; `enable_internet: "false"` → actual
  no-internet. First push of cb-slow-wide-gpu wasted 2 retries on
  data-mount failure caused by this. Fix: ALWAYS edit the template
  to use bare booleans (`true`, `false`) — pull a known-working
  prior kernel (`kaggle kernels pull <user>/<slug> -m`) for
  reference. Add to do-and-dont.md anti-pattern list.

- `tag: kaggle-input-rglob` — Comp data path under `/kaggle/input/`
  varies (`/kaggle/input/<slug>/`, `/kaggle/input/competitions/<slug>/`,
  or via attached private dataset). Hardcoding `/kaggle/input/<slug>/`
  fails ~30% of pushes. Pattern from irrigation-catboost-v2-gpu:
  `train_path = next(Path('/kaggle/input').rglob('train.csv'))`.
  Always use rglob for kernel data discovery. Add to
  examples/kernel-template.md.

- `tag: pgrep-heredoc-match` — `while pgrep -f "scripts/X.py" do
  sleep` patterns in pipeline-orchestrator bash scripts matched the
  outer Bash shell that contained the heredoc with the script
  source string. Pipelines never advanced. Hit twice in one session
  (E5→A/B/M5c chain and β→ζ→M5d chain). Fix: don't use heredoc-
  embedded script-source pgrep; either (a) wait on PID directly,
  (b) check for file-output presence, (c) write the pipeline as a
  detached file invoked by path-only.

- `tag: lesson-not-applied` — Logged tail-pipe-buffering friction
  early (M2 XGB), then made the SAME mistake immediately after on
  E2 L1-meta script (also `| tail -50`). The skill amendment was
  documented but not internalised in the same session. Meta-fix:
  when a friction is logged, IMMEDIATELY apply it to all
  in-flight or about-to-launch invocations; don't leave the lesson
  for the next session. Also: pgrep-heredoc-match was a related
  same-session repeat (built two near-identical broken pipelines).

- `tag: hgbc-cat-cardinality-cap` — sklearn HGBC raises
  `ValueError: Categorical feature 'Driver' is expected to have a
  cardinality <= 255 but actually has a cardinality of 874`.
  Surprised E3 first run; fix: label-encode high-card cats as
  numeric int, keep low-card (≤255) as `category`. Document in
  do-and-dont.md as HGBC-specific gotcha.

- `tag: pool-redundancy-gap-widen` — Added β HGBC variants (deep,
  shallow) to M5d pool. Standalone OOF ≈ E3 (~99% correlated).
  M5d Strat OOF +2.3bp over M5c, but LB gap WIDENED from −3.5bp
  (M5b) to −6.0bp (M5d). Adding redundant bases inflates OOF
  beyond LB transfer. Fix: gate new pool additions by pairwise
  correlation against existing pool members (drop ρ ≥ 0.97).
  Documented as Day-3 H3.

- `tag: premature-day-close` — When PI said "I say the day is done"
  at 2/5 slots used, I started the EOD wrap. PI then redirected
  ("submit already in the meantime") and I had to recompute. The
  initial wrap was wasted. Fix: when a PI EOD signal is ambiguous
  (especially when slots remain), confirm intent before starting
  the wrap. Skill amendment now distinguishes "PI pause" from "PI
  irrevocable EOD" — when in doubt, ask once. (Note: this is
  different from auto-recognition once the day-end is unambiguous.)

- `tag: slot-confirmation-loop-friction` — Repeatedly asked PI to
  confirm submit slot even after Rule 12 ("use all 5/day") was
  established. Rule 1 (single-shot, PI-approved) gates the SUBMIT;
  Rule 12 gates the BUDGET. Conflated them. Fix: ask only
  "which candidate for slot N?" not "should I submit slot N?"
  unless the candidate itself is in question.

- `tag: subagent-non-execution` — S1 subagent for M2 XGB wrote the
  full script but never executed it before returning "I'll wait
  for the monitor to fire" (truncated/incoherent). Distinct
  symptom from `subagent-monitor-truncation`: here the python
  process was never started at all. Fix: subagent contract must
  REQUIRE direct execution + log read + summary in one tool call,
  not delegate to Monitor and exit early.

- `tag: rule2-smoke-skip-realmlp-day3` — Day-3 launched RealMLP-TD
  full 5-fold on Kaggle T4 without first running a 1-fold smoke
  probe. Kernel ran 175 min total. Two prior versions failed in
  ~40s on P100 sm_60 (different issue), but the full run had no
  smoke gate behind it. Cost: 175 min Kaggle GPU quota; if the
  full run had been 5h instead of 3h we'd have learned that only
  after burning 5h. Fix: codify "1-fold smoke first, project to
  5-fold, kill if projection ≥1h" — `--folds 1` flag in any new
  GPU kernel. Add to do-and-dont.md GPU-workflow checklist.

- `tag: minimal-orth-basis-falsified-day3` — Day-3 evening
  hypothesis: "the 10 GBDT consensus clones in M5h are redundant
  and removing them will tighten the OOF→LB gap." Tested via M5p
  (K=6: 3 most-diverse + LR-FE + EBM + baseline) and M5n_3b (K=4:
  most-diverse only). Both REGRESSED substantially: M5p −237bp LB,
  M5n_3b −291bp LB. The OOF→LB gap WIDENED, not tightened (52bp →
  85bp → 108bp). Lesson: even bases that look "redundant" by
  Spearman correlation provide ensemble averaging that improves
  generalization. The pool's LB rank IS the consensus; removing
  clones exposes the rank to whichever model's idiosyncratic
  errors dominate the smaller pool. Fix: do not drop bases purely
  on diversity / L1 grounds. Pruning must be inner-CV-validated
  (the L1-prune rule from M5h was diversity-conscious AND OOF-
  preserving — that's the right shape).

- `tag: lr-meta-rank-lock-strong-anchor` — Day-4 slot-2 exploration:
  M5q (M5h + RealMLP, Strat 0.95057, LB 0.95005) is the new
  PRIMARY. Tested 4 layered candidates on top: M5t (+H1),
  M5u (+H1+EBM), M5v (+LR-FE), all ρ ≥ 0.9997 vs M5q → TIE_EXPECTED
  on LB. Even LR-FE (most-diverse base from Day-3) got L1=0.675
  in M5v but ρ=0.9998. The LR-meta-on-strong-anchor is rank-
  saturated: adding orthogonal bases redistributes L1 weights
  internally but the test ranking is locked. Strategic
  implication: to break a strong-anchor stack's LB, change the
  ANCHOR composition (replace bases, change mechanism family),
  not stack on top. Add to do-and-dont.md: "When ρ between candidate
  and anchor is ≥0.9997, slot is wasted as a calibration probe;
  prefer ANCHOR-replacement variants (swap, not add)."

- `tag: pre-submit-rank-diff-check` — Day-3 burned 3 slots (M5h, M5h2,
  M5j) all landing at LB 0.94991. Post-hoc diff of the submissions:
  predictions differ noticeably in ABSOLUTE values (M5h vs M5j: 44%
  of rows differ >1e-3, max abs diff 6%), BUT Spearman rank
  correlation ≥0.9997 across all pairs. AUC depends only on rank,
  so near-identical rank → identical LB. The LR meta over highly
  correlated GBDT bases produces near-identical RANKINGS regardless
  of which marginal base is included/swapped/dropped. Fix:
  ALWAYS pre-submit-diff against the most recent same-class submission.
  If Spearman > 0.999 vs the prior submission, the LB will tie within
  Kaggle's quantization (5 decimals) — the slot is wasted as a
  calibration probe. Add to do-and-dont.md: "Before any submit, run
  `pre_submit_diff(new, last_submitted)` printing Spearman + rank-shift
  stats; if rho > 0.999, abort and propose a structurally different
  candidate." Today's signal: in-pool tweaks (LR-meta-on-correlated-
  GBDTs) cannot move LB — only different MECHANISM FAMILIES can.

- `tag: posthoc-isotonic-overfits-OOF` — per-(Year,Race) isotonic
  fit on M5h OOF showed +24.6bp Strat OOF lift in-sample; inner-CV
  (5-fold split on the OOF rows themselves, fit isotonic on 4 folds,
  eval on 5th) gave **−10.9bp**. Per-Race alone: +11.8 in-sample,
  **−5.3 inner-CV**. The OOF predictions are out-of-fold but fitting
  per-group isotonic on the same OOF rows we evaluate on is just
  fitting noise. Fix: any post-hoc transformation of OOF (isotonic,
  Platt, per-group rescaling) MUST be inner-CV validated before
  treating its OOF lift as a real candidate. Reliability bins on
  M5h showed it is already globally well-calibrated (gap ≤0.003
  across all 10 deciles), so the "miscalibration to fix" was
  imaginary. Add to do-and-dont.md: "post-hoc calibration on OOF
  must use a held-out inner CV; never trust the in-sample lift."

- `tag: kaggle-p100-torch-sm60-incompat` — RealMLP kernel v1 failed
  in 39s on Kaggle P100 with `torch.AcceleratorError: CUDA error: no
  kernel image is available for execution on the device`. Cause:
  P100 is sm_60 (CUDA capability 6.0); current PyPI torch (pulled
  in via `pip install pytabkit`) supports only sm_70+. Existing
  `cb-slow-wide-gpu` kernel uses CatBoost's own GPU runtime (not
  torch) so P100 worked there — the gotcha is *torch-on-P100
  specifically*. Fix: set `"machine_shape": "GpuT4x2"` in
  kernel-metadata.json for any torch-based kernel. T4 is sm_75 and
  supported by current torch. Add to do-and-dont.md kernel-template:
  "any torch / pytabkit / pytorch-lightning kernel: use T4x2, not
  the default P100. P100 is fine for CatBoost-GPU and LGBM-GPU which
  ship their own CUDA kernels."

- `tag: rule-R1-miss-groupkf-day3` — Day-3 mid-session, ran GroupKF
  anchor on d3a, d3b, M5i, M5j, M5k despite Rule R1 ("GroupKF dropped
  Day-3+ — U3 confirmed i.i.d. test, Strat is LB proxy, gap +3.8bp").
  Cause: copied two-anchor pattern from baseline_two_anchor.py and
  d2a_target_encoding.py without re-checking R1. Burned ~50% of
  per-run compute on artifacts that informed no decision (Strat alone
  drives both LB-proxy and stack inclusion). Fix: agent rule —
  before writing any new probe / base / stack script, grep CLAUDE.md
  for rules tagged R1..R8 and apply current verdicts; never copy
  two-anchor scaffolding from pre-R1-update scripts. Codify by
  amending common.py with a `STRAT_ONLY = True` flag (s6e5-specific)
  and removing GroupKF blocks from new scripts.

- `tag: bootstrap-env-var-mismatch` — `bootstrap.sh` gates on
  `KAGGLE_API_TOKEN` and prompts interactively when unset; the sandbox
  provides the same secret under `KAGGLE_KEY` (alongside
  `KAGGLE_USERNAME`). The patched kaggle CLI here also reads
  `KAGGLE_API_TOKEN` (vanilla CLI uses `KAGGLE_USERNAME`+`KAGGLE_KEY`).
  Result: agent surfaced a false "missing token" blocker and asked PI
  despite the secret being present under a different name. Workaround
  used: `KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle competitions download …`.
  Fix: (a) update `bootstrap.sh` to fall back `KAGGLE_API_TOKEN ←
  KAGGLE_KEY` when the latter is set, skipping the prompt; (b) agent
  rule: before asking PI for a credential, `env | grep -i <service>`
  for any standard CLI var name, not just the one the local script
  references; (c) update the skill template `bootstrap.sh` mirror.

- `tag: eod-auto-recognition` — PI had to redirect agent twice on
  day-end behavior in one session: first to clarify the day-end
  definition (slot-exhaustion-or-PI-EOD), then to clarify the
  automation (no slash commands; recognize from context). Both
  are now in the skill (loops.md auto-trigger section,
  do-and-dont.md DO/DON'T pair). The agent had a tendency to
  PROPOSE rather than ENACT — propose slash commands, propose
  hooks, propose templates. The PI wants in-context recognition
  + execution, not infrastructure proposals. Skill now forbids
  proposing slash commands as the automation mechanism.

## 2026-05-05

- `tag: layered-orthogonal-base-tie-3x-confirmed` — Day-4 slot-2
  exploration added two structurally orthogonal bases on top of M5q:
  CatBoost YetiRank (pairwise loss; ρ=0.666 vs M5q test — most diverse
  base ever measured) and Gaussian-NB-mixed (ρ=0.853). Both are
  fundamentally different model families from the GBDT/NN pool.
  Stack-level ρ vs M5q was 0.99966 (yetirank) and 0.99981 (nb), both
  TIE_EXPECTED. Combined M5z (yetirank + nb) ρ=0.99957, also TIE.
  3rd independent confirmation of `lr-meta-rank-lock-strong-anchor`:
  the LR meta with expand() produces near-identical TEST RANKINGS
  regardless of what orthogonal base you stack onto a strong GBDT-heavy
  anchor. Even ρ=0.666 underlying diversity gets washed out at the
  meta level. Fix: do not burn slot adding orthogonal bases via LR
  meta on top of M5q. Slot-add via LR meta is dead. Either change the
  meta-learner OR the BASE pool itself.

- `tag: rho-0.995-not-tie-meta-switch-bounded` — Day-4 slot-2 actual
  submit was m5_meta_lgbm_shallow (LGBM d=3 over the same K=14 base
  pool that M5q's LR meta uses). ρ vs M5q test = 0.99508 — well below
  the 0.999 tie-threshold. Result: LB came in at 0.95001 (M5q LB
  0.95005, Δ -4bp), NOT a tie. This validates the 0.999 threshold
  empirically: ρ=0.995 produces ~4bp LB movement at this scale of
  pool, ρ≥0.999 produces tie. OOF→LB transfer for meta-switch was
  ~50% of the OOF regression (-0.92bp OOF → -0.4bp LB), in contrast
  to RealMLP's 10× OOF→LB amplification on base-add. Strategic
  takeaway: rank-lock is PARTIALLY a meta-learner artifact (different
  meta DOES move LB) but the LR meta is close to optimal for this
  pool — switching costs, doesn't lift. **Base-pool signal ceiling
  is the binding constraint**, not meta-learner choice. Add to
  do-and-dont.md: "If you're considering meta-learner alternatives,
  test the THEORY first — the OOF tells you whether the ceiling is
  the meta or the bases. If candidate meta OOF < anchor OOF, expect
  LB to follow downward at ~50% transfer."

- `tag: bigger-moves-overrride-seed-variance` — Day-4 evening, I
  proposed multi-seed bagging as a slot-2 improvement; PI corrected
  with "We have plenty of headroom. Don't think small (seed
  variance) yet". Seed-bag is ~+1-3bp/base; with 34bp headroom and
  23 days remaining, the EV calculus dictates multi-bp moves
  (pseudo-labeling, NN-family multiplication, recursive bases) over
  single-bp tuning. Fix: when proposing next-move ranking, weight
  candidates by EV_bp / day_invested AND headroom_bp_to_target
  before sequencing. Sub-1bp moves are saved for the final-window
  R5 probe.

- `tag: external-data-already-tested-d2` — Proposed external-data
  integration as an unmined lever in a strategy review. PI corrected:
  `audit/2026-05-04-d2-probe1-external-join.md` already shows the
  external join (`aadigupta1601/f1-strategy-dataset-pit-stop-prediction`)
  fails at 5.6% test match rate — host shuffled or synthesized rows
  beyond the original. Plus `Normalized_TyreLife` is host-forbidden.
  Fix: before listing any "unmined lever" in a strategy review,
  grep `audit/` for prior probes on that mechanism. Strategy reviews
  must reference what's already been tested, not duplicate-propose.


- `tag: submit-without-confirmation` — Day-10: agent submitted d9c
  K=20 swap+FM after user said "go" to recommended-next-moves
  (FM bagging + sweep), interpreting "go" as approval to also
  submit. PI corrected: "go" was approval for the experiments,
  NOT for submission. Per CLAUDE.md Rule 1 every `kaggle competitions
  submit` requires EXPLICIT single-shot approval — "go" on a
  multi-step plan does not transfer to the submission step.
  Fix: when a multi-step plan ends in "submit best candidate",
  treat the submit step as a separate gate; report results, then
  WAIT for explicit "submit" / "yes" / "go ahead and submit"
  before calling `kaggle competitions submit`. Do not auto-submit
  even when EV is positive.

- `tag: pred-lb-heuristics-broken-for-hier-meta` — Day-13: my
  pre-submit gates ALL FAILED for d13 Stint τ=100000 yet it landed
  LB 0.95041 (+7bp NEW PRIMARY, 11.6× OOF→LB upside):
  - G3 rare-class flip ratio 0.211 < 0.5 ("FAIL") — was actually
    benign; row-extreme reshuffling aligned with public LB
  - ρ=0.998 sub-tie ("expect -1 to -2bp LB penalty") — was actually
    +10bp lift vs the d9f K=21 swap
  - R7 253-flips > 200 ("HEDGE-only") — was a PRIMARY-grade lift
  Three precedent-driven heuristics all wrong simultaneously means
  this is a new model class, not a tuning variant. The hier-meta's
  per-segment partial-pooling produces predictions whose row-extreme
  structure is GENUINELY DIFFERENT from the global-LR meta's, in
  ways that align with public LB. Fix: when a candidate is in a
  *new mechanism family* (FM-class was, hier-meta is), the
  precedent-derived heuristics from prior families do not apply;
  treat OOF lift + leakage-robustness probe as the primary gates,
  not the G3/ρ/R7 thresholds. Compute the GKF probe BEFORE
  assuming a sub-tie ρ candidate will under-perform.

- `tag: lr-convergence-stall-on-small-segments` — Day-13: d13
  Compound×Stint hier-meta sweep ran 41 minutes at 99% CPU stuck
  past fold-2 logs. Cause: per-segment LR fits on 24-row segments
  didn't converge within max_iter=2000 lbfgs iterations on the
  63-feature expanded space; lbfgs oscillated indefinitely.
  Fix in d13e: min_rows=1000 (skip small segments to global
  fallback) AND max_iter=500 (cap pathological convergence). 5-fold
  Compound×Stint sweep then completed in 7 minutes total.
  Generalization: any per-segment LR routine with arbitrary segment
  sizes needs a min_rows guard PLUS a sanity-bounded max_iter — the
  lbfgs solver in scikit-learn does not raise on convergence
  failure, just keeps iterating until max_iter.

- `tag: leak-corrected-meta-over-corrects-row-extremes` — Day-12/13:
  d10d attempted to fix the Strat-meta's leakage bias by refitting
  LR on GKF OOFs and applying coefficients to GKF-test predictions.
  FM_B got the predicted L1=6.96 dominance, but G3 flip ratio came
  out 0.001 (1751 rows demoted out of top-1% vs only 2 promoted).
  The reasoning failure: GKF OOFs structurally cannot see test-row-
  specific extremes (a held-out Race has no train-mate context), so
  the GKF-fit meta over-credits FM bases by under-crediting GBDT
  row-specific signal — but the i.i.d. test set DOES contain those
  row-extremes, so smoothing them away destroys real predictive
  value. Path B (per-segment partial-pooled meta) is the correct
  synthesis: preserves global-LR's row-extreme calibration on
  common segments while letting FM dominate on rare/edge segments.
  Fix: when correcting a leakage bias in validation, identify what
  the unbiased validation REMOVES that you NEED to keep; per-segment
  partial-pooling beats wholesale re-fit-on-leak-blocked-OOFs every
  time the test set is i.i.d. with train.

- `tag: 1-3bp-probes-cannot-close-40bp-gap` — Day-13 evening: PI
  pushed back on a "submit τ=20000 for +2bp" recommendation: "we
  want to improve by 40bp not 2." Fair pushback. The agent had
  drifted into incremental τ-tuning after the d13 Stint +7bp win.
  Sequencing fix: after a structural-breakthrough submit (FM-class
  d9c, hier-meta d13), the next move should be ANOTHER structural
  candidate (TabPFN, SCARF, DeepFM, pseudo-label cascade) — not
  τ-sweep tuning of the same mechanism. Tuning candidates belong
  in the calibration-probe budget (1 per day max during the comp
  middle, R5 final-window only at end).

## 2026-05-13 PM (branch `claude/review-ml-handover-VTvWw`)

- `tag: fm-class-amplification-not-universal` — Day-13 PM: d13a S3
  K=24 (5 FMs in pool, ρ=0.99976 vs PRIMARY) submitted at OOF
  +0.20bp pred → LB 0.95032 = TIE/−0.02bp regress. Five prior
  FM-class submits (d9c, d9f, d9h, d9i) showed 5–300× OOF→LB
  amplification at similar ρ. The discriminating variable is
  whether the FM adds NEW INPUT FIELDS (Cd/Ld/Nx/Pv augmentation
  in d9h/d9i) vs only RESHUFFLES the existing 12-field set
  (d13a/d13d V2/V3). Same-field reshuffles are TIE-class even
  when ρ < 0.9998. Fix: separate "new-field FM" and "new-partition
  FM" as distinct axes in the EV calculus. Pre-flight Q5 (gate
  precedent) should match on NEW INPUT presence, not just ρ band.
  Pre_submit_diff ρ>0.999 → TIE warning *was correct* here, even
  though d9h/d9i previously beat it.

- `tag: gkf-vs-strat-stack-pool-refactor-asymmetry` — Day-13 PM:
  d13b GKF FULL_22 stack matrix says "drop d9c_FM costs −0.01bp"
  (substitutable). d13c Strat refactor confirms (T1 K=23 = T0 K=24,
  −0.01bp). BUT d13c also says "drop GBDT leak-eaters
  (e5_optuna_lgbm + cb_slow-wide-bag) costs −2.5 to −2.6bp Strat"
  — the same bases that drop −209 to −247bp under GKF. **Pool-
  refactor decisions need BOTH gates.** Single-axis (GKF-only) read
  of d12 Option 1 would have wrongly dropped GBDTs and burned 2.5bp
  on submit. Fix: amend HANDOVER critical-rule §4 — for pool-removal
  decisions, the candidate must be substitutable on BOTH Strat AND
  GKF; substitutability on GKF alone (rank-lock dissolution under
  leak-blocking) is a necessary but not sufficient gate. Public LB
  is row-iid (U3); GBDT leak-eaters absorb fold-mate signal that
  IS in test rows.

- `tag: cross-branch-converging-same-conclusion-redundant-submit` —
  Day-13 PM: main and this branch independently submitted
  same-12-field FM-partition probes (main's V1 5/3 with Ln,
  ours d13a S3 K=24 with Cd). Both landed LB 0.95032 TIE.
  Multi-agent independent confirmation IS valuable for dead-listing,
  but two slot-burns on the same conclusion at 9-slots/day budget
  is wasteful. **Cause**: this session merged origin/main only after
  experiments completed (when preparing handover) — the parallel
  V1 5/3 commit was already on main when we started d13a, but we
  didn't fetch. Fix: session-start ritual — `git fetch origin && git
  log --oneline HEAD..origin/main && git diff HEAD..origin/main
  HANDOVER.md` BEFORE any base build, not after. Each agent's
  HANDOVER read on start is stale within ~30min of parallel work;
  refresh-then-act prevents same-mechanism re-runs.

- `tag: review-branch-bootstrap-cost` — Day-13 PM: claude/review-ml-handover-VTvWw
  container started with empty `data/` and no numpy/torch/pandas.
  ~3 min spent on `pip install numpy pandas scikit-learn scipy
  torch` + `kaggle competitions download -c playground-series-s6e5`
  before first experiment. Not blocking but wasted 5% of session.
  Fix: SessionStart hook for review/feature-branch sessions that
  pip-installs requirements.txt + kaggle-downloads data into `data/`
  if absent. Hook should be idempotent (skip if `data/train.csv`
  exists). See `~/.claude/skills/session-start-hook` skill.

### Process improvements (Day-13 PM consolidated)

1. **Session-start ritual** = `git fetch origin && git log
   HEAD..origin/main && diff HANDOVER.md` BEFORE any base build.
   Single-author HANDOVER not enough when parallel agents commit
   bases mid-session.
2. **Encode mechanism family in submission description** — e.g.
   `family=fm-partition-reshuffle` or `family=hier-meta-segment` —
   so cross-agent dedup is grep-able from `kaggle competitions
   submissions` log.
3. **Pool-refactor needs BOTH Strat AND GKF gates**, not just one.
   GKF-substitutability ⊂ Strat-substitutability for leak-eaters.
4. **FM-class precedent applies only to new-input FMs**, not
   partition-reshuffles. Add to do-and-dont.md: "Reshuffling the
   same fields across FM partitions is a meta-routing change, not
   a base-class change. Expect TIE_EXPECTED."
5. **Pre-warm hook for review branches** — pip install + kaggle
   download in SessionStart, idempotent.
