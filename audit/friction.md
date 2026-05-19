# Friction log

**Format**: `YYYY-MM-DD  <tag>  <one-line description>`. Append-only
during the day. **Reset every Monday** (per `self-improvement.md`)
by archiving last-week entries to `audit/friction-archive.md` and
keeping only this-week + recurring-process frictions here.

This file is ≤150 lines. The full historical detail is in
`audit/friction-archive.md` (1,450+ lines; do not read by default).
Pre-distillation snapshots: `audit/archive-YYYY-MM-DD-friction-*.md`.

## This week (2026-05-12 → 2026-05-18)

```
2026-05-14  cv-gate-misleading-wide-rho   K=12 control-LGBM cross-val +18.194 bp; ρ_test 0.928; LB -15.4 bp REGRESSION. Bands updated.
2026-05-14  synth-label-decoupling        PitNextLap[L]=PitStop[L+1] holds only 80.95% of observable pairs; per-(D,R,Y) sum Spearman 0.60. K=11 OOF 0.95443 = Bayes-ceiling proxy.
2026-05-14  noise-ceiling-5-null-probes   5 mechanism classes overnight all NULL/REGRESSION; PRIMARY unchanged. Pivot to NEW-INFORMATION mechanisms.
2026-05-18  research-loop-overdue         10-day gap from prior research-loop (2026-05-08) despite plateau hitting Rule 7 thresholds on 2026-05-14. Rule 7 auto-fire not triggered; promotion candidate.
2026-05-18  kaggle-pages-recaptcha-gated  WebFetch on Kaggle /code and /discussion returns only page title; reCAPTCHA blocks notebook scrape. Switch R22 scan to authenticated `kaggle kernels list -c <slug> --sort-by voteCount`.
2026-05-18  tier-a3-menu-stale            10 of 13 Tier-A2/A3 picks from 2026-05-08 still pending 10 days later; menu was never executed. Rule 18 (claim-the-leaf) didn't bind because session focus drifted to K=11/Path-B variants instead.
2026-05-18  operator-vs-mechanism-axis    Same OOF (K=11+seg+HMM, 0.95446) at LR-meta gives LB 0.95382; under Path-B C×Stint τ=100k gives LB 0.95387. +5 bp LB transfer purely from operator. Mechanism-class and operator-class are orthogonal axes; past LR-meta nulls should be retested under Path-B.
2026-05-18  pathb-bag-seed-invariant-test 5-seed bag of K=13+Path-B produced identical test predictions (ρ=1.0). run_pathb fits FULL train for test (lines 116-128), only fold-OOF varies with seed. ~17 min CPU wasted. Multi-seed bag harness must use fold-fit averaging for variance reduction.
2026-05-18  gpu-kernel-prior-pattern-skip Phase F transformer needed 3 push iterations (data path rglob; sm_60 P100 torch reinstall; enable_internet for pip). All 3 fixes documented in prior kernels (p1-xgb-v4-gpu, d15b-dae-gpu). Cost ~30 min ops; would have been 1 push if I'd grep'd kernels/ first.
2026-05-18  snapshot-missing-orig-dataset Slim-kNN rebuild Phase A failed for 5/6 builders: data/original/f1_strategy_dataset_v4.csv not in local snapshot. Downloaded via `kaggle datasets download -d aadigupta1601/f1-strategy-dataset-pit-stop-prediction`. Bootstrap.sh should validate or pull this file.
2026-05-18  cpu-contention-phase-c-starved Phase A (qAO/qAF) + Phase C (probe_r5_graph) ran concurrently; Phase C smoke took 8+ min CPU (vs ~30s alone) before being killed and restarted post Phase A. Rule 30 "≤2 concurrent CPU-heavy" was technically met but contention was severe.
2026-05-18  two-axis-operator-sweep-missed Path-B introduced Round 5 with hard-coded Compound × Stint segmentation. Round 7 found DriverClass × Stint beats it by +0.106 bp OOF / +0.02 bp LB. Operator has TWO hyperparameter axes (τ AND segmentation); current Rule 21 demands ≥3 variants of one key hyperparameter only. 6 weeks of default segmentation missed +0.02 bp LB. NOT YET promoted; awaiting more data.
2026-05-18  fold-bag-quantize-on-public-LB Fold-fit bag of LB-confirmed candidate at OOF +0.1-0.3 bp ties public LB at TIE_ZONE. Observed twice in 2 days: R6.1 (fold-bag of R5.2, OOF +0.212 → LB tied 0.95387) and R7.2 (fold-bag of R7.1, OOF +0.264 → LB tied 0.95389). Pattern: variance reduction registers on OOF but quantizes away at 5-decimal LB. Submit ONLY as private-LB hedge or with otherwise-idle slot. NOT YET promoted; need a 3rd data point or final-LB confirmation.
2026-05-18  seg-axis-driver-only-lift     R8 tested 4 more Path-B segmentations (Year×Stint, DriverTier×Stint, RaceCluster×Stint, Compound×FirstPitWindow). 0 cleared +0.10 bp gate, 2 marginal at +0.05 (DriverTier, RaceCluster). 1 win + 3 marginal + 3 null across 7 segs tested R7+R8. Driver-axis (named-vs-D0XX 2-class) is the unique +0.10 bp lift dim; 4-quartile DriverTier captures same signal at lower magnitude. Promotion gate Rule 21 now satisfied; awaiting PI call.
2026-05-18  kaggle-cli-401-auth           kaggle CLI returns 401 Unauthorized on both `competitions submissions playground-series-s6e5` and `kernels list --competition <slug>`. KaggleAPIToke=KGAT_a1858... env var or stripped form both fail. Blocks R22 public-notebook scan AND quota-verify. Creds may need refresh; tracking as new operational fix.
2026-05-18  pre-submit-diff-floor-clip    pre_submit_diff Spearman 0.998 for rank-uniform blend CSV vs R7.1 CSV; true rank divergence is 0.99997 (TIE_ZONE). R7.1 CSV has 15.6 % rows floored at np.clip(0.001, 0.999); rank-uniform blend has 0 % floored rows. Spearman degraded by tie-structure mismatch (not actual rank divergence). Action: rank-normalize both inputs in pre_submit_diff or warn on distribution mismatch.
2026-05-18  segment-failure-noise-floor   Strategy-critic Section 1 surfaced MEDIUM × Stint 2 as worst PRIMARY segment (AUC 0.897, 25k rows, prior 0.448). 5-min specialist probe (kitchen-sink LGBM on subset, 18 features incl. 6 targeted interactions) yielded AUC 0.881 — 16 bp BELOW pool's 0.897 on the same subset. The pool extracts more signal than a 25k-row specialist can; segment IS at noise floor for row features. Lesson: per-segment AUC failure maps surface noise-floor segments, not feature-deficient ones. Don't equate "low-AUC segment" with "lift surface" without specialist probe. Promotion candidate (one-data-point; needs corroboration on other low-AUC segments).
2026-05-18  research-loop-output-queued-not-executed AM research-loop's 10-item Tier-A batch (`tier-a-batch.md`) was started but session diverged to R5 K=11 rebuild after C3 falsified + a2_2/a3_1 weak; the remaining 7 Tier-A items (Heilmeier residual, per-track fuel coef, nested TE, **C4 UID magic**, KNN-target-mean, F3 field-state, quantile groupby) never ran. Findings: research-loop output decays into queue debt without explicit slot-by-slot execution gating; PM addendum re-prioritises **NB4 / C4 / pit cascade** as next-session top-3. Promotion candidate: research-loop should emit BOTH a Tier list AND an execution-gate calendar (1 item per ≤1-h slot).
2026-05-18  research-loop-dedup-miss-vs-ledger PM research-loop emitted top-3 (NB4 / C4 / pit cascade). R9 session-start Explore reconnaissance found 2 of 3 already tested today: C4 smoke -16.2 bp (`tier-a-batch.md:92-102`) and pit cascade in 3 variants null/absorbed (`fe_picks_a2a3.py:218-332`, A3-1 K=4+1 +0.337 weak, F3 K=24 -0.015 null, pit-pressure K=11 -0.012 null). The research-loop didn't dedup against same-session prior work. Promotion: research-loop synthesis prompts must include same-session prior-run results, not just the historical mechanism-ledger.
2026-05-18  rank-lock-confirmed-three-axes R9 dual-track (NB4 Compound×Stint TE-as-base + C1 Aadigupta per-Race scalars) both regressed at K=14+Path-B (-0.022 / -0.045 bp NULL). Confirms K=13+Path-B is at structural row-feature ceiling across operator family / mechanism class / data class. Forced posture pivot to mechanism-expansion (seq2seq / graph / survival) for R10. Promotion: when 3 structurally-distinct mechanisms fall to rank-lock in <1 week, escalate to posture-pivot review automatically.
2026-05-18  k14-output-collision-extra-bases `build_K13_pathb_multiseg.py --extra-bases` writes `oof_K14_*` outputs that collide when 2 different K=14 bases are tested in same session (NB4 vs C1). Last writer wins; first artifact unrecoverable from disk. Mitigation: name suffix from `--extra-bases` list, or one --extras-tag CLI param. Not critical (JSONs captured both runs) but tedious.
2026-05-18  pathb-ref-baseline-hardcoded-r52 `build_K13_pathb_multiseg.py:282-286` hard-codes R5.2 reference (Compound × Stint τ=100k OOF). Δ vs current PRIMARY R7.1 (DriverClass × Stint) requires manual post-hoc compute. Parametrise `--ref-oof` for honest sweep comparisons.
2026-05-19  rank-lock-confirmed-four-axes R10 morning multi-constituent LR-meta alt-stack (LambdaRank stint + LambdaRank race + rolling LGBM + kernel hazard, 4 constituents) blended w/ R7.1 returned Δ < 0 at every weight (best w_R71=0.99 Δ=-0.045 bp). Closes alt-stack alongside R9's single-base closure. 4 mechanism families closed in <2 sessions: operator family / mechanism class / data class / alt-stack blend. Triggers immediate hedge-prep pivot.
2026-05-19  csv-rho-misread-on-floor-clip Initial misread of `pre_submit_diff.py` CSV ρ=0.998 as "OK band" led to staging 6 candidates under "recalibrated bands"; in fact 2026-05-18 friction `pre-submit-diff-floor-clip` already documents this as artifact (R7.1 CSV floors 15.6% rows at 0.001 vs rank-uniform blends with 0% floored — tie-structure mismatch degrades Spearman without reflecting actual rank divergence). TRUE divergence is .npy ρ. R8 60/20/20 is TIE_ZONE not OK band; R10 R7.2+K27 75/25 IS OK band by .npy ρ=0.999882. Same friction surfaced twice in 2 days; promotion candidate: fix pre_submit_diff.py to rank-normalize both inputs OR warn loudly on tie-structure mismatch.
2026-05-19  blend-sweep-r72-dominance-tie R10 blend-operator sweep top-by-OOF was R7.2+R6.1 80/20 arith at +0.262 bp Δ vs R7.1 — but R7.2 alone has +0.026 bp OOF vs R7.1 so the blend is mostly R7.2 dominated (R6.1 20% adds almost nothing). All top-20 OOF candidates are R7.2-dominated TIE_ZONE blends. Genuine cross-pool diversity ONLY surfaces when K27 is included (OK band ρ=0.9998). Lesson: when ranking blend candidates, sort by ρ-band first then OOF-Δ within band, not OOF-Δ overall — high-OOF + TIE_ZONE = no LB hedge value.
2026-05-19  kaggle-cli-kgat-auth-misdiagnosed Session-start `kaggle competitions submit` returned 401 across all endpoints. I incorrectly diagnosed as "token rotated, kaggle.json stale" and asked PI for new token; PI corrected: "the issue is with you". Root cause: KGAT_-prefixed Kaggle Access Tokens are Bearer-auth (`kagglesdk.get_access_token_from_env`) and must live in env var `KAGGLE_API_TOKEN` — NOT the `key` field of `~/.kaggle/kaggle.json` (which is HTTP-Basic legacy API key, expecting raw 32-char hex). Container had `KaggleAPIToke=KGAT_...` (typo'd env var name, ignored by SDK) and kaggle.json with KGAT in legacy `key` slot → CLI sent KGAT as Basic-auth password → 401. **Fix:** prefix every kaggle invocation with `KAGGLE_API_TOKEN="$KaggleAPIToke" kaggle ...`. Cost: 1 turn of misdirected PI dialogue; would have wasted ≥1 submission slot if PI hadn't redirected. Promotion candidate: kickoff-runbook needs a step "if kaggle.json key starts with `KGAT_`, set KAGGLE_API_TOKEN env var; do not rely on kaggle.json `key` field".
2026-05-19  hedge-3-lb-confirms-ok-band R10 HEDGE 3 (75/25 arith R7.2+K27) submitted, LB **0.95387** vs PRIMARY 0.95389. .npy ρ=0.999882 (OK band) registered as -0.02 bp LB delta — just-inside-TIE_EXPECTED behaviour, validating the .npy-ρ-based band call (vs CSV ρ which would have misread as REGRESSION_RISK at 0.998). First successful cross-mechanism diversity hedge of the comp at this rho band; pool-level coarse-segmentation diversity (HEDGE 2) remains untested. Calibration data-point: OK-band lower-boundary (ρ≈0.99988) → ≈-0.02 bp LB delta, well within R2d's 30-bp regression cap.
2026-05-19  r10-priority-queue-internal-dedup-miss HANDOVER R10 queue listed "A. seq2seq transformer" as a fresh mechanism while HANDOVER line 134's CLOSED list explicitly notes "Transformer v1 / v2 (R6 / R7)" already absorbed (R5 v1 std 0.91974 absorbed at K=11+Path-B; R6 v2 K=14+Path-B Δ=−0.014 bp absorbed). The "predict next-N PitNextLap" framing is a head-swap, not a structural pivot. Session-start mechanism-ledger lookup caught the dedup before push (saving ~2h Kaggle T4); PI directed redirect to B. Same shape as 2026-05-18 friction `research-loop-dedup-miss-vs-ledger` — dedup happens at session-start, not in priority-queue authoring. Promotion candidate: HANDOVER R10-style priority queue authoring must cross-reference its own CLOSED list and the mechanism-ledger before listing items.
2026-05-19  r11-c-coxph-stint-summary-collapses Cox-PH at TyreLife with per-stint covariate aggregates (compound + stint + year dummies + lap_start + position + lapdelta + cum_deg + is_named): standalone OOF 0.65 G1-FAIL because row-level differentiation across rows of a stint comes ONLY from the baseline hazard h₀(TyreLife) — covariates are constant within stint. A time-varying Cox PH (start-stop format with per-row covariates) would mechanically reduce to a smooth-baseline logit, removing most orthogonality vs LGBM. Survival/hazard mechanism CLOSED at this formulation; richer formulations expensive without obvious win. Cost: 75 s wall; cheaper than the BOTE +0.20 bp expected midpoint suggested.
2026-05-19  r11-b-max-drivers-truncation-bug R11-B transverse-attention kernel set MAX_DRIVERS=24 on the assumption F1 has ≤22 drivers per race-lap. Synthetic data has 887 unique Drivers and per-(Year, Race, LapNumber) groups with median 58 / mean 71 / max 373 drivers (4951 of 6182 train groups exceed 24). 70% of OOF entries left at 0.0; standalone 0.62 G1-FAIL. K=14+Path-B Δ=-0.035 bp NULL even with the truncated base. Promotion candidate: any per-(Race, Lap) cohort-aggregation mechanism needs to first measure n_unique(grouping_cols) from the data — never assume real-F1 dimensions on synthetic playground data.
2026-05-19  synth-data-871-pseudo-drivers-dissolves-cohort STRUCTURAL FINDING: synth generator densely upsamples drivers — 887 unique pseudo-drivers vs real F1's ~22. Per-(Year, Race, LapNumber) cohort has median 58 / mean 71 / max 373 drivers. F1-style inter-driver-competition mechanisms (undercut, pit window, safety-car cohort response) are STRUCTURALLY DEGRADED in this dataset: 71+ pseudo-drivers per race-lap dissolve the natural ~22-car cohort cohesion that gives those mechanisms predictive signal. Closes the cross-driver-interaction mechanism class for s6e5 (3 prior pit-cascade hand-aggregate variants + R11-B learned attention all NULL/absorbed). Implication: finer grouping (e.g., [Year, Race, Stint, Compound, LapBin]) or a stint-cohort-aware mechanism might re-introduce the signal; raw (Race, Lap) does not.
```

## Last week (2026-05-08 → 2026-05-11) — one-liner summary

```
2026-05-09  rank-lock-conditional-target-corr  Features can be feature-orthogonal to K=4 and STILL absorbed if target-corr is parallel to existing logit direction conditional on row.
2026-05-09  transductive-feature-1D-at-K4      V4 kNN-target-mean = +0.24 bp; V5/V6 with extra TE/MLP-embed absorb (ρ 0.989/0.988). Transductive lift is 1-D in K=4 logit space.
2026-05-09  rule-27-abort-too-strict-sub-bp    K=5 V4 kNN-aug ρ_test 0.99989 (above 0.999 threshold) but LB +0.8 bp. Bands recalibrated.
2026-05-09  bote-on-falsified-prong            Ran BOTEs for Prongs B/S before checking PI's axis-of-permission; 5-10 min wasted. Ask first when directive is "creative/original".
2026-05-08  research-scan-duplicate-claim      Proposed Frontiers AI peer-effect features as "untried" — already in A3-1 RankSortedGaps and nulled. Grep ledger first.
2026-05-08  tree-stack-meta-overfits-small-K   A2-8 LightGBM stack-meta on K=4 with 43 features lost -1.30 bp vs Path-B; -0.96 bp vs LR-meta. Convex LR beats GBDT at small K.
2026-05-08  day-counter-drift                  Prose drifted to Day-17/18/19 calendar-aligned; they were experiment-iteration codes. Today comp-day-8. ISO dates only forward.
2026-05-08  pool-rank-lock-logit-direction     3 inductive biases (LambdaRank/inter-stint/dual-head) NULL at K=10+1 within ±0.05 bp despite ρ 0.41-0.73.
2026-05-08  K4-sparse-promoted-PRIMARY         K=4 forward-greedy LB 0.95351 vs K=27 0.95368 (Δ -1.7 bp). 17 bases were dead weight.
2026-05-08  kernel-class-fails-at-300bp-gap    Kernel-SVM / NCA-kNN family null at K=27+1; structural diversity insufficient when AUC gap to GBDT > 300 bp.
2026-05-08  non-LR-meta-on-K4-regresses        Gradient-boosted meta -1.20 bp; MLP meta -7.77 bp; augmented LR flat. A30 dropped to FALSIFIED.
2026-05-08  rf-feat-breadth-doesnt-scale       Kitchen-sink RF (57 feat) -1.24 bp vs yekenot-only (38 feat); RF can't ignore weak features.
2026-05-08  rf-optuna-cant-tune-past-0.25bp    +0.24-0.27 bp ceiling across 4 RF runs (Optuna seeds 42/7, hand). Set by meta architecture, not RF.
2026-05-08  path-b-absorbs-base-lifts-below-0.5bp  K=5 = K=4 + RF τ=100k OOF Δ +0.02 bp from +0.25 bp K=4+1 base lift. Path-B absorbs single-base adds < +0.5 bp standalone.
2026-05-08  isotonic-overfits-when-base-calibrated  Per-gap + per-Compound isotonic both regressed (-2.18, -1.78 bp). Promotion candidate: ECE > 1% gating.
2026-05-08  anchor-cited-from-memory           Cited K=10+1 plain LR-meta OOF ~0.94850 in PI-facing prose; actual 0.95417 (5.7 bp off). Grep calibration-ladder before pasting numbers.
```

## Process frictions (recurring across weeks; promote to rule or automate)

```
recurring  subagent-non-execution            Subagents SIGTERM Python children at timeout/exit. 4-of-4 recurrence. Rule 28; agent-ops.md.
recurring  pre-submit-diff-missing           3 identical LB 0.94991 submits in one day for missing ρ-check. Rule 27 mandatory.
recurring  lesson-not-applied                Logged friction not applied same-session. Rule 29.
recurring  kaggle-p100-torch-sm60-incompat   P100 + torch reproduced 12 days apart (Day-3 → Day-15). Rule 30.
recurring  cpu-contention-multi-probe-batch  7 parallel LightGBM 4× slower; 3-parallel OOM. Rule 31.
recurring  handover-collisions               Multi-agent HANDOVER 5 rewrites/4 conflicts in one session. Rule 32 (session-start fetch).
recurring  premature-day-close               "Experiments done" interpreted as EOD. Day = Kaggle UTC quota. do-and-dont.md.
recurring  jargon-drift-without-glossary     CLAUDE.md acronyms unauditable by PI on first read. Rule 0.
recurring  bootstrap-token-name-mismatch     KAGGLE_API_TOKEN vs KAGGLE_KEY; env|grep first. agent-ops.md.
```

## Killed-and-do-not-retry — pointer only

Deduplicated list in `state/hypothesis-board.md ## Killed`. Full
enumeration in `state/mechanism-ledger.md`. Recent highlights:
target reformulation single-add (all leaky); Day-16 virgin-axes
(11 of 11 null); non-LR meta on K=4 (LightGBM/MLP/RF all regress);
kernel SVM 8 variants; Yao/Vehtari covariance Path-B; qBA Manhattan
kNN (LB regress); K=34 unrolled at any C-sweep value.

## How to add an entry

1. One line, tag-prefixed. If you need a paragraph, file an audit
   postmortem instead.
2. Append-only. Chronology is the signal.
3. If a fix is already a rule, reference the rule number rather
   than restating.
4. When a tag recurs 3+ times in a week, it's an automation
   candidate — see `.claude/skills/kaggle-comp/self-improvement.md`
   "Automation candidates".

## Weekly distillation checklist (Mondays)

- [ ] Tag-frequency count: any tag ≥3 entries this week?
- [ ] For each, decide: tighten existing guardrail, add new rule,
      or automate (settings.json / agent-ops.md / skill edit).
- [ ] Move last-week entries to `audit/friction-archive.md`.
- [ ] Reset this file to ≤150 lines.
- [ ] Update `.claude/skills/kaggle-comp/improvements.md` with any
      promoted edits.
