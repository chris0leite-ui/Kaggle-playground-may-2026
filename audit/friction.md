# Friction log

**Format**: `YYYY-MM-DD  <tag>  <one-line description>`. Append-only
during the day. **Reset every Monday** (per `self-improvement.md`)
by archiving last-week entries to `audit/friction-archive.md` and
keeping only this-week + recurring-process frictions here.

This file is ≤150 lines. The full historical detail is in
`audit/friction-archive.md` (1,450+ lines; do not read by default).
Pre-distillation snapshots: `audit/archive-YYYY-MM-DD-friction-*.md`.

## This week (2026-05-12 → 2026-05-14)

```
2026-05-14  cv-gate-misleading-wide-rho   K=12 control-LGBM cross-val +18.194 bp; ρ_test 0.928; LB -15.4 bp REGRESSION. Bands updated.
2026-05-14  synth-label-decoupling        PitNextLap[L]=PitStop[L+1] holds only 80.95% of observable pairs; per-(D,R,Y) sum Spearman 0.60. K=11 OOF 0.95443 = Bayes-ceiling proxy.
2026-05-14  noise-ceiling-5-null-probes   5 mechanism classes overnight all NULL/REGRESSION; PRIMARY unchanged. Pivot to NEW-INFORMATION mechanisms.
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
