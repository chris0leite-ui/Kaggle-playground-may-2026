# Friction log

One-liners. Distilled weekly per `~/.claude/skills/kaggle-comp/self-improvement.md`.

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
