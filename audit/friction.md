# Friction log

One-liners. Distilled weekly per `~/.claude/skills/kaggle-comp/self-improvement.md`.

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
