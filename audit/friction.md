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
