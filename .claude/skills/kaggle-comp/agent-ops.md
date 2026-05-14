# Agent ops — subagents, monitors, long-running compute

Operational discipline for launching subagents, watching processes,
and dispatching long-running Python jobs. Consolidates frictions
that previously lived across Rules 28-31 and several
`audit/friction.md` entries.

## Subagent dispatch

- **Never delegate long-running compute to a `general-purpose`
  subagent.** Subagents SIGTERM their child processes when they
  time out or exit; artifacts are half-written. Friction tag
  `subagent-friction-4-of-4-recurrence`.
- **5-minute ceiling** on any Python launched from a subagent. If
  the candidate could exceed 5 min, launch from the main thread
  instead.
- **Subagent contract for python wrapping**: `python script.py >
  log 2>&1 ; wait for exit ; read log ; summarize`. Forbid Monitor
  + early-exit composition.
- **Use Explore / Plan / Agent for research, not for execution.**
  Reading, searching, planning is fine; running probes is not.

## Process monitoring

- **File-existence sentinels beat pgrep.** When polling for a
  python process completion, `pgrep -f "<script>"` matches the
  bash wrapper itself because Claude Code bash wrappers `eval` the
  command string — symptoms: until-loop never exits, multiple
  zombie watchers. Preferred (in order):
  1. `until [ -f <artifact_sentinel> ]; do sleep N; done`
  2. `pgrep -f "^python.*<script>"` (anchored)
  3. `Monitor` tool with `tail -F` on a sentinel line in script output
- **Cancel stale monitors** the moment the watched artifact appears
  (via TaskStop). Unbounded `tail -f` after the run ends produces
  ~10 stale events that fill chat tokens.
- **Don't pipe long-running scripts through `tail -N`.** The pipe
  buffers all output until process exit; on timeout-kill, no log
  reaches disk. Always redirect: `python script.py > log 2>&1 &`.

## Concurrent CPU

- **≤2 CPU-heavy jobs at a time.** Three parallel LightGBM trains
  hit OOM under sandbox CPU contention. Seven LightGBM probes ran
  4× slower than one.
- **If 3+ jobs are unavoidable, set `n_jobs=floor(N_CORES/3)`
  explicitly per process.** Don't trust the OS scheduler to share
  fairly under LightGBM/CatBoost OpenMP.
- **Schedule cheap probes (<30 s) ahead of slow ones.** Keeps the
  queue draining instead of head-of-line blocked.

## Kaggle kernel templates

- **GPU type by model class:**
  - **GpuT4x2** — default for torch / pytabkit / RealMLP /
    any NN kernel. PyPI torch wheels do not ship sm_60 (P100).
  - **P100** — CatBoost-GPU and LightGBM-GPU only (their own CUDA
    kernels handle sm_60).
- **Metadata gotcha**: `kaggle kernels init` writes string-quoted
  booleans; Kaggle silently treats them as `false`. Edit to bare
  `true` / `false` before pushing.
- **Data discovery**: `Path('/kaggle/input').rglob(...)` over
  hardcoded paths. Hardcoded paths break on dataset-version
  changes.

## Probe extrapolation

- **2-3× safety factor on cat-heavy mechanisms.** 1-fold probe
  time × 5 underestimates 5-fold by 2-3× when high-cardinality
  cats are present (e.g., Driver=887 with depth=8). Apply before
  any "fits in 1h" verdict.
- **1-hour cap is on single-fold wall time, not extrapolated
  5-fold.** If one fold completes within 1h on production hardware,
  run it, see the result, then decide.

## Sandbox-vs-host environment

- **`env | grep -i <service>` before asking for a credential.**
  The bootstrap script previously gated on `KAGGLE_API_TOKEN` while
  the sandbox exposes `KAGGLE_KEY`; surfaced a false "missing
  token" blocker twice.
- **Permission preview**: before any compute that needs network,
  `curl -sI <url>` to confirm reachability rather than discovering
  a network gate mid-run.

## Friction tags this file replaces

`subagent-non-execution`, `subagent-monitor-truncation`,
`subagent-friction-4-of-4-recurrence`,
`subagent-shell-children-die-on-subagent-exit`,
`bash-watcher-pgrep-self-match-zombie-loops`,
`stale-monitor-noise-fills-chat-after-process-ends`,
`tail-pipe-buffering`,
`cpu-contention-multi-probe-batch`,
`parallel-lgbm-3way-contention-oom`,
`kaggle-p100-torch-sm60-incompat`,
`kaggle-p100-fallback-reproduced-day15`,
`probe-extrapolation-drift`,
`bootstrap-token-name-mismatch`.

When any of these recurs, the fix is in this file, not in
CLAUDE.md.
