# comp-context.md — settled-once facts (Orbit Wars)

Filled in once on Day 1 by the kickoff agent (Kaggle API auto-fill +
batch confirm with PI). Loaded on every session start. **Never re-asked.**

## Auto-filled by `kaggle competitions view -c orbit-wars`

```yaml
slug: orbit-wars
url: https://www.kaggle.com/competitions/orbit-wars
title: Orbit Wars
task: code/agent (multi-agent tournament; likely RL-flavoured)
submission_format: kernel                # NOT a CSV — agent code via Kaggle Notebook
metric: TBD                              # likely TrueSkill / Elo; confirm from comp page
deadline: TBD                            # YYYY-MM-DD HH:MM:SS UTC
team_size_limit: TBD
submission_budget: TBD                   # daily cap; confirm from comp page
final_submissions: TBD                   # the N agents that count for private LB
data_license: TBD
external_data_allowed: TBD
prize_pool_usd: 50000
```

## Auto-filled by kickoff inventory

```yaml
agent_io_spec: TBD                       # observation tensor shape + action space
episode_length: TBD                      # max steps / wallclock cap
who_moves_when: TBD                      # simultaneous / alternating
opponent_pool: TBD                       # baseline opponents Kaggle ships (random, rules, ...)
replay_format: TBD                       # JSON shape; per-step state log
compute_quota_per_submit: TBD            # CPU sec / RAM / time at eval
local_simulator_works: TBD               # kaggle_environments.make('orbit_wars')?
```

## LB context

```yaml
lb_best_at_kickoff: TBD
total_teams_at_kickoff: TBD
top_5pct_rank: TBD
top_5pct_score: TBD
```

## Strategic decisions (PI-answered, batched on Day 1)

```yaml
external_data_strategy: TBD              # use / hold / skip
time_budget_total_days: TBD
compute_budget:
  local_sandbox: cpu_only
  kaggle_notebook: gpu_available         # P100 (single) or T4 x2
gpu_workflow: kaggle_notebook            # heavy training on Kaggle
agent_class_preference: TBD              # heuristic / search / IL / RL / hybrid
```

## Pre-baseline gate (Day 1, code-comp variant — see pre-baseline-gate.md)

```yaml
gate_artifacts:
  brief: brief.md                        # comp description verbatim
  io_spec: audit/YYYY-MM-DD-day-1-data-inventory.md
  replay_summaries: audit/YYYY-MM-DD-day-1-data-inventory.md
  baseline_opponent_panel: audit/YYYY-MM-DD-day-1-data-inventory.md
  reference_kernel_replication: audit/YYYY-MM-DD-day-1-data-inventory.md
gate_status: TBD                         # cleared once PI signs off
```

## Anti-patterns

- Don't re-ask any field above. Read this file instead.
- Don't add new strategic-decision fields after Day 1 without flagging
  it as a friction event in `audit/friction.md`.
- Don't put per-experiment results here. They go in
  `audit/YYYY-MM-DD-*.md` and the calibration ladder.
