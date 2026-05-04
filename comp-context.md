# comp-context.md — settled-once facts

Filled out once on Day 1 by the kickoff agent (API auto-fill +
batch confirm with PI). NEVER re-asked. Loaded by every subagent
on session start.

## Facts (auto-filled by kaggle CLI)

```yaml
slug: playground-series-s6e5
url: https://www.kaggle.com/competitions/playground-series-s6e5
title: Predicting F1 Pit Stops
task: binary
metric: roc_auc                 # to confirm — scores ~0.95 consistent with AUC
train_rows: 439140
test_rows: 188165
deadline: 2026-05-31 23:59:00 UTC
team_size_limit: TBD            # confirm from comp rules
submission_budget: 5/day
final_submissions: 2            # Playground default — confirm
data_license: CC BY 4.0         # Playground default — confirm
external_data_allowed: yes      # Playground default — confirm
```

## Schema (auto-filled by kickoff EDA)

```yaml
target_col: PitNextLap
id_col: id
feature_count:
  numeric: 11
  categorical: 3
class_priors: "pos=0.199, neg=0.801"
missingness_train: 0.0
missingness_test: TBD           # run after test EDA if needed
```

## LB context (auto-filled from leaderboard download)

```yaml
lb_best_at_kickoff: 0.95435
pack_score_at_rank_100: 0.95138
total_teams_at_kickoff: 542
top_5pct_rank: 27               # 0.05 * 542
top_5pct_score: 0.95345         # score at rank 27
public_split_pct: 20            # default for Playground; confirm for Featured
probe_resolution_floor: 0.00005 # 80/20 split × N_TEST (re-derive if Featured)
```

## Strategic decisions (PI-answered, batched Q4 in kickoff)

```yaml
lb_stability: stable
external_data_strategy: use               # https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/data
time_budget_total_days: 27               # 2026-05-04 → 2026-05-31
compute_budget: cpu_and_gpu_kaggle
model_preferences: trees first, GPU if NN shows clear edge
```

## Daily-updated facts

These do change. The Bookkeeper updates them in CLAUDE.md
current-state, NOT here.

```yaml
# In CLAUDE.md, not comp-context.md:
# - lb_best_today
# - our_lb_best
# - submissions_used_today
# - saturation_count
```

## Anti-patterns

- Don't re-ask any field above. Read this file instead.
- Don't add new strategic-decision fields after Day 1 without
  flagging it as a friction event (`audit/friction.md`,
  `tag: settled-once`).
- Don't put per-experiment OOF/LB results here. They go in
  `audit/YYYY-MM-DD-*.md` and the calibration ladder.
