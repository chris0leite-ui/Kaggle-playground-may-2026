# comp-context.md — settled-once facts

Filled in once on Day 1 by the kickoff agent (Kaggle API auto-fill +
batch confirm with PI). Loaded on every session start. **Never re-asked.**

## Auto-filled by `kaggle competitions view -c <slug>`

```yaml
slug: TBD
url: https://www.kaggle.com/competitions/TBD
title: TBD
task: TBD                           # classification / regression / ranking / multi
metric: TBD                         # roc_auc / log_loss / rmse / etc
train_rows: TBD
test_rows: TBD
deadline: YYYY-MM-DD HH:MM:SS UTC
team_size_limit: TBD                # confirm from comp rules
submission_budget: TBD              # daily cap
final_submissions: TBD              # the 2 you select for private LB
data_license: TBD
external_data_allowed: TBD          # yes / no / TBD
```

## Auto-filled by kickoff EDA

```yaml
target_col: TBD
id_col: TBD
feature_count:
  numeric: TBD
  categorical: TBD
categorical_cols: []
categorical_levels: {}              # cardinality per cat col
class_priors: TBD                   # for classification
missingness_train: TBD
missingness_test: TBD
```

## LB context

```yaml
lb_best_at_kickoff: TBD
pack_score_at_rank_100: TBD
total_teams_at_kickoff: TBD
top_5pct_rank: TBD
top_5pct_score: TBD
public_split_pct: TBD               # 20 for Playground; check Featured
probe_resolution_floor: TBD         # 1 / sqrt(N_pos * N_neg)
```

## Strategic decisions (PI-answered, batched on Day 1)

```yaml
lb_stability: TBD                   # stable / unstable
external_data_strategy: TBD         # use / hold / skip; with URL if use
time_budget_total_days: TBD
compute_budget:
  local_sandbox: cpu_only
  kaggle_notebook: gpu_available    # P100 (single) or T4 x2
gpu_workflow: kaggle_notebook       # heavy training on Kaggle
gpu_when_required:
  - any 5-fold > 1h CPU projection
  - NN with > 100k rows
  - deep CatBoost (depth >= 8) for 5-fold
model_preferences: TBD              # trees first, GPU if NN shows clear edge
```

## Pre-baseline gate (Day 1)

```yaml
gate_artifacts:
  brief: brief.md                                # host verbatim
  schema_target_groups: audit/YYYY-MM-DD-pre-baseline-gate.md
  prior_art: audit/YYYY-MM-DD-pre-baseline-gate.md
  domain_notes: audit/YYYY-MM-DD-pre-baseline-gate.md
  metric_notes: audit/YYYY-MM-DD-pre-baseline-gate.md
gate_status: TBD                                 # cleared once PI signs off
group_key_for_R1_anchor_b: TBD                   # the natural group for GroupKFold
forbidden_columns: []                            # host-removed columns; do NOT reintroduce
structural_findings: {}                          # save discovery numbers from U1-U5
```

## Anti-patterns

- Don't re-ask any field above. Read this file instead.
- Don't add new strategic-decision fields after Day 1 without flagging
  it as a friction event in `audit/friction.md`.
- Don't put per-experiment OOF/LB results here. They go in
  `audit/YYYY-MM-DD-*.md` and the calibration ladder.
