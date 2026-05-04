# comp-context.md — settled-once facts

Filled out once on Day 1 by the kickoff agent (API auto-fill +
batch confirm with PI). NEVER re-asked. Loaded by every subagent
on session start.

## Facts (auto-filled by kaggle CLI)

```yaml
slug: playground-series-s6e5
url: https://www.kaggle.com/competitions/playground-series-s6e5
title: Predicting F1 Pit Stops
task: classification
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
target_col: PitNextLap                  # float64 {0.0, 1.0}
id_col: id
feature_count:
  numeric: 11
  categorical: 3
categorical_cols: [Driver, Compound, Race]
categorical_levels:
  Driver: 887   # 801 also in test (full overlap)
  Compound: 5   # MEDIUM HARD SOFT INTERMEDIATE WET
  Race: 26      # 26 grand prix names; ALL overlap train↔test
class_priors: "pos=0.199, neg=0.801"
missingness_train: 0.0
missingness_test: 0.0
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
time_budget_total_days: 27                # 2026-05-04 → 2026-05-31
compute_budget:
  local_sandbox: cpu_only                 # ~8-core, no GPU; for FE, light GBDT, smoke probes, stacking
  kaggle_notebook: gpu_available          # P100 (single) or T4 x2; USE for heavy training
gpu_workflow: kaggle_notebook             # heavy training runs on Kaggle; pull artifacts back via
                                          # kaggle kernels output / dataset upload, then stack locally
gpu_when_required:                        # mechanisms that MUST go to Kaggle GPU, not local CPU
  - RealMLP / PyTabKit (any NN)
  - deep CatBoost (depth >= 8) for 5-fold
  - any 5-fold whose local-CPU projection > 1h
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

## Pre-baseline gate (2026-05-04)

```yaml
gate_artifacts:
  brief: brief.md                                    # 149 lines, host verbatim
  schema_target_groups: audit/2026-05-04-pre-baseline-gate.md
  prior_art: audit/2026-05-04-pre-baseline-gate.md   # appended block
  domain_notes: audit/2026-05-04-pre-baseline-gate.md
  metric_notes: audit/2026-05-04-pre-baseline-gate.md
gate_status: cleared                                 # PI signed off 2026-05-04
group_key_for_R1_anchor_b: Race                      # 26 levels; 5-fold ≈ 5 races/fold
forbidden_columns:
  - Normalized_TyreLife    # host-removed from original; do NOT reintroduce
structural_findings:
  pitstop_pitnextlap_match_rate: 0.724    # ≈ chance (independent baseline 0.719 at priors 0.136 / 0.199)
  lead_pitstop_single_feature_auc: 0.512  # U2 probe — basically random; lead_PitStop is NOT a leak signal
  train_test_split_structure: iid_row_level   # U3 probe — alt-ratio 0.447, 0/13185 contiguous groups
  test_lead_pitstop_computable_pct: 0.974     # 97.4% of test rows have a same-(Race, Driver) successor in test
```

## Anti-patterns

- Don't re-ask any field above. Read this file instead.
- Don't add new strategic-decision fields after Day 1 without
  flagging it as a friction event (`audit/friction.md`,
  `tag: settled-once`).
- Don't put per-experiment OOF/LB results here. They go in
  `audit/YYYY-MM-DD-*.md` and the calibration ladder.
