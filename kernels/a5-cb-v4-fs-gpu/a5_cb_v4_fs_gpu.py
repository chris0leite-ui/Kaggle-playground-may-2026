"""A5 — CB-v4-fs Kaggle GPU kernel.

A5-full: CatBoost-GPU with v4 yekenot recipe + ~24 field-state cross-row
aggregates. Tests friction tag `lr-meta-rank-lock-strong-anchor`'s
prescription: integrate orthogonal fs signal INTO the strong anchor by
retraining (bypass rank-lock at meta level).

Recipe = v4 (yekenot transfer + orig-aug + items 2/3/4/7) + fs cross-row
aggregates per (Race, Year, LapNumber) ± Compound.

Fold-safety:
  - fs aggregates use PitStop column (feature, not label). Rule 25 PASS
    (AV-AUC 0.502). `cross-row-aggregates-survive-strict-fold-safe-audit`
    confirmed; can compute on train+test combined.
  - fs_a (label-derived per-(Race,Year,Compound,Driver) aggregates) refit
    inside each CV fold using ti rows only. Rule 24 compliance.

Outputs:
  oof_p1_single_cb_v4_fs_gpu_strat.npy
  test_p1_single_cb_v4_fs_gpu_strat.npy
  p1_single_cb_v4_fs_gpu_results.json

To push (PI runs locally):
  kaggle kernels init -p kernels/a5-cb-v4-fs-gpu/
  kaggle kernels push -p kernels/a5-cb-v4-fs-gpu/

To pull artifacts back after run:
  kaggle kernels output <user>/a5-cb-v4-fs-gpu -p scripts/artifacts/
  # rename: oof_a5_cb_v4_fs_gpu_seed42.npy → oof_p1_single_cb_v4_fs_gpu_strat.npy
"""
from __future__ import annotations

# This is a near-identical copy of kernels/p1-single-cb-v4-gpu/p1_single_cb_v4_gpu.py
# with two additions:
#   1. `add_field_state(df, src)` — same aggregates as scripts/probe_field_state.py
#   2. fs_* columns appended to feats list before training
#
# All other lines (CB params, fold loop, orig-aug) are unchanged.
#
# Implementation kept close to v4 to preserve calibration:
#   - Same CB hyperparameters
#   - Same fold splits (StratifiedKFold seed=42)
#   - Same TE configs + smoothings
#   - Same orig-aug per-fold concat
#   - Same yekenot items 2/3/4 (floor-cat / count-enc / KBins)
#
# DELTAS FROM v4:
#   1. `add_field_state(df, src=concat(train,test))` after make_features_static
#      and before fit_fs_a. fs columns become numeric features.
#   2. `feats` list extends with the ~24 fs_* column names.

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
WALL_BUDGET_S = 5 * 3600
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

WITH_ORIG_DATA = True
N_SEEDS = 1
SEEDS_BAG = [42, 13, 71]
MAX_ROUNDS = 4000
SMOKE = False

COMPOUND_MAX_LIFE_MAP = {"SOFT": 15, "MEDIUM": 30, "HARD": 50,
                        "INTERMEDIATE": 25, "WET": 20}
COMBO_COLS = [("Race", "Compound"), ("Race", "Year"), ("Driver", "Compound")]


def add_field_state(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Field-state cross-row aggregates. PitStop is feature column (Rule 25 OK)."""
    g = source_df.groupby(["Race", "Year", "LapNumber"])
    a = g.agg(
        fs_field_size=("id", "size"),
        fs_n_pitting_now=("PitStop", "sum"),
        fs_pit_rate_now=("PitStop", "mean"),
        fs_mean_TyreLife=("TyreLife", "mean"),
        fs_max_TyreLife=("TyreLife", "max"),
        fs_min_TyreLife=("TyreLife", "min"),
        fs_std_TyreLife=("TyreLife", "std"),
        fs_mean_Stint=("Stint", "mean"),
        fs_max_Stint=("Stint", "max"),
        fs_mean_Position=("Position", "mean"),
        fs_mean_LapTime=("LapTime (s)", "mean"),
        fs_mean_RaceProgress=("RaceProgress", "mean"),
    ).reset_index()
    df = df.merge(a, on=["Race", "Year", "LapNumber"], how="left")
    rs = (source_df.sort_values(["Race", "Year", "LapNumber"])
                  .groupby(["Race", "Year", "LapNumber"])["PitStop"]
                  .sum().reset_index())
    rs["fs_cum_pits"] = rs.groupby(["Race", "Year"])["PitStop"].cumsum()
    rs["fs_cum_pit_lap_count"] = (rs.groupby(["Race", "Year"])["PitStop"]
                                    .cumcount() + 1)
    rs["fs_cum_pit_rate"] = rs["fs_cum_pits"] / rs["fs_cum_pit_lap_count"]
    df = df.merge(rs[["Race", "Year", "LapNumber",
                      "fs_cum_pits", "fs_cum_pit_lap_count", "fs_cum_pit_rate"]],
                  on=["Race", "Year", "LapNumber"], how="left")
    gc = source_df.groupby(["Race", "Year", "LapNumber", "Compound"])
    ac = gc.agg(
        fs_compound_n=("id", "size"),
        fs_compound_n_pitting=("PitStop", "sum"),
        fs_compound_pit_rate=("PitStop", "mean"),
        fs_compound_mean_TyreLife=("TyreLife", "mean"),
        fs_compound_max_TyreLife=("TyreLife", "max"),
    ).reset_index()
    df = df.merge(ac, on=["Race", "Year", "LapNumber", "Compound"], how="left")
    df["fs_TyreLife_vs_field_mean"] = df["TyreLife"] - df["fs_mean_TyreLife"]
    df["fs_TyreLife_vs_field_max"] = df["TyreLife"] - df["fs_max_TyreLife"]
    df["fs_Position_vs_field_mean"] = df["Position"] - df["fs_mean_Position"]
    df["fs_Stint_vs_field_mean"] = df["Stint"] - df["fs_mean_Stint"]
    return df


# Rest of script (FE pipeline, training loop, orig-aug, save) — see
# kernels/p1-single-cb-v4-gpu/p1_single_cb_v4_gpu.py for full details.
# Apply this single delta to that script:
#   src = pd.concat([train, test], ignore_index=True)
#   train_S = add_field_state(train_S, src)
#   test_S = add_field_state(test_S, src)
#   fs_cols = [c for c in train_S.columns if c.startswith("fs_")]
#   feats = feats + fs_cols   # AFTER feature_columns_for_lgbm
# CB cat-cols don't need fs_* (all numeric); just add to feats.

print("=== A5 CB-v4-fs kernel scaffold ===")
print("This file is a SCAFFOLD. Copy v4 kernel and apply the diff above")
print("before pushing. The complete version requires the full v4 FE chain")
print("which is 460 lines and exceeds the 150-line doc cap; lift it from")
print("kernels/p1-single-cb-v4-gpu/p1_single_cb_v4_gpu.py at push time.")
