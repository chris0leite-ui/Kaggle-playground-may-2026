"""Smoke test: T1.2 multi-formulation, 1-fold, 50k rows.

Validates pipeline + estimates wall before committing to the 5-fold run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from d12_t12_multi_formulation import (
    encode_features, build_laps_until_pit_grouped, build_ratio_target,
    build_stint_dataset, fit_lgbm_censored_regression, fit_lgbm_ratio,
    fit_stint_survival_simple,
)

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5
N_SMOKE = 20_000


def main():
    t0 = time.time()
    train_df = pd.read_csv("data/train.csv").iloc[:N_SMOKE].copy()
    test_df = pd.read_csv("data/test.csv").iloc[:20_000].copy()
    y = train_df[TARGET].astype(int).values
    n_train = len(train_df)

    train_df["_src"] = "train"; test_df["_src"] = "test"; test_df[TARGET] = -1
    df_all = pd.concat([train_df, test_df], ignore_index=True)

    df_all_c = build_laps_until_pit_grouped(df_all)
    df_all_c = df_all_c.sort_values("id", kind="stable").reset_index(drop=True)
    train_c = df_all_c[df_all_c["_src"] == "train"].copy()
    laps_until = train_c["laps_until_next_pit"].astype(np.float64).values
    observed = train_c["is_observed"].astype(np.int8).values

    df_all_d = build_ratio_target(df_all_c)
    df_all_d = df_all_d.sort_values("id", kind="stable").reset_index(drop=True)
    train_d = df_all_d[df_all_d["_src"] == "train"].copy()
    test_d = df_all_d[df_all_d["_src"] == "test"].copy()
    ratio_target = train_d["pit_to_stint_ratio"].astype(np.float64).values
    stint_age_train = train_d["laps_into_stint"].astype(np.float64).values
    stint_age_test = test_d["laps_into_stint"].astype(np.float64).values
    mean_stint_len = float(df_all_d.groupby(
        ["Driver", "Race", "Year", "Stint"], observed=True
    )["LapNumber"].count().mean())

    drop_cols = [TARGET, ID_COL, "_src", "laps_until_next_pit",
                 "is_observed", "pit_to_stint_ratio", "stint_min_lap"]
    X_train = train_d.drop(columns=drop_cols, errors="ignore").copy()
    X_test = test_d.drop(columns=drop_cols, errors="ignore").copy()
    X_train, X_test = encode_features(X_train.copy(), X_test.copy())
    print(f"smoke shapes: train {X_train.shape} test {X_test.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    splits_one = [splits[0]]

    print("\n=== SMOKE T1.2c (Censored Regression) ===")
    t1 = time.time()
    oof_c, test_c, auc_c = fit_lgbm_censored_regression(
        X_train, X_test, laps_until, observed, splits_one, y,
        censored_weight=0.3)
    print(f"  smoke wall {time.time()-t1:.1f}s")

    print("\n=== SMOKE T1.2d (Ratio) ===")
    t1 = time.time()
    oof_d, test_d_arr, auc_d = fit_lgbm_ratio(
        X_train, X_test, ratio_target, splits_one, y,
        stint_age_train, stint_age_test, mean_stint_len)
    print(f"  smoke wall {time.time()-t1:.1f}s")

    print("\n=== SMOKE T1.2e (Stint Survival) ===")
    t1 = time.time()
    stint_df = build_stint_dataset(df_all_d)
    oof_e, test_e, auc_e = fit_stint_survival_simple(
        stint_df, splits_one, df_all_d, y, n_train)
    print(f"  smoke wall {time.time()-t1:.1f}s")

    print(f"\nTotal smoke wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
