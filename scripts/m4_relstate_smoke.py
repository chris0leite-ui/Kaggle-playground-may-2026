"""M4 smoke — 1 fold StratKFold, 50k row subsample. Verify FE pipeline."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED

TARGET = "PitNextLap"
ID_COL = "id"
LAPTIME_COL = "LapTime (s)"


def make_lgb_params() -> dict:
    return dict(
        objective="binary", learning_rate=0.05, num_leaves=63,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        min_data_in_leaf=200, verbose=-1, seed=SEED,
    )


def add_relstate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Sort by (Race, Driver, LapNumber); add relative-state features.
    Caller is responsible for restoring original row order afterwards.

    Returns (df_with_features, added_names, skipped_names).
    """
    df = df.sort_values(["Race", "Driver", "LapNumber"], kind="stable")

    candidates = [
        "Position_Change", "LapTime_Delta", "RaceProgress",
        "Cumulative_Degradation", "Recent_Degradation",
        "Traffic_Pressure_Proxy",
    ]
    present = set(df.columns)
    added, skipped = [], []

    if "Position_Change" not in present:
        df["Position_Change"] = (df["Position"]
                                  - df.groupby(["Race", "Driver"])["Position"].shift(1)
                                  ).fillna(0)
        added.append("Position_Change")
    else:
        skipped.append("Position_Change")

    if "LapTime_Delta" not in present:
        df["LapTime_Delta"] = (df[LAPTIME_COL]
                                 - df.groupby(["Race", "Driver"])[LAPTIME_COL].shift(1)
                                 ).fillna(0)
        added.append("LapTime_Delta")
    else:
        skipped.append("LapTime_Delta")

    if "RaceProgress" not in present:
        df["RaceProgress"] = df["LapNumber"] / df.groupby("Race")["LapNumber"].transform("max")
        added.append("RaceProgress")
    else:
        skipped.append("RaceProgress")

    if "Cumulative_Degradation" not in present:
        first_lap = df.groupby(["Race", "Driver", "Stint"])[LAPTIME_COL].transform("first")
        df["Cumulative_Degradation"] = (df[LAPTIME_COL] - first_lap).fillna(0)
        added.append("Cumulative_Degradation")
    else:
        skipped.append("Cumulative_Degradation")

    # Recent_Degradation: rolling mean window=3 of LapTime_Delta within (Race, Driver, Stint)
    df["Recent_Degradation"] = (
        df.groupby(["Race", "Driver", "Stint"])["LapTime_Delta"]
          .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    ).fillna(0)
    added.append("Recent_Degradation")

    # Traffic_Pressure_Proxy: Position - min(Position) per (Race, LapNumber)
    df["Traffic_Pressure_Proxy"] = (df["Position"]
                                      - df.groupby(["Race", "LapNumber"])["Position"].transform("min"))
    added.append("Traffic_Pressure_Proxy")

    return df, added, skipped


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    train = train.sample(n=50_000, random_state=SEED).reset_index(drop=True)
    print(f"smoke train: {train.shape}  t={time.time()-t0:.1f}s")

    orig_id = train[ID_COL].values.copy()
    train_fe, added, skipped = add_relstate_features(train)
    train_fe = train_fe.set_index(ID_COL).loc[orig_id].reset_index()
    assert (train_fe[ID_COL].values == orig_id).all(), "row order restore failed"

    print(f"added: {added}")
    print(f"skipped (already present): {skipped}")

    y = train_fe[TARGET].astype(int).values
    X = train_fe.drop(columns=[TARGET, ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    print(f"feature columns ({len(X.columns)}): {list(X.columns)}")
    print(f"NaN check: {X.isna().sum().sum()} total NaN cells across X")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))

    dtrain = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
    dval = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols)
    model = lgb.train(make_lgb_params(), dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p = model.predict(X.iloc[va])
    auc = roc_auc_score(y[va], p)
    print(f"smoke fold AUC={auc:.5f}  best_iter={model.best_iteration}  total t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
