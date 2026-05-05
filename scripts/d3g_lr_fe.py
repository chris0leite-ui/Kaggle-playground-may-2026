"""D3-G — Logistic Regression with careful FE (linear-family base).

PI ask Day-3: add LR-with-FE as a structurally-different family. The
current 13-base pool is dominated by 10 GBDT consensus clones (per
disagreement diagnostic). A linear model with explicit FE has a
fundamentally different inductive bias — no implicit interactions,
monotonic in features unless made non-monotone — and could break
the consensus rank on hard rows.

Feature engineering plan:
  Numeric (raw + scaled): LapNumber, TyreLife, Position, LapTime_s,
    LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change,
    Stint, Year, PitStop
  + sequence features (leak-free): cum_pits_this_race, laps_since_last_pit
    (from d3b recipe; PitStop-only, no target dependence)
  + nonlinear transforms: LapNumber², TyreLife², Position², log(LapTime+1)
  + interactions: Stint×LapNumber, Compound×TyreLife,
    RaceProgress×TyreLife (signal: "late race + worn tyres → likely pit")

  Categorical (one-hot): Compound (5 levels), Year (4 levels)
  Categorical (target-encoded, OOF-safe): Driver, Race
    α=80 (matches d2a_te recipe).

LR with C=1.0, L2 penalty, class_weight='balanced' (handles 20/80 prior).
StandardScaler on all numeric features (LR is scale-sensitive).

R1: Strat-only.

Output: oof_d3g_lr_fe_strat.npy, test_d3g_lr_fe_strat.npy, results.json,
submission.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import N_FOLDS, SEED, save_oof

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
ALPHA = 80.0
TE_KEYS = ["Driver", "Race"]
ONEHOT_COLS = ["Compound", "Year"]
NUMERIC_COLS = [
    "LapNumber", "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
    "Cumulative_Degradation", "RaceProgress", "Position_Change",
    "Stint", "PitStop",
]


def add_seqfe(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-free sequence FE (cum_pits_this_race, laps_since_last_pit)."""
    df = df.copy()
    sort_idx = df.sort_values(["Race", "Driver", "LapNumber"]).index
    df_s = df.loc[sort_idx].copy()
    grp = df_s.groupby(["Race", "Driver"], sort=False)
    df_s["cum_pits_this_race"] = grp["PitStop"].cumsum()
    df_s["_last_pit_marker"] = df_s["LapNumber"].where(df_s["PitStop"] == 1)
    df_s["_last_pit_lap"] = grp["_last_pit_marker"].ffill()
    df_s["laps_since_last_pit"] = (df_s["LapNumber"] - df_s["_last_pit_lap"]).fillna(
        df_s["LapNumber"]
    )
    df_s = df_s.drop(columns=["_last_pit_marker", "_last_pit_lap"])
    return df_s.loc[df.index]


def smoothed_te(train_y, train_key, apply_key, alpha):
    global_mean = float(train_y.mean())
    df = pd.DataFrame({"k": train_key, "y": train_y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    g["te"] = (g["sum"] + alpha * global_mean) / (g["count"] + alpha)
    return pd.Series(apply_key).map(g["te"]).fillna(global_mean).to_numpy()


def oof_te(y, key, alpha, n_inner=5, seed=42):
    out = np.zeros(len(y), dtype=np.float64)
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, va in kf.split(np.zeros(len(y))):
        out[va] = smoothed_te(y[tr], key[tr], key[va], alpha)
    return out


def build_features(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray):
    """Build feature matrices for LR. Returns (X_train_base, X_test_base,
    other args needed for per-fold TE)."""
    train_fe = add_seqfe(train)
    test_fe = add_seqfe(test)

    # Numeric features + nonlinear + interactions (no TE yet)
    def transform(df):
        out = pd.DataFrame(index=df.index)
        for c in NUMERIC_COLS + ["cum_pits_this_race", "laps_since_last_pit"]:
            out[c] = df[c].astype(float)
        # nonlinear transforms
        out["LapNumber_sq"] = out["LapNumber"] ** 2
        out["TyreLife_sq"] = out["TyreLife"] ** 2
        out["Position_sq"] = out["Position"] ** 2
        out["LapTime_log"] = np.log1p(out["LapTime (s)"].clip(lower=0))
        # interactions
        out["Stint_x_LapNumber"] = out["Stint"] * out["LapNumber"]
        out["Compound_TyreLife"] = (df["Compound"].astype("category").cat.codes
                                    .astype(float) * out["TyreLife"])
        out["RaceProgress_x_TyreLife"] = out["RaceProgress"] * out["TyreLife"]
        # one-hot
        for c in ONEHOT_COLS:
            d = pd.get_dummies(df[c], prefix=c, drop_first=True)
            out = pd.concat([out, d], axis=1)
        return out

    X_train = transform(train_fe)
    X_test = transform(test_fe)
    # Align columns (train one-hot may differ from test)
    missing = set(X_train.columns) - set(X_test.columns)
    for c in missing:
        X_test[c] = 0.0
    extra = set(X_test.columns) - set(X_train.columns)
    for c in extra:
        X_train[c] = 0.0
    X_test = X_test[X_train.columns]

    return X_train, X_test, train_fe, test_fe


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    print("Building base features (numeric + nonlinear + onehot + interactions)...")
    X_train, X_test, train_fe, test_fe = build_features(train, test, y)
    print(f"X_train shape: {X_train.shape}  X_test shape: {X_test.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(len(test), dtype=np.float32)
    fold_scores = []

    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        # Per-fold TE on outer-train only
        Xtr = X_train.copy()
        Xte = X_test.copy()
        for key in TE_KEYS:
            inner_oof = oof_te(y[tr], train_fe[key].values[tr], ALPHA,
                               n_inner=5, seed=SEED)
            te_va = smoothed_te(y[tr], train_fe[key].values[tr],
                                train_fe[key].values[va], ALPHA)
            te_test = smoothed_te(y[tr], train_fe[key].values[tr],
                                  test_fe[key].values, ALPHA)
            te_col = np.zeros(len(Xtr), dtype=np.float64)
            te_col[tr] = inner_oof
            te_col[va] = te_va
            Xtr[f"te_{key}"] = te_col
            Xte[f"te_{key}"] = te_test

        # Standardize
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # Replace any non-finite values from log/division (paranoid)
        Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_s = np.nan_to_num(Xte_s, nan=0.0, posinf=0.0, neginf=0.0)

        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                class_weight="balanced", n_jobs=-1)
        lr.fit(Xtr_s[tr], y[tr])
        p_va = lr.predict_proba(Xtr_s[va])[:, 1]
        oof[va] = p_va
        test_proba += lr.predict_proba(Xte_s)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        fold_scores.append(s)
        print(f"  fold {k}: AUC={s:.5f}  wall={time.time()-t0:.0f}s  "
              f"feats={Xtr_s.shape[1]}")

    auc_full = float(roc_auc_score(y, oof))
    delta_bp = (auc_full - 0.94075) * 1e4
    print(f"\nLR-FE Strat OOF: {auc_full:.5f}  std={np.std(fold_scores):.5f}  "
          f"Δ baseline={delta_bp:+.1f}bp")

    save_oof("d3g_lr_fe_strat",
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(oof_score=auc_full, fold_std=float(np.std(fold_scores)),
                  fold_scores=fold_scores,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_bp,
                  feature_count=int(X_train.shape[1] + len(TE_KEYS)),
                  notes="LR with FE (numeric+nonlinear+onehot+TE+interactions); "
                        "linear family base for stack diversity"))

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_proba
    sample_sub.to_csv("submissions/submission_d3g_lr_fe.csv", index=False)


if __name__ == "__main__":
    main()
