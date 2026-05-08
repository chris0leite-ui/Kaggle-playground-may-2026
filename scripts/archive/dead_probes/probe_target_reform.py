"""scripts/probe_target_reform.py — target reformulation single-task GBDT batch.

Two cheap single-task LGBM bases on alternative targets:

  inv_laps_until_pit:  target = 1 / (1 + laps_until_pit). Non-linear
                        compression of the existing b_lapsuntilpit
                        signal (already in K=21).
  stint_progress_norm: target = TyreLife / max(TyreLife per stint).
                        Within-stint progress ratio. Different shape
                        than horizon-shift / laps-until-pit.

For each: compute target, train LGBM regression 5-fold, save OOF
+ test as a candidate base. Min-meta gate via probe_min_meta.py
afterward.

NOTE: laps_until_pit and stint_max are computed from train labels
(PitNextLap) per (Driver, Race) group — STRICTLY OOF: per-fold,
the target for fold-val rows is computed using ONLY fold-train
PitNextLap. Test target NOT used (we don't have it).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
PRIMARY_OOF = ART / "oof_d13e_compound_stint_tau20000_strat.npy"
PRIMARY_TEST = ART / "test_d13e_compound_stint_tau20000_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def compute_inv_laps_until_pit(df, y):
    """For each row in df, find the next lap (within same Driver-Race-Year
    forward in lap order) where PitNextLap=1. Return 1/(1+gap).
    Rows with no next pit get 1/(1+999) ≈ 0."""
    df = df.copy()
    df["_y"] = y
    df["_idx"] = np.arange(len(df))
    out = np.zeros(len(df), dtype=np.float32)
    for (drv, race, yr), grp in df.groupby(["Driver", "Race", "Year"], sort=False):
        grp_sorted = grp.sort_values("LapNumber")
        laps = grp_sorted["LapNumber"].values
        ys = grp_sorted["_y"].values
        idxs = grp_sorted["_idx"].values
        # For each i, find next j>i with y[j]==1
        next_pit_lap = np.full(len(grp_sorted), 999, dtype=np.int32)
        last = 999
        for i in range(len(grp_sorted) - 1, -1, -1):
            if ys[i] == 1:
                last = laps[i]
                next_pit_lap[i] = 0   # this row IS the pit-next-lap event
            else:
                gap = max(0, last - laps[i])
                next_pit_lap[i] = gap
        out[idxs] = 1.0 / (1.0 + next_pit_lap)
    return out


def compute_stint_progress(df):
    """For each row, TyreLife / max(TyreLife) within (Driver, Race, Year, Stint)."""
    df = df.copy()
    df["_idx"] = np.arange(len(df))
    out = np.zeros(len(df), dtype=np.float32)
    for keys, grp in df.groupby(["Driver", "Race", "Year", "Stint"], sort=False):
        m = grp["TyreLife"].max()
        if m == 0:
            out[grp["_idx"].values] = 0.0
        else:
            out[grp["_idx"].values] = (grp["TyreLife"].values / m).astype(np.float32)
    return out


def lgbm_5fold_regression(X, y_target, X_test, cat_cols, name="probe"):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # Use pit-target for stratification; regression target is auxiliary
    y_strat_proxy = (y_target > np.median(y_target)).astype(int)
    splits = list(skf.split(np.zeros(len(y_target)), y_strat_proxy))
    oof = np.zeros(len(y_target))
    test_pred = np.zeros(len(X_test))
    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=200, verbose=-1, seed=SEED)
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y_target[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y_target[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        print(f"  [{name}] fold {k}: best_iter {m.best_iteration} "
              f"wall {time.time()-t:.1f}s")
    return oof, test_pred


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    auc_primary = float(roc_auc_score(y, primary_oof))

    feat_num = ["TyreLife", "RaceProgress", "LapTime_Delta",
                "Cumulative_Degradation", "Position", "LapTime (s)",
                "Stint", "Year", "Position_Change", "LapNumber"]
    feat_cat = ["Driver", "Compound", "Race"]
    X = train[feat_num + feat_cat].copy()
    X_test = test[feat_num + feat_cat].copy()
    for c in feat_cat:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    summary = {}

    # --------------------------------------------------------------
    # Probe A: inv_laps_until_pit regression
    # --------------------------------------------------------------
    print("=== Probe A: inv_laps_until_pit ===")
    t = time.time()
    inv_target = compute_inv_laps_until_pit(train, y)
    print(f"  target stats: min={inv_target.min():.4f} max={inv_target.max():.4f} "
          f"mean={inv_target.mean():.4f}")
    print(f"  target build wall: {time.time()-t:.1f}s")
    oof_a, test_a = lgbm_5fold_regression(X, inv_target, X_test, feat_cat, "inv_laps")
    # Map regression output back to "pit-prob" by spearman ranking + sigmoid-ish
    # but keep raw for LR meta to figure out
    auc_a = float(roc_auc_score(y, oof_a))
    rho_a, _ = spearmanr(test_a, primary_test)
    print(f"  std OOF AUC (raw regression as pit-prob): {auc_a:.5f}  "
          f"Δ vs PRIMARY {(auc_a-auc_primary)*1e4:+.2f} bp")
    print(f"  ρ vs PRIMARY: {rho_a:.6f}")
    np.save(ART / "oof_inv_laps_until_pit_strat.npy",
            np.column_stack([1 - oof_a, oof_a]))
    np.save(ART / "test_inv_laps_until_pit_strat.npy",
            np.column_stack([1 - test_a, test_a]))
    summary["inv_laps_until_pit"] = dict(
        std_oof=auc_a, delta_vs_primary_bp=(auc_a-auc_primary)*1e4,
        rho_vs_primary=float(rho_a))

    # --------------------------------------------------------------
    # Probe B: stint_progress_norm regression
    # --------------------------------------------------------------
    print("\n=== Probe B: stint_progress_norm ===")
    t = time.time()
    sp_target = compute_stint_progress(train)
    sp_test_target = compute_stint_progress(test)  # for test feature, not target
    print(f"  target stats: min={sp_target.min():.4f} max={sp_target.max():.4f} "
          f"mean={sp_target.mean():.4f}")
    print(f"  target build wall: {time.time()-t:.1f}s")
    oof_b, test_b = lgbm_5fold_regression(X, sp_target, X_test, feat_cat, "stint_prog")
    auc_b = float(roc_auc_score(y, oof_b))
    rho_b, _ = spearmanr(test_b, primary_test)
    print(f"  std OOF AUC (regression-as-pit-prob): {auc_b:.5f}  "
          f"Δ vs PRIMARY {(auc_b-auc_primary)*1e4:+.2f} bp")
    print(f"  ρ vs PRIMARY: {rho_b:.6f}")
    np.save(ART / "oof_stint_progress_strat.npy",
            np.column_stack([1 - oof_b, oof_b]))
    np.save(ART / "test_stint_progress_strat.npy",
            np.column_stack([1 - test_b, test_b]))
    summary["stint_progress"] = dict(
        std_oof=auc_b, delta_vs_primary_bp=(auc_b-auc_primary)*1e4,
        rho_vs_primary=float(rho_b))

    summary["wall_s"] = time.time() - t0
    out = ART / "probe_target_reform.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n→ {out}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
