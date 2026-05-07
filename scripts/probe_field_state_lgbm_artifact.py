"""scripts/probe_field_state_lgbm_artifact.py — Generate harness-format
OOF and test predictions for a field-state LGBM, for K=24 stack-add gate.

Uses the STRICT per-fold field-state aggregation (matches the +13.35 bp
honest-OOF result from probe_field_state_strict.py F4). Test predictions
use full-train+test combined aggregates (transductive at inference time
is fine — the val-fold protocol is what we need for OOF integrity).

Output:
  scripts/artifacts/oof_field_state_lgbm_strat.npy  (n_train, 2)
  scripts/artifacts/test_field_state_lgbm_strat.npy (n_test, 2)
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    verbose=-1, n_jobs=-1, seed=SEED,
)
SOURCE_CAT_COLS = ["Driver", "Compound", "Race"]
CAT_COLS = ["Driver_cat", "Compound_cat", "Race_cat"]
RAW_FEATS = [
    "Driver_cat", "Compound_cat", "Race_cat", "Year",
    "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
FS_FEATS = [
    "fs_field_size", "fs_n_pitting_now", "fs_pit_rate_now",
    "fs_mean_TyreLife", "fs_max_TyreLife", "fs_min_TyreLife",
    "fs_std_TyreLife", "fs_mean_Stint", "fs_max_Stint",
    "fs_mean_Position", "fs_mean_LapTime", "fs_mean_RaceProgress",
    "fs_cum_pits", "fs_cum_pit_lap_count", "fs_cum_pit_rate",
    "fs_compound_n", "fs_compound_n_pitting", "fs_compound_pit_rate",
    "fs_compound_mean_TyreLife", "fs_compound_max_TyreLife",
    "fs_TyreLife_vs_field_mean", "fs_TyreLife_vs_field_max",
    "fs_Position_vs_field_mean", "fs_Stint_vs_field_mean",
]


def encode_cats(*dfs: pd.DataFrame) -> None:
    for c in SOURCE_CAT_COLS:
        all_vals = pd.concat([d[c].astype(str) for d in dfs])
        codes, _ = all_vals.factorize()
        cuts = np.cumsum([0] + [len(d) for d in dfs])
        for i, d in enumerate(dfs):
            d[f"{c}_cat"] = codes[cuts[i]:cuts[i + 1]].astype("int32")


def build_field_state(source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    rs = (source_df.sort_values(["Race", "Year", "LapNumber"])
                  .groupby(["Race", "Year", "LapNumber"])["PitStop"]
                  .sum().reset_index())
    rs["fs_cum_pits"] = rs.groupby(["Race", "Year"])["PitStop"].cumsum()
    rs["fs_cum_pit_lap_count"] = (rs.groupby(["Race", "Year"])["PitStop"]
                                    .cumcount() + 1)
    rs["fs_cum_pit_rate"] = rs["fs_cum_pits"] / rs["fs_cum_pit_lap_count"]
    a = a.merge(rs[["Race", "Year", "LapNumber",
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
    return a, ac


def merge_field_state(df: pd.DataFrame, a: pd.DataFrame,
                      ac: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(a, on=["Race", "Year", "LapNumber"], how="left")
    out = out.merge(ac, on=["Race", "Year", "LapNumber", "Compound"],
                    how="left")
    out["fs_TyreLife_vs_field_mean"] = (out["TyreLife"]
                                        - out["fs_mean_TyreLife"])
    out["fs_TyreLife_vs_field_max"] = (out["TyreLife"]
                                       - out["fs_max_TyreLife"])
    out["fs_Position_vs_field_mean"] = (out["Position"]
                                        - out["fs_mean_Position"])
    out["fs_Stint_vs_field_mean"] = (out["Stint"]
                                     - out["fs_mean_Stint"])
    return out


def main() -> None:
    print("Loading data...")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    encode_cats(tr, te)
    tr = tr.sort_values("id").reset_index(drop=True)
    te = te.sort_values("id").reset_index(drop=True)
    y = tr[TARGET].astype(int).to_numpy()

    feats = RAW_FEATS + FS_FEATS

    # ===== Strict per-fold OOF (matches probe_field_state_strict F4) =====
    print(f"\n[OOF] Strict per-fold field-state, train-only aggregates...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []
    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr_in_fold = tr.iloc[tr_idx].copy()
        va_rows = tr.iloc[va_idx].copy()
        a, ac = build_field_state(tr_in_fold)
        tr_fs = merge_field_state(tr_in_fold, a, ac)
        va_fs = merge_field_state(va_rows, a, ac)
        dtrain = lgb.Dataset(tr_fs[feats], label=tr_fs[TARGET].astype(int).values,
                             categorical_feature=CAT_COLS)
        dval = lgb.Dataset(va_fs[feats], label=va_fs[TARGET].astype(int).values,
                           categorical_feature=CAT_COLS, reference=dtrain)
        m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(va_fs[feats])
        fa = roc_auc_score(y[va_idx], oof[va_idx])
        fold_aucs.append(float(fa))
        print(f"  fold {k}: AUC={fa:.5f}  best_iter={m.best_iteration}  ({time.time() - t0:.0f}s)")
    overall = float(roc_auc_score(y, oof))
    print(f"  Strict OOF AUC = {overall:.5f}  fold_std = {np.std(fold_aucs):.5f}")

    # ===== Test predictions: train on FULL train, aggregates on combined ===
    print(f"\n[TEST] Full-train fit, combined-frame aggregates for test...")
    combined = pd.concat([tr, te], ignore_index=True)
    a_full, ac_full = build_field_state(combined)
    tr_full_fs = merge_field_state(tr, a_full, ac_full)
    te_full_fs = merge_field_state(te, a_full, ac_full)
    # Use 5-fold bagging (same SEED) for test predictions for stability
    test_pred = np.zeros(len(te))
    t1 = time.time()
    for k, (tr_idx, _) in enumerate(skf.split(np.zeros(len(y)), y)):
        sub = tr_full_fs.iloc[tr_idx]
        dtrain = lgb.Dataset(sub[feats], label=sub[TARGET].astype(int).values,
                             categorical_feature=CAT_COLS)
        m = lgb.train(LGB_PARAMS, dtrain, num_boost_round=600)  # fixed iters
        test_pred += m.predict(te_full_fs[feats]) / N_FOLDS
        print(f"  test fold {k} done  ({time.time() - t1:.0f}s)")

    # ===== Save in harness format (n, 2) ===============================
    oof2 = np.column_stack([1.0 - oof, oof])
    test2 = np.column_stack([1.0 - test_pred, test_pred])
    np.save(ART / "oof_field_state_lgbm_strat.npy", oof2)
    np.save(ART / "test_field_state_lgbm_strat.npy", test2)
    print(f"\n  → saved oof/test_field_state_lgbm_strat.npy")
    print(f"  OOF AUC: {overall:.5f}")
    print(f"  test_pred mean: {test_pred.mean():.4f}  std: {test_pred.std():.4f}")


if __name__ == "__main__":
    main()
