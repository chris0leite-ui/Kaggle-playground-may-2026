"""scripts/d15_physics_residual.py — Lens 3: physics-residual base.

Idea: F1 lap times have a near-deterministic structure:
  LapTime ≈ baseline(Driver, Race, Year) + degradation(Compound, TyreLife)
            + fuel_burn(LapNumber) + traffic(Position) + ε

The synthesizer corrupts joint structure, but the marginal physics
holds. Fit a robust regression of LapTime on physics features, then
use residuals as a base. Residuals isolate "race-state" signal that
is decoupled from the GBDT pool's main signal sources.

A second residual: Cumulative_Degradation has anomalies (negative
values, very large values) that reveal synthesizer over-extrapolation;
build a "cumulative degradation noise score" by per-Race-Compound
quantile-residual.

Combined LGBM with these residual features as the base.
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)
SEED, N_FOLDS = 42, 5
TARGET = "PitNextLap"


def main():
    print("Loading data...")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].astype(int).values

    # === Build physics-residual features ===
    for df in [tr, te]:
        # Per-stint position within stint
        g = df.groupby(["Driver", "Race", "Year", "Stint"])
        df["stint_max_tl"] = g["TyreLife"].transform("max").clip(lower=1)
        df["stint_min_lap"] = g["LapNumber"].transform("min")
        df["NTL_estimate"] = (df["TyreLife"] / df["stint_max_tl"]).astype(np.float32)
        df["lap_into_stint"] = (df["LapNumber"] - df["stint_min_lap"]).astype(np.float32)

        # Race-level total laps (proxy)
        gr = df.groupby(["Race", "Year"])
        df["race_total_laps"] = gr["LapNumber"].transform("max").clip(lower=1)
        df["race_progress_lap"] = (df["LapNumber"] / df["race_total_laps"]).astype(np.float32)

    # === Physics-residual: LapTime ~ baseline(driver,race,year) + compound × TyreLife ===
    # Use Ridge with one-hot on Driver,Race,Year + Compound + numerical interaction TyreLife*Compound
    # Done in 5-fold OOF on TRAIN + apply once on TEST (so the residual is leak-free).
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = ["Driver", "Compound", "Race", "Year"]
    # Convert to string for OHE
    tr_cat = tr[cat_cols].astype(str)
    te_cat = te[cat_cols].astype(str)

    # Numerical physics features
    num_cols = ["TyreLife", "LapNumber", "Stint", "Position",
                "RaceProgress", "lap_into_stint", "race_progress_lap"]
    tr_num = tr[num_cols].fillna(0).values.astype(np.float32)
    te_num = te[num_cols].fillna(0).values.astype(np.float32)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe.fit(pd.concat([tr_cat, te_cat], axis=0))

    tr_oh = ohe.transform(tr_cat)
    te_oh = ohe.transform(te_cat)

    from scipy.sparse import hstack, csr_matrix
    tr_X = hstack([tr_oh, csr_matrix(tr_num)]).tocsr()
    te_X = hstack([te_oh, csr_matrix(te_num)]).tocsr()
    print(f"  feature space: {tr_X.shape}")

    # === Fit OOF Ridge for LapTime residual ===
    lt = tr["LapTime (s)"].fillna(tr["LapTime (s)"].median()).values.astype(np.float32)
    lt_te = te["LapTime (s)"].fillna(te["LapTime (s)"].median()).values.astype(np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lt_resid = np.zeros(len(tr), dtype=np.float32)
    lt_pred_te = np.zeros(len(te), dtype=np.float32)
    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        rid = Ridge(alpha=10.0)
        rid.fit(tr_X[tr_idx], lt[tr_idx])
        lt_resid[va_idx] = lt[va_idx] - rid.predict(tr_X[va_idx])
        lt_pred_te += rid.predict(te_X) / N_FOLDS

    lt_resid_te = lt_te - lt_pred_te
    print(f"  ridge LapTime fit: train MAE={np.abs(lt_resid).mean():.3f}s, test pred MAE={np.abs(lt_resid_te).mean():.3f}s ({time.time()-t0:.0f}s)")

    tr["lt_resid_phys"] = lt_resid
    te["lt_resid_phys"] = lt_resid_te

    # === Cumulative_Degradation residual: log-clipped + per-Race-Compound z-score ===
    for df in [tr, te]:
        df["cumdeg_clip"] = df["Cumulative_Degradation"].clip(-50, 200)
    cd_mean = tr.groupby(["Race", "Compound"])["cumdeg_clip"].transform("mean")
    cd_std = tr.groupby(["Race", "Compound"])["cumdeg_clip"].transform("std").clip(lower=1)
    tr["cumdeg_z_phys"] = (tr["cumdeg_clip"] - cd_mean) / cd_std
    # For test, use train's group stats
    tr_stats = tr.groupby(["Race", "Compound"])["cumdeg_clip"].agg(["mean", "std"]).reset_index()
    tr_stats["std"] = tr_stats["std"].clip(lower=1)
    te = te.merge(tr_stats, on=["Race", "Compound"], how="left", suffixes=("", "_g"))
    te["cumdeg_z_phys"] = (te["cumdeg_clip"] - te["mean"]) / te["std"]
    te["cumdeg_z_phys"] = te["cumdeg_z_phys"].fillna(0)

    print(f"  lt_resid_phys (train) range: [{tr['lt_resid_phys'].min():.2f}, {tr['lt_resid_phys'].max():.2f}], std={tr['lt_resid_phys'].std():.2f}")
    print(f"  cumdeg_z_phys (train) range: [{tr['cumdeg_z_phys'].min():.2f}, {tr['cumdeg_z_phys'].max():.2f}]")

    # === LGBM with physics-residual features ===
    cat_cols_lgb = ["Driver", "Compound", "Race"]
    for c in cat_cols_lgb:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
        te[c] = te[c].cat.set_categories(tr[c].cat.categories)

    feature_cols = [
        "Driver", "Compound", "Race", "Year",
        "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
        # Physics-residual additions
        "lt_resid_phys", "cumdeg_z_phys",
        "NTL_estimate", "lap_into_stint", "race_progress_lap",
    ]
    print(f"\n  features ({len(feature_cols)}): {feature_cols}")

    X = tr[feature_cols]
    Xte = te[feature_cols]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr), dtype=np.float64)
    test_pred = np.zeros(len(te), dtype=np.float64)

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        n_jobs=-1,
        seed=SEED,
    )

    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols_lgb)
        dval = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols_lgb, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
        oof[va_idx] = model.predict(X.iloc[va_idx])
        test_pred += model.predict(Xte) / N_FOLDS
        fa = roc_auc_score(y[va_idx], oof[va_idx])
        print(f"  fold {k}: AUC={fa:.5f}  best_iter={model.best_iteration}  ({time.time()-t0:.0f}s)")

    auc = roc_auc_score(y, oof)
    print(f"\n=== d15_physics_residual OOF AUC: {auc:.5f} ===")

    oof2 = np.column_stack([1 - oof, oof])
    test2 = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_d15_physics_residual_strat.npy", oof2)
    np.save(ART / "test_d15_physics_residual_strat.npy", test2)
    print(f"  → saved oof/test_d15_physics_residual_strat.npy")


if __name__ == "__main__":
    main()
