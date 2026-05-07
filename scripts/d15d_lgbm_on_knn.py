"""scripts/d15d_lgbm_on_knn.py — LightGBM on raw + KNN features.

Loads original 11 numeric + 4 categorical (label-encoded) features and
appends the 10 KNN distance features from d15d_knn_features.py. Trains
a 5-fold StratifiedKFold LightGBM and saves OOF / test in 2-col format.

Inputs:
  data/train.csv, data/test.csv
  scripts/artifacts/d15d_knn_X_train.npy  (439140, 10)
  scripts/artifacts/d15d_knn_X_test.npy   (188165, 10)

Outputs:
  scripts/artifacts/oof_d15d_lgbm_knn_strat.npy   (439140, 2)
  scripts/artifacts/test_d15d_lgbm_knn_strat.npy  (188165, 2)
  scripts/artifacts/d15d_lgbm_knn_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb


ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED = 42
N_FOLDS = 5

NUMERIC_COLS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop", "Year",
]
CAT_COLS = ["Driver", "Compound", "Race", "Year"]
KNN_FEAT_NAMES = [
    "comp_mean", "comp_min", "comp_max", "comp_std", "comp_top1",
    "drv_mean", "drv_min", "drv_max", "drv_std", "drv_top1",
]


def main() -> None:
    t0 = time.time()
    print("=== d15d LGBM on raw + KNN features ===")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    print(f"Train: {len(train):,}  Test: {len(test):,}  pos rate: {y.mean():.4f}")

    # Numeric features
    X_num_tr = train[NUMERIC_COLS].astype(np.float32).values
    X_num_te = test[NUMERIC_COLS].astype(np.float32).values

    # Label-encode categoricals on train+test combined
    cat_arrays_tr = []
    cat_arrays_te = []
    for c in CAT_COLS:
        le = LabelEncoder()
        combined = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
        le.fit(combined.values)
        cat_arrays_tr.append(le.transform(train[c].astype(str).values).astype(np.int32))
        cat_arrays_te.append(le.transform(test[c].astype(str).values).astype(np.int32))
    X_cat_tr = np.column_stack(cat_arrays_tr)
    X_cat_te = np.column_stack(cat_arrays_te)
    print(f"Cat encoded shape: tr={X_cat_tr.shape} te={X_cat_te.shape}")

    # KNN features
    X_knn_tr = np.load(ART / "d15d_knn_X_train.npy").astype(np.float32)
    X_knn_te = np.load(ART / "d15d_knn_X_test.npy").astype(np.float32)
    assert X_knn_tr.shape == (len(train), 10), X_knn_tr.shape
    assert X_knn_te.shape == (len(test), 10), X_knn_te.shape
    print(f"KNN feat shape: tr={X_knn_tr.shape} te={X_knn_te.shape}")

    # Stack:  numeric (11) + cat (4) + knn (10) = 25 cols
    # The cat cols include Year, which is ALSO in numeric. We pass categorical
    # indices to LGBM so the cat-version is treated as a category. Year as
    # numeric is fine to keep too (small redundancy).
    X_tr = np.hstack([X_num_tr, X_cat_tr.astype(np.float32), X_knn_tr])
    X_te = np.hstack([X_num_te, X_cat_te.astype(np.float32), X_knn_te])

    feat_names = (
        NUMERIC_COLS
        + [f"cat_{c}" for c in CAT_COLS]
        + KNN_FEAT_NAMES
    )
    cat_idx = list(range(len(NUMERIC_COLS), len(NUMERIC_COLS) + len(CAT_COLS)))
    print(f"Final X: tr={X_tr.shape} te={X_te.shape}  cat_idx={cat_idx}")
    print(f"Features: {feat_names}")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    fold_aucs = []
    fold_walls = []
    fold_iters = []

    params = dict(
        objective="binary",
        metric="auc",
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        verbose=-1,
        seed=SEED,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        tf = time.time()
        dtr = lgb.Dataset(
            X_tr[tr_idx], label=y[tr_idx],
            feature_name=feat_names, categorical_feature=cat_idx,
        )
        dva = lgb.Dataset(
            X_tr[va_idx], label=y[va_idx],
            feature_name=feat_names, categorical_feature=cat_idx,
            reference=dtr,
        )
        model = lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        va_pred = model.predict(X_tr[va_idx], num_iteration=model.best_iteration)
        te_pred = model.predict(X_te, num_iteration=model.best_iteration)
        oof[va_idx] = va_pred
        test_pred += te_pred / N_FOLDS

        auc = float(roc_auc_score(y[va_idx], va_pred))
        wall = time.time() - tf
        fold_aucs.append(auc)
        fold_walls.append(wall)
        fold_iters.append(int(model.best_iteration))
        print(f"  f{fold}: AUC={auc:.5f}  iters={model.best_iteration}  wall={wall:.1f}s")

    auc_oof = float(roc_auc_score(y, oof))
    print(f"\nStandalone OOF AUC: {auc_oof:.5f}")
    print(f"Per-fold AUCs: {[f'{a:.5f}' for a in fold_aucs]}")
    print(f"Total wall: {time.time() - t0:.1f}s")

    # 2-col canonical save
    oof_2col = np.column_stack([1 - oof, oof])
    test_2col = np.column_stack([1 - test_pred, test_pred])
    np.save(ART / "oof_d15d_lgbm_knn_strat.npy", oof_2col)
    np.save(ART / "test_d15d_lgbm_knn_strat.npy", test_2col)
    print(f"Saved: oof_d15d_lgbm_knn_strat.npy {oof_2col.shape}")
    print(f"       test_d15d_lgbm_knn_strat.npy {test_2col.shape}")

    # ρ vs PRIMARY
    primary = np.load(ART / "test_d13e_compound_stint_tau20000_strat.npy")[:, 1]
    from scipy.stats import spearmanr
    rho, _ = spearmanr(test_pred, primary)
    print(f"ρ (Spearman) vs PRIMARY (d13e_compound_stint_tau20000): {rho:.4f}")

    results = {
        "auc_oof": auc_oof,
        "fold_aucs": fold_aucs,
        "fold_walls": fold_walls,
        "fold_iters": fold_iters,
        "rho_vs_primary": float(rho),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feat_names": feat_names,
        "n_features": int(X_tr.shape[1]),
        "total_wall_s": float(time.time() - t0),
    }
    (ART / "d15d_lgbm_knn_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved results JSON.")


if __name__ == "__main__":
    main()
