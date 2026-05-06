"""scripts/probe_kd.py — knowledge distillation: small LGBM mimics K=21 OOF.

Synthetic-data lens: if the K=21 stack's predictive structure is
compressible into a single small model, that compression error
(or its prediction) is itself information for the LR meta.

Train a small LightGBM (depth 4, ~200 trees) on raw features, with
target = K=21 LR-meta OOF (the d12_lr_meta artifact is the K=21 LR
meta OOF). Use squared loss for regression-on-probabilities. Save
the distilled model's OOF + test as a 22nd-base candidate.
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
KD_TARGET_OOF = ART / "oof_d12_lr_meta_strat.npy"        # K=21 LR-meta OOF
KD_TARGET_TEST = ART / "test_d12_lr_meta_strat.npy"


def _pos(p):
    a = np.load(p)
    return a[:, 1].astype(np.float64) if a.ndim == 2 else a.ravel()


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    primary_oof = _pos(PRIMARY_OOF)
    primary_test = _pos(PRIMARY_TEST)
    soft_target_oof = _pos(KD_TARGET_OOF)        # what we mimic
    soft_target_test = _pos(KD_TARGET_TEST)
    print(f"KD target = K=21 LR-meta OOF; AUC {roc_auc_score(y, soft_target_oof):.5f}")

    feat_cols = ["TyreLife", "RaceProgress", "LapTime_Delta",
                 "Cumulative_Degradation", "Position", "LapTime (s)",
                 "Stint", "Year", "Position_Change", "LapNumber",
                 "Driver", "Compound", "Race"]
    cat_cols = ["Driver", "Compound", "Race"]
    X = train[feat_cols].copy()
    X_test = test[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    # Soft-label MSE regression (logit space → smoother target)
    eps = 1e-6
    soft_logit = np.log(np.clip(soft_target_oof, eps, 1-eps) /
                         (1 - np.clip(soft_target_oof, eps, 1-eps)))
    soft_logit_test_target = np.log(
        np.clip(soft_target_test, eps, 1-eps) /
        (1 - np.clip(soft_target_test, eps, 1-eps)))

    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.05, num_leaves=31, max_depth=5,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  min_data_in_leaf=400, verbose=-1, seed=SEED)
    oof_kd = np.zeros(len(y))
    test_kd = np.zeros(len(test))
    for k, (tr, va) in enumerate(splits):
        t = time.time()
        dtr = lgb.Dataset(X.iloc[tr], soft_logit[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], soft_logit[va], categorical_feature=cat_cols)
        m = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        oof_kd[va] = m.predict(X.iloc[va])
        test_kd += m.predict(X_test) / N_FOLDS
        print(f"  fold {k}: best_iter {m.best_iteration} wall {time.time()-t:.1f}s")

    # Convert KD logits back to probability for AUC
    oof_kd_p = 1.0 / (1.0 + np.exp(-np.clip(oof_kd, -30, 30)))
    test_kd_p = 1.0 / (1.0 + np.exp(-np.clip(test_kd, -30, 30)))
    auc_kd = float(roc_auc_score(y, oof_kd_p))
    rho, _ = spearmanr(test_kd_p, primary_test)
    rho_target, _ = spearmanr(test_kd_p, soft_target_test)
    print(f"\nKD distilled OOF AUC: {auc_kd:.5f}")
    print(f"  ρ(KD test, PRIMARY test):       {rho:.6f}")
    print(f"  ρ(KD test, K=21 LR-meta test):  {rho_target:.6f}")

    np.save(ART / "oof_kd_lgbm_strat.npy",
            np.column_stack([1 - oof_kd_p, oof_kd_p]))
    np.save(ART / "test_kd_lgbm_strat.npy",
            np.column_stack([1 - test_kd_p, test_kd_p]))
    summary = dict(
        kd_target_oof_auc=float(roc_auc_score(y, soft_target_oof)),
        kd_distilled_oof_auc=auc_kd,
        rho_vs_primary=float(rho),
        rho_vs_target=float(rho_target),
        wall_s=time.time() - t0,
    )
    (ART / "probe_kd.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ scripts/artifacts/probe_kd.json (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
