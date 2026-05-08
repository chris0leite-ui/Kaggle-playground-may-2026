"""D2-A 1-fold time-probe — full data, fold 0 only, measure wall time.

Projects 5-fold both-anchor cost. If projection ≥1h → shrink config.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import N_FOLDS, SEED
from d2a_target_encoding import (
    ALPHA, TE_INTERACTIONS, TE_KEYS, build_keys, make_lgb_params,
    oof_te_train, smoothed_te,
)

TARGET = "PitNextLap"
ID_COL = "id"


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train = build_keys(train)
    test = build_keys(test)
    print(f"loaded: train {train.shape}, test {test.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    all_te_keys = TE_KEYS + [f"{a}_{b}" for a, b in TE_INTERACTIONS]

    base_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    X = train[base_cols].copy()
    X_test = test[[c for c in base_cols if c in test.columns]].copy()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    print(f"cat_cols: {cat_cols}  t={time.time()-t0:.1f}s")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(iter(skf.split(np.zeros(len(y)), y)))

    t_te = time.time()
    Xtr_aug = X.copy()
    Xte_aug = X_test.copy()
    for k in all_te_keys:
        full_key = train[k].values
        inner_oof = oof_te_train(y[tr_idx], full_key[tr_idx], ALPHA, n_inner=5, seed=SEED)
        te_va = smoothed_te(y[tr_idx], full_key[tr_idx], full_key[va_idx], ALPHA)
        te_test = smoothed_te(y[tr_idx], full_key[tr_idx], test[k].values, ALPHA)
        te_col = np.zeros(len(Xtr_aug), dtype=np.float64)
        te_col[tr_idx] = inner_oof
        te_col[va_idx] = te_va
        Xtr_aug[f"te_{k}"] = te_col
        Xte_aug[f"te_{k}"] = te_test
    te_secs = time.time() - t_te
    print(f"TE built (per fold)  te_time={te_secs:.1f}s  t={time.time()-t0:.1f}s")

    t_fit = time.time()
    dtrain = lgb.Dataset(Xtr_aug.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
    dval = lgb.Dataset(Xtr_aug.iloc[va_idx], y[va_idx], categorical_feature=cat_cols)
    model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    fit_secs = time.time() - t_fit
    p_va = model.predict(Xtr_aug.iloc[va_idx])
    p_test = model.predict(Xte_aug)
    auc = roc_auc_score(y[va_idx], p_va)
    print(f"fold 0 AUC={auc:.5f}  best_iter={model.best_iteration}")
    print(f"fit_time={fit_secs:.1f}s  test_pred_time={time.time()-t_fit-fit_secs:.1f}s")

    per_fold = te_secs + fit_secs
    proj_5fold = per_fold * 5
    proj_two_anchor = proj_5fold * 2
    print(f"\nper-fold: {per_fold:.1f}s   5-fold proj: {proj_5fold:.0f}s ({proj_5fold/60:.1f}m)")
    print(f"two-anchor (10 fits) proj: {proj_two_anchor:.0f}s ({proj_two_anchor/60:.1f}m)")
    print(f"1h gate: {'OK' if proj_two_anchor < 3600 else 'EXCEEDS — shrink config'}")


if __name__ == "__main__":
    main()
