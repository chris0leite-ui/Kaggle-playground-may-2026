"""D2-A smoke — 1 fold, 50k rows, verify TE pipeline runs end-to-end."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED
from d2a_target_encoding import (
    ALPHA, TE_INTERACTIONS, TE_KEYS, build_keys, make_lgb_params,
    oof_te_train, smoothed_te,
)

TARGET = "PitNextLap"
ID_COL = "id"


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    train = train.sample(n=50_000, random_state=SEED).reset_index(drop=True)
    train = build_keys(train)
    print(f"smoke train: {train.shape}  t={time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    all_te_keys = TE_KEYS + [f"{a}_{b}" for a, b in TE_INTERACTIONS]

    base_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    X = train[base_cols].copy()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(iter(skf.split(np.zeros(len(y)), y)))

    Xtr_aug = X.copy()
    for k in all_te_keys:
        full_key = train[k].values
        inner_oof = oof_te_train(y[tr_idx], full_key[tr_idx], ALPHA, n_inner=5, seed=SEED)
        te_va = smoothed_te(y[tr_idx], full_key[tr_idx], full_key[va_idx], ALPHA)
        te_col = np.zeros(len(Xtr_aug), dtype=np.float64)
        te_col[tr_idx] = inner_oof
        te_col[va_idx] = te_va
        Xtr_aug[f"te_{k}"] = te_col

    print(f"TE built  t={time.time()-t0:.1f}s")
    dtrain = lgb.Dataset(Xtr_aug.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
    dval = lgb.Dataset(Xtr_aug.iloc[va_idx], y[va_idx], categorical_feature=cat_cols)
    model = lgb.train(make_lgb_params(), dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    p = model.predict(Xtr_aug.iloc[va_idx])
    auc = roc_auc_score(y[va_idx], p)
    print(f"smoke fold AUC={auc:.5f}  best_iter={model.best_iteration}  total t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
