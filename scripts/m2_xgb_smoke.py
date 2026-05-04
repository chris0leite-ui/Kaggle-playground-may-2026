"""M2 XGB smoke test — 1 fold, 50k rows subsample, reduced trees.

Verifies the XGB native-categorical pipeline runs end-to-end. Wall ≤ 2 min target.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED

TARGET = "PitNextLap"
ID_COL = "id"
SUBSAMPLE = 50_000


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(train), size=SUBSAMPLE, replace=False)
    train = train.iloc[idx].reset_index(drop=True)

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    print(f"loaded subsample {len(train)} rows; cats={cat_cols}; "
          f"load_secs={time.time()-t0:.1f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))

    t1 = time.time()
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=20,
        n_estimators=500,
        early_stopping_rounds=50,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    p = model.predict_proba(X.iloc[va])[:, 1]
    auc = float(roc_auc_score(y[va], p))
    fit_secs = time.time() - t1
    print(f"smoke fold0 AUC={auc:.5f} best_iter={model.best_iteration} "
          f"fit_secs={fit_secs:.1f}")
    print(f"total wall={time.time()-t0:.1f}s  OK")


if __name__ == "__main__":
    main()
