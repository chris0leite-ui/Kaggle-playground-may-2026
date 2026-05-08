"""M3 CatBoost smoke — 1 fold StratKFold, 50k subsample, ~500 iters.

Goal: end-to-end pipeline sanity check. Wall ≤ 2 min target.
Prints fold AUC and timing breakdown.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train), size=50000, replace=False)
    train = train.iloc[idx].reset_index(drop=True)
    print(f"loaded subsample: {train.shape} in {time.time()-t0:.1f}s")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.zeros(len(y)), y)))

    t1 = time.time()
    pool_tr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
    pool_va = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=50,
        verbose=0,
        thread_count=-1,
        allow_writing_files=False,
    )
    model.fit(pool_tr, eval_set=pool_va)
    fit_t = time.time() - t1

    p = model.predict_proba(X.iloc[va])[:, 1]
    auc = float(roc_auc_score(y[va], p))
    print(f"fold AUC = {auc:.5f}  best_iter={model.get_best_iteration()}  "
          f"fit={fit_t:.1f}s  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
