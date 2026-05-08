"""Smoke probe: CatBoost lossguide fold0 timing for d12 bag-probe.

Single seed, single fold, full data. Lossguide CPU variant (no GPU).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID_COL = "PitNextLap", "id"
CAT_COLS = ["Driver", "Race", "Compound", "Year"]
N_FOLDS = 5

P = dict(iterations=800, learning_rate=0.08, l2_leaf_reg=3.0,
         random_seed=42, eval_metric="AUC", od_type="Iter", od_wait=50,
         verbose=0, thread_count=-1, allow_writing_files=False,
         grow_policy="Lossguide", num_leaves=64, max_depth=8)


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[0]
    print(f"[setup] load {time.time()-t0:.1f}s, n_train={len(y)}")

    t1 = time.time()
    ptr = Pool(X.iloc[tr], y[tr], cat_features=CAT_COLS)
    pva = Pool(X.iloc[va], y[va], cat_features=CAT_COLS)
    m = CatBoostClassifier(**P)
    m.fit(ptr, eval_set=pva)
    p_va = m.predict_proba(pva)[:, 1]
    wall = time.time() - t1
    auc = roc_auc_score(y[va], p_va)
    bi = int(m.get_best_iteration())
    print(f"[fold0/seed42 lossguide] AUC={auc:.5f} bi={bi} wall={wall:.1f}s")

    proj_5seed_5fold = wall * 5 * 5
    print(f"\n=== projection ===")
    print(f"  5 seed × 5 fold: {proj_5seed_5fold:.0f}s ({proj_5seed_5fold/60:.1f} min)")
    print(f"  3 seed × 5 fold: {wall*5*3:.0f}s ({wall*5*3/60:.1f} min)")


if __name__ == "__main__":
    main()
