"""Smoke probe: e3 HGBC fold0 timing for d12 monolithic-bag-probe.

Single seed, single fold, full data. Prints wall time → projects 5-seed
× 5-fold cost.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID_COL = "PitNextLap", "id"
N_FOLDS = 5
HIGH_CARD = ["Driver"]
LOW_CARD = ["Compound", "Race"]


def make_hgbc(seed):
    return HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=seed,
        categorical_features="from_dtype",
    )


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mp).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mp).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr, va = splits[0]
    print(f"[setup] data load {time.time()-t0:.1f}s, n_train={len(y)}, n_test={len(X_test)}")

    t1 = time.time()
    m = make_hgbc(seed=42)
    m.fit(X.iloc[tr], y[tr])
    p_va = m.predict_proba(X.iloc[va])[:, 1]
    p_te = m.predict_proba(X_test)[:, 1]
    wall = time.time() - t1
    auc = roc_auc_score(y[va], p_va)
    print(f"[fold0/seed42] AUC={auc:.5f} iters={m.n_iter_} wall={wall:.1f}s")

    proj_one_seed_5fold = wall * 5
    proj_5seed_5fold = wall * 5 * 5
    print(f"\n=== projection ===")
    print(f"  1 seed × 5 fold: {proj_one_seed_5fold:.0f}s ({proj_one_seed_5fold/60:.1f} min)")
    print(f"  5 seed × 5 fold: {proj_5seed_5fold:.0f}s ({proj_5seed_5fold/60:.1f} min)")
    print(f"  3 seed × 5 fold: {wall*5*3:.0f}s ({wall*5*3/60:.1f} min)")


if __name__ == "__main__":
    main()
