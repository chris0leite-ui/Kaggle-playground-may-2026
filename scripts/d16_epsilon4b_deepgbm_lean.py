"""Day-16 ε4b — DeepGBM-style (lean): LGBM-leaf-encoding -> sparse-LR head.

Replaces ε4 (cat-LGBM stage-2 was over-engineered with 627 cat features,
killed mid-stage-2). This version:
  Stage 1: smaller LGBM (300 trees, num_leaves=31) -> leaf indices.
  Stage 2: sparse one-hot of leaf indices -> Logistic regression.

Same DeepGBM idea but Stage-2 is convex (LR) so no leaf-cat-LGBM
combinatorial explosion. ~5 min wall.

Output:
  oof_d16_epsilon4b_deepgbm_lean_strat.npy   (n_train, 2)
  test_d16_epsilon4b_deepgbm_lean_strat.npy  (n_test, 2)
  d16_epsilon4b_deepgbm_lean_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from scipy.sparse import hstack as sparse_hstack

ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
CATS = ["Driver", "Compound", "Race", "Year"]

STAGE1_PARAMS = dict(objective="binary", metric="auc",
                     num_leaves=31, learning_rate=0.05,
                     min_child_samples=200, feature_fraction=0.85,
                     bagging_fraction=0.85, bagging_freq=1,
                     verbose=-1, seed=SEED)
STAGE1_BOOST = 300


def _onehot_leaves(leaves: np.ndarray, n_leaves_per_tree: int) -> csr_matrix:
    """Encode (n_rows, n_trees) integer leaf indices into a sparse one-hot
    matrix of shape (n_rows, n_trees * n_leaves_per_tree)."""
    n_rows, n_trees = leaves.shape
    rows = np.repeat(np.arange(n_rows), n_trees)
    cols = (np.arange(n_trees) * n_leaves_per_tree)[None, :].repeat(n_rows, axis=0)
    cols = (cols + leaves).ravel()
    data = np.ones(n_rows * n_trees, dtype=np.float32)
    return csr_matrix((data, (rows, cols)),
                      shape=(n_rows, n_trees * n_leaves_per_tree),
                      dtype=np.float32)


def main():
    t0 = time.time()
    print("[ε4b] loading data ...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).values
    n_train, n_test = len(train), len(test)
    print(f"[ε4b] train {n_train}  test {n_test}", flush=True)

    encoders = {}
    full = pd.concat([train[CATS], test[CATS]], axis=0, ignore_index=True)
    for c in CATS:
        vals = full[c].astype(str).unique().tolist()
        encoders[c] = {v: i for i, v in enumerate(vals)}
    for df in (train, test):
        for c in CATS:
            df[c + "_idx"] = df[c].astype(str).map(encoders[c]).astype(np.int32)

    feat_cols = NUMERICS + [c + "_idx" for c in CATS]
    cat_idx_cols = [c + "_idx" for c in CATS]
    Xtr = train[feat_cols]
    Xte = test[feat_cols]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n_train), y))

    oof = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(n_test, dtype=np.float32)

    for fold, (tr, va) in enumerate(splits):
        # Stage 1
        dtr = lgb.Dataset(Xtr.iloc[tr], y[tr], categorical_feature=cat_idx_cols)
        dva = lgb.Dataset(Xtr.iloc[va], y[va], categorical_feature=cat_idx_cols,
                          reference=dtr)
        m1 = lgb.train(STAGE1_PARAMS, dtr, num_boost_round=STAGE1_BOOST,
                       valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        n_iter = m1.best_iteration or STAGE1_BOOST
        leaves_tr = m1.predict(Xtr.iloc[tr], pred_leaf=True,
                               num_iteration=n_iter).astype(np.int32)
        leaves_va = m1.predict(Xtr.iloc[va], pred_leaf=True,
                               num_iteration=n_iter).astype(np.int32)
        leaves_te = m1.predict(Xte, pred_leaf=True,
                               num_iteration=n_iter).astype(np.int32)

        # Pad to n_iter cols if early-stop-trimmed
        if leaves_tr.shape[1] != n_iter:
            n_iter = leaves_tr.shape[1]
        n_leaves_per_tree = STAGE1_PARAMS["num_leaves"]
        Xs_tr = _onehot_leaves(leaves_tr, n_leaves_per_tree)
        Xs_va = _onehot_leaves(leaves_va, n_leaves_per_tree)
        Xs_te = _onehot_leaves(leaves_te, n_leaves_per_tree)

        # Stage 2: sparse LR
        lr = LogisticRegression(C=1.0, max_iter=300, solver="saga",
                                penalty="l2", n_jobs=-1)
        lr.fit(Xs_tr, y[tr])
        oof[va] = lr.predict_proba(Xs_va)[:, 1]
        test_pred += lr.predict_proba(Xs_te)[:, 1] / N_FOLDS
        fold_auc = roc_auc_score(y[va], oof[va])
        print(f"  fold {fold}: stage1_iter={n_iter}  "
              f"sparse_dim={Xs_tr.shape[1]}  fold_AUC={fold_auc:.5f}", flush=True)

    full_auc = float(roc_auc_score(y, oof))
    print(f"\n[ε4b] OOF AUC = {full_auc:.6f}", flush=True)

    np.save(ART / "oof_d16_epsilon4b_deepgbm_lean_strat.npy",
            np.column_stack([1.0 - oof, oof]))
    np.save(ART / "test_d16_epsilon4b_deepgbm_lean_strat.npy",
            np.column_stack([1.0 - test_pred, test_pred]))
    res = dict(stage1_n_trees=STAGE1_BOOST,
               stage1_num_leaves=STAGE1_PARAMS["num_leaves"],
               standalone_oof_auc=full_auc,
               n_train=n_train, n_test=n_test,
               wall_s=time.time() - t0)
    (ART / "d16_epsilon4b_deepgbm_lean_results.json").write_text(json.dumps(res, indent=2))
    print(f"[ε4b] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
