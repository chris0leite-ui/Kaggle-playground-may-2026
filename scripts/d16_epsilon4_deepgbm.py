"""Day-16 ε4 — DeepGBM-style: LGBM-leaf-encoding -> 2nd-stage stack.

ε4 axis (pool-architecture). One strong LGBM (5-fold OOF) emits per-tree
leaf-indices for each row -> (N, n_trees) matrix of integer categoricals.
A second-stage LGBM (or sparse-LR) trained on those leaf categoricals
captures non-linear interactions between trees that the K=21 LR-meta
cannot form (LR-meta is linear in [raw, rank, logit]).

Pipeline:
  Stage 1: LGBM 5-fold on raw features -> OOF leaf-indices, test leaf-indices.
  Stage 2: LGBM 5-fold on leaf-indices (categorical) -> OOF + test preds.

Output (under scripts/artifacts/):
  oof_d16_epsilon4_deepgbm_strat.npy   (n_train, 2)
  test_d16_epsilon4_deepgbm_strat.npy  (n_test, 2)
  d16_epsilon4_deepgbm_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SEED, N_FOLDS = 42, 5

NUMERICS = [
    "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "PitStop",
]
CATS = ["Driver", "Compound", "Race", "Year"]

STAGE1_PARAMS = dict(objective="binary", metric="auc",
                     num_leaves=63, learning_rate=0.05,
                     min_child_samples=200, feature_fraction=0.85,
                     bagging_fraction=0.85, bagging_freq=1,
                     verbose=-1, seed=SEED)
STAGE1_BOOST = 600
STAGE2_BOOST = 600
STAGE2_PARAMS = dict(objective="binary", metric="auc",
                     num_leaves=255, learning_rate=0.05,
                     min_child_samples=200, feature_fraction=0.50,
                     bagging_fraction=0.85, bagging_freq=1,
                     verbose=-1, seed=SEED+1)


def main():
    t0 = time.time()
    print("[ε4] loading data ...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["PitNextLap"].astype(int).values
    n_train, n_test = len(train), len(test)
    print(f"[ε4] train {n_train}  test {n_test}", flush=True)

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

    # Stage 1: LGBM, get leaf indices per fold
    print("[ε4] Stage 1: 5-fold LGBM (raw features) -> leaves ...", flush=True)
    leaves_tr = np.zeros((n_train, STAGE1_BOOST), dtype=np.int32)
    leaves_te = np.zeros((n_test, STAGE1_BOOST), dtype=np.int32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(Xtr.iloc[tr], y[tr], categorical_feature=cat_idx_cols)
        dva = lgb.Dataset(Xtr.iloc[va], y[va], categorical_feature=cat_idx_cols,
                          reference=dtr)
        model = lgb.train(STAGE1_PARAMS, dtr, num_boost_round=STAGE1_BOOST,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        n_iter = model.best_iteration or STAGE1_BOOST
        # Predict leaf indices for valid + test (use n_iter trees only)
        leaves_va = model.predict(Xtr.iloc[va], pred_leaf=True,
                                  num_iteration=n_iter).astype(np.int32)
        leaves_te_fold = model.predict(Xte, pred_leaf=True,
                                       num_iteration=n_iter).astype(np.int32)
        # Pad/truncate to STAGE1_BOOST cols
        if leaves_va.shape[1] < STAGE1_BOOST:
            pad = np.zeros((leaves_va.shape[0],
                            STAGE1_BOOST - leaves_va.shape[1]), dtype=np.int32)
            leaves_va = np.concatenate([leaves_va, pad], axis=1)
            pad = np.zeros((leaves_te_fold.shape[0],
                            STAGE1_BOOST - leaves_te_fold.shape[1]), dtype=np.int32)
            leaves_te_fold = np.concatenate([leaves_te_fold, pad], axis=1)
        leaves_tr[va] = leaves_va[:, :STAGE1_BOOST]
        leaves_te += leaves_te_fold[:, :STAGE1_BOOST]
        print(f"  fold {fold}: best_iter={n_iter}", flush=True)
    leaves_te = (leaves_te / N_FOLDS).round().astype(np.int32)

    # Stage 2: LGBM on leaf categoricals
    print("[ε4] Stage 2: 5-fold LGBM (leaf categorical) ...", flush=True)
    leaf_col_names = [f"L{i}" for i in range(STAGE1_BOOST)]
    Xtr_leaf = pd.DataFrame(leaves_tr, columns=leaf_col_names)
    Xte_leaf = pd.DataFrame(leaves_te, columns=leaf_col_names)
    cat_leaf = leaf_col_names

    oof2 = np.zeros(n_train, dtype=np.float32)
    test2_pred = np.zeros(n_test, dtype=np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(Xtr_leaf.iloc[tr], y[tr], categorical_feature=cat_leaf)
        dva = lgb.Dataset(Xtr_leaf.iloc[va], y[va], categorical_feature=cat_leaf,
                          reference=dtr)
        model = lgb.train(STAGE2_PARAMS, dtr, num_boost_round=STAGE2_BOOST,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        n_iter = model.best_iteration or STAGE2_BOOST
        oof2[va] = model.predict(Xtr_leaf.iloc[va], num_iteration=n_iter)
        test2_pred += model.predict(Xte_leaf, num_iteration=n_iter) / N_FOLDS
        print(f"  fold {fold}: best_iter={n_iter}", flush=True)

    auc = float(roc_auc_score(y, oof2))
    print(f"\n[ε4] Stage-2 OOF AUC = {auc:.6f}", flush=True)

    np.save(ART / "oof_d16_epsilon4_deepgbm_strat.npy",
            np.column_stack([1.0 - oof2, oof2]))
    np.save(ART / "test_d16_epsilon4_deepgbm_strat.npy",
            np.column_stack([1.0 - test2_pred, test2_pred]))
    res = dict(stage1_n_trees=STAGE1_BOOST,
               stage2_n_trees=STAGE2_BOOST,
               standalone_oof_auc=auc,
               n_train=n_train, n_test=n_test,
               wall_s=time.time() - t0)
    (ART / "d16_epsilon4_deepgbm_results.json").write_text(json.dumps(res, indent=2))
    print(f"[ε4] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
