"""β — HGBC architectural variants for stack diversity.

Two HGBC variants with different leaf budgets + seeds to add
architectural diversity (not just bagging variance reduction):
- F1: max_leaf_nodes=127 (deeper), seed=123
- F2: max_leaf_nodes=31 (shallower), seed=7
Both with label-encoded Driver, native cat for Race+Compound (same
encoding as E3 HGBC). Two-anchor 5-fold each. Adds 2 new bases to
the M5c pool for M5d refit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import N_FOLDS, SEED, save_oof

TARGET, ID_COL = "PitNextLap", "id"
BASE_S, BASE_G = 0.94075, 0.92059

VARIANTS = [
    ("f1_hgbc_deep", dict(max_iter=2000, learning_rate=0.05, max_leaf_nodes=127,
                           min_samples_leaf=100, l2_regularization=0.1,
                           early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=50, random_state=123,
                           categorical_features="from_dtype")),
    ("f2_hgbc_shallow", dict(max_iter=1500, learning_rate=0.05, max_leaf_nodes=31,
                              min_samples_leaf=400, l2_regularization=0.1,
                              early_stopping=True, validation_fraction=0.1,
                              n_iter_no_change=50, random_state=7,
                              categorical_features="from_dtype")),
]


def run_anchor(name, splits, X, y, X_test, hp):
    oof = np.zeros(len(y), dtype=np.float32)
    tp = np.zeros(len(X_test), dtype=np.float32)
    fs, walls = [], []
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        m = HistGradientBoostingClassifier(**hp)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        tp += m.predict_proba(X_test)[:, 1] / N_FOLDS
        wall = time.time() - t0
        s = float(roc_auc_score(y[va], p))
        fs.append(s); walls.append(wall)
        print(f"  [{name}] f{k}: AUC={s:.5f} iters={m.n_iter_} wall={wall:.1f}s")
    return oof, tp, float(roc_auc_score(y, oof)), fs, float(np.std(fs)), walls


def main():
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).values

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    HIGH_CARD = ["Driver"]
    LOW_CARD = ["Compound", "Race"]
    for c in HIGH_CARD:
        if c in X.columns:
            uniq = pd.concat([X[c], X_test[c]], ignore_index=True).astype(str).unique()
            mapping = {v: i for i, v in enumerate(sorted(uniq))}
            X[c] = X[c].astype(str).map(mapping).astype(np.int32)
            X_test[c] = X_test[c].astype(str).map(mapping).astype(np.int32)
    for c in LOW_CARD:
        if c in X.columns:
            X[c] = X[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits_a = list(skf.split(np.zeros(len(y)), y))
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_b = list(gkf.split(np.zeros(len(y)), y, train["Race"].values))

    for name, hp in VARIANTS:
        print(f"\n=== {name} (max_leaf_nodes={hp['max_leaf_nodes']}, seed={hp['random_state']}) ===")
        print("--- Strat ---")
        oof_a, test_a, auc_a, fs_a, sd_a, w_a = run_anchor(name, splits_a, X, y, X_test, hp)
        print("--- GroupKF ---")
        oof_b, test_b, auc_b, fs_b, sd_b, w_b = run_anchor(name, splits_b, X, y, X_test, hp)
        da = (auc_a - BASE_S) * 1e4; db = (auc_b - BASE_G) * 1e4
        print(f"  Strat: {auc_a:.5f}  Δ={da:+.1f}bp")
        print(f"  GroupKF: {auc_b:.5f}  Δ={db:+.1f}bp")

        save_oof(f"{name}_strat",
                 np.column_stack([1 - oof_a, oof_a]),
                 np.column_stack([1 - test_a, test_a]),
                 dict(oof_score=auc_a, fold_std=sd_a, fold_scores=fs_a,
                      cv="StratKF", metric="roc_auc",
                      delta_vs_baseline_bp=da, hp=hp))
        save_oof(f"{name}_groupkf",
                 np.column_stack([1 - oof_b, oof_b]),
                 np.column_stack([1 - test_b, test_b]),
                 dict(oof_score=auc_b, fold_std=sd_b, fold_scores=fs_b,
                      cv="GroupKF(Race)", metric="roc_auc",
                      delta_vs_baseline_bp=db, hp=hp))

        sample_sub[TARGET] = test_a
        sample_sub.to_csv(f"submissions/submission_{name}.csv", index=False)

    total = time.time() - t0
    print(f"\nβ total wall: {total:.0f}s")
    body = f"# β — HGBC variants (2026-05-04)\n\nWall {total:.0f}s. Variants: {[v[0] for v in VARIANTS]}.\nSee individual *_results.json for two-anchor scores.\n"
    Path("audit/2026-05-04-beta-hgbc-variants.md").write_text(body)


if __name__ == "__main__":
    main()
