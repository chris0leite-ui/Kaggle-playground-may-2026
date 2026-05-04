"""D3-E — EBM (Explainable Boosting Machine) base, Strat 5-fold.

PI request: add a non-GBDT mechanism family to the pool. EBM is a
GA²M (Generalized Additive Model with pairwise interactions) — fits
shape functions per feature and learnable pairwise interactions.
Inductive bias is fundamentally different from xgb/lgbm/cb (which
are gradient-boosted trees).

interpret-core ExplainableBoostingClassifier handles native cats
internally, no manual encoding needed.

R1: Strat-only (R1: GroupKF dropped Day-3+).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import N_FOLDS, SEED, save_oof

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
BASE_S = 0.94075


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    print(f"train={X.shape}  test={X_test.shape}")

    # interpret-core EBM accepts mixed dtypes; pass categoricals as object
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"cat_cols: {cat_cols}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    oof = np.zeros(len(y), dtype=np.float32)
    test_proba = np.zeros(len(test), dtype=np.float32)
    fold_scores = []
    fold_walls = []

    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        # Defaults are decent; tune only if smoke fold is fast and AUC is weak
        ebm = ExplainableBoostingClassifier(
            random_state=SEED + k,
            n_jobs=-1,
            max_bins=256,
            max_interaction_bins=64,
            interactions=10,            # 10 pairwise interactions
            outer_bags=4,                # 4 inner bagged rounds
            inner_bags=0,
            learning_rate=0.05,
            max_rounds=2500,
            early_stopping_rounds=50,
            early_stopping_tolerance=1e-5,
        )
        ebm.fit(X.iloc[tr], y[tr])
        p_va = ebm.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p_va
        test_proba += ebm.predict_proba(X_test)[:, 1] / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        wall = time.time() - t0
        fold_scores.append(s)
        fold_walls.append(wall)
        print(f"  fold {k}: AUC={s:.5f}  wall={wall:.0f}s ({wall/60:.1f}min)")

        # Crash-safe save after every fold
        save_oof("d3e_ebm_strat",
                 np.column_stack([1 - oof, oof]),
                 np.column_stack([1 - test_proba, test_proba]),
                 dict(oof_score=float(roc_auc_score(y[oof != 0], oof[oof != 0])
                                      if (oof != 0).any() else 0.0),
                      partial_folds=k + 1, fold_scores=fold_scores,
                      fold_walls=fold_walls,
                      cv="StratifiedKFold(5)", metric="roc_auc"))

    auc_full = float(roc_auc_score(y, oof))
    delta_bp = (auc_full - BASE_S) * 1e4
    print(f"\nEBM Strat OOF: {auc_full:.5f}  std={np.std(fold_scores):.5f}  "
          f"Δ baseline={delta_bp:+.1f}bp")

    save_oof("d3e_ebm_strat",
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(oof_score=auc_full, fold_std=float(np.std(fold_scores)),
                  fold_scores=fold_scores, fold_walls=fold_walls,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_bp,
                  notes="EBM (interpret-core ExplainableBoostingClassifier); GA²M family"))

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_proba
    sample_sub.to_csv("submissions/submission_d3e_ebm.csv", index=False)


if __name__ == "__main__":
    main()
