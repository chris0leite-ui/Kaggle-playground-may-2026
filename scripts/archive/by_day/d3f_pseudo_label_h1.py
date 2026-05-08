"""D3-F — Pseudo-label H1 with multi-base agreement guard.

Strategy critique 2026-05-04 flagged H1 as a Day-3 lever:
  "pseudo-labeling guarded by multi-base agreement (≥10/13 of M5h)"

Recipe:
  1. For each test row, count how many of 13 M5h pool bases predict >0.5.
  2. Confidence-2-tail subset = rows where ≥10/13 bases agree on the
     same direction.
     - "high-confidence pit": ≥10/13 bases predict >0.5
     - "high-confidence no-pit": ≤3/13 bases predict >0.5
  3. Add those test rows with their predicted class as pseudo-labels
     to train.
  4. Retrain a single LGBM (baseline_two_anchor params) on the
     augmented training set.
  5. Generate fresh OOF + test predictions.
  6. Save as new pool member: pseudo_lgbm.

R1: Strat-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import N_FOLDS, SEED, save_oof

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
BASE_S = 0.94075

POOL = [
    ("baseline", "baseline_two_anchor"),
    ("d2a_te", "d2a_te"),
    ("m2_xgb", "m2_xgb"),
    ("e1_cb_sub", "e1_catboost_sub"),
    ("e3_hgbc", "e3_hgbc"),
    ("e5_optuna_lgbm", "e5_optuna_lgbm"),
    ("a_horizon", "a_horizon"),
    ("b_lapsuntilpit", "b_lapsuntilpit"),
    ("f1_hgbc_deep", "f1_hgbc_deep"),
    ("f2_hgbc_shallow", "f2_hgbc_shallow"),
    ("cb_year-cat", "cb_year-cat"),
    ("cb_lossguide", "cb_lossguide"),
    ("cb_slow-wide-bag", "cb_slow-wide-bag"),
]
AGREE_HI = 10   # ≥10/13 bases predict >0.5 → pseudo-label "1"
AGREE_LO = 3    # ≤3/13 bases predict >0.5 → pseudo-label "0"


def make_lgb_params() -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=63,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_data_in_leaf=200, verbose=-1, seed=SEED)


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore")
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # Build agreement matrix on test predictions
    print("Loading 13 base test predictions for agreement matrix...")
    test_votes = np.zeros(len(test), dtype=np.int32)
    for _, name in POOL:
        p_test = np.load(ART / f"test_{name}_strat.npy")[:, 1].astype(np.float64)
        test_votes += (p_test > 0.5).astype(np.int32)

    pseudo_pos_mask = test_votes >= AGREE_HI
    pseudo_neg_mask = test_votes <= AGREE_LO
    pseudo_mask = pseudo_pos_mask | pseudo_neg_mask
    n_pos = int(pseudo_pos_mask.sum())
    n_neg = int(pseudo_neg_mask.sum())
    n_total = int(pseudo_mask.sum())
    print(f"Agreement guard: ≥{AGREE_HI}/13 → pos pseudo-label, "
          f"≤{AGREE_LO}/13 → neg pseudo-label")
    print(f"  high-confidence pos: {n_pos} ({100*n_pos/len(test):.1f}% of test)")
    print(f"  high-confidence neg: {n_neg} ({100*n_neg/len(test):.1f}% of test)")
    print(f"  total pseudo-labelable: {n_total} ({100*n_total/len(test):.1f}% of test)")

    if n_total < 1000:
        print("WARNING: <1000 high-confidence rows. Pseudo-label signal may be too weak.")

    # Build pseudo-labels
    pseudo_y = np.zeros(len(test), dtype=np.int32)
    pseudo_y[pseudo_pos_mask] = 1
    pseudo_y[pseudo_neg_mask] = 0
    pseudo_X = X_test.iloc[pseudo_mask].copy()

    # Augmented train: original train + pseudo-labeled subset of test
    X_aug = pd.concat([X, pseudo_X], axis=0, ignore_index=True)
    # Categorical dtype is lost on concat when train/test have different
    # category sets (Driver: 887 train vs 801 test). Re-cast.
    for c in cat_cols:
        X_aug[c] = X_aug[c].astype("category")
    y_aug = np.concatenate([y, pseudo_y[pseudo_mask]])
    print(f"Augmented train: {len(X_aug)} rows ({len(X)} original + {n_total} pseudo)")

    # 5-fold StratKFold on augmented train; OOF only over the ORIGINAL train
    # rows (pseudo rows aren't in test eval). Pinned seed=42.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))   # original-train-only splits

    n_train_orig = len(X)
    oof = np.zeros(n_train_orig, dtype=np.float32)
    test_proba = np.zeros(len(test), dtype=np.float32)
    fold_scores = []

    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        # Train on outer-train + ALL pseudo (pseudo rows are always in train,
        # never in val — they're test rows)
        pseudo_idx_in_aug = np.arange(n_train_orig, len(X_aug))
        train_idx_aug = np.concatenate([tr, pseudo_idx_in_aug])
        val_idx_aug = va  # original-only; never includes pseudo

        dtrain = lgb.Dataset(X_aug.iloc[train_idx_aug], y_aug[train_idx_aug],
                             categorical_feature=cat_cols)
        dval = lgb.Dataset(X_aug.iloc[val_idx_aug], y_aug[val_idx_aug],
                           categorical_feature=cat_cols)
        model = lgb.train(make_lgb_params(), dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100),
                                     lgb.log_evaluation(0)])
        p_va = model.predict(X_aug.iloc[val_idx_aug])
        oof[va] = p_va
        test_proba += model.predict(X_test) / N_FOLDS
        s = float(roc_auc_score(y[va], p_va))
        fold_scores.append(s)
        print(f"  fold {k}: AUC={s:.5f}  wall={time.time()-t0:.0f}s")

    auc_full = float(roc_auc_score(y, oof))
    delta_bp = (auc_full - BASE_S) * 1e4
    delta_baseline_lgbm = (auc_full - 0.94075) * 1e4
    print(f"\nPseudo-label H1 LGBM Strat OOF: {auc_full:.5f}  "
          f"std={np.std(fold_scores):.5f}  "
          f"Δ baseline={delta_bp:+.1f}bp")
    # Compare to baseline_two_anchor (no pseudo): 0.94075
    print(f"  Δ vs baseline_two_anchor (no pseudo): {delta_baseline_lgbm:+.1f}bp")

    save_oof("d3f_pseudo_lgbm_strat",
             np.column_stack([1 - oof, oof]),
             np.column_stack([1 - test_proba, test_proba]),
             dict(oof_score=auc_full, fold_std=float(np.std(fold_scores)),
                  fold_scores=fold_scores,
                  cv="StratifiedKFold(5)", metric="roc_auc",
                  delta_vs_baseline_bp=delta_bp,
                  n_pseudo_pos=n_pos, n_pseudo_neg=n_neg,
                  agree_hi=AGREE_HI, agree_lo=AGREE_LO,
                  pool_size=len(POOL),
                  notes="H1 pseudo-label with multi-base agreement guard"))

    Path("submissions").mkdir(exist_ok=True)
    sample_sub[TARGET] = test_proba
    sample_sub.to_csv("submissions/submission_d3f_pseudo_lgbm.csv", index=False)


if __name__ == "__main__":
    main()
