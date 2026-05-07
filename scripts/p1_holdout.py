"""P1 honest 80/20 holdout test — simulate LB without burning a slot.

The 5-fold OOF AUC tells us how well the model predicts on val rows
inside CV. The holdout AUC tells us how well the model predicts on
truly held-out rows that the FE state never saw. If
  holdout_AUC ≈ OOF_AUC  → FE is clean, OOF→LB transfer should hold.
  holdout_AUC ≪ OOF_AUC  → FE has residual cross-fold or distribution-
                            shift leakage; OOF inflated; LB will regress.

Workflow:
1. StratifiedKFold(seed=99, 5-split) → take fold 0 as 20% holdout.
   (Different seed from main 5-fold so the split is independent.)
2. Build FE state on the 80% TRAIN ONLY (no 20% labels seen).
3. Apply FE to the 20% holdout via state lookup (test-style).
4. Run inner 5-fold CV TE on the 80% to produce TE features.
5. Train LGBM on the 80% with full FE; predict 20%; report AUC.

Saves p1_holdout_results.json with the AUC + per-feature dropdown
hooks (e.g., AUC without CV TE).

Usage:
  python3 scripts/p1_holdout.py             # full FE
  python3 scripts/p1_holdout.py --no-te     # FE without CV TE
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from p1_features import (
    TE_CONFIGS, cv_target_encode, feature_columns_for_lgbm, make_features_A,
)

ART = Path("scripts/artifacts")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5
HOLDOUT_SEED = 99   # different from main pipeline so split is independent

ROZEN_LGB = dict(
    objective="binary", metric="auc",
    n_estimators=6000, learning_rate=0.025,
    num_leaves=255, min_child_samples=25, min_data_in_bin=10,
    feature_fraction=0.65, bagging_fraction=0.85, bagging_freq=1,
    lambda_l1=1.2, lambda_l2=2.5, max_depth=10, path_smooth=0.1,
    random_state=SEED, n_jobs=-1, verbose=-1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-te", action="store_true",
                    help="Drop CV TE features (ablation)")
    ap.add_argument("--max-rounds", type=int, default=6000)
    args = ap.parse_args()
    print(f"=== P1 holdout test | no_te={args.no_te} ===")
    t0 = time.time()

    train = pd.read_csv("data/train.csv")
    y_full = train[TARGET].astype(int).values

    # Step 1: 80/20 stratified split (independent seed)
    skf_holdout = StratifiedKFold(5, shuffle=True, random_state=HOLDOUT_SEED)
    train_idx, holdout_idx = next(skf_holdout.split(np.zeros(len(y_full)), y_full))
    print(f"  train idx: {len(train_idx)}  holdout idx: {len(holdout_idx)}")

    df_train = train.iloc[train_idx].reset_index(drop=True)
    df_holdout = train.iloc[holdout_idx].reset_index(drop=True)

    # Step 2: build FE state on TRAIN ONLY
    print(f"  building FE on 80% train...")
    train_A, state = make_features_A(df_train, fit=True)
    print(f"  applying FE to 20% holdout...")
    holdout_A, _ = make_features_A(df_holdout, fit=False, state=state)
    y = train_A[TARGET].astype(int).reset_index(drop=True)
    y_h = holdout_A[TARGET].astype(int).reset_index(drop=True)

    feats, cat_cols = feature_columns_for_lgbm(train_A)

    # Step 3: CV TE on the 80% train only (inner 5-fold)
    if not args.no_te:
        print(f"  inner CV TE...")
        skf_inner = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        fold_list = list(skf_inner.split(np.zeros(len(y)), y))
        for cols, smooth, te_name in TE_CONFIGS:
            if all(c in train_A.columns for c in cols):
                oof_enc, ho_enc = cv_target_encode(
                    train_A, holdout_A, cols, y, fold_list, smoothing=smooth)
                train_A[te_name] = oof_enc
                holdout_A[te_name] = ho_enc
                feats.append(te_name)
        print(f"  feats: {len(feats)} (with TE)")
    else:
        print(f"  feats: {len(feats)} (NO TE)")

    X = train_A[feats].copy()
    X_h = holdout_A[feats].copy()
    for c in cat_cols:
        X[c] = X[c].astype("int32")
        X_h[c] = X_h[c].astype("int32")
    num_cols = [c for c in feats if c not in cat_cols]
    X[num_cols] = X[num_cols].fillna(0).astype(np.float32)
    X_h[num_cols] = X_h[num_cols].fillna(0).astype(np.float32)

    # Step 4: train LGBM on 80%, eval on 20% holdout
    print(f"  training LGBM (Rozen hparams)...")
    params = dict(ROZEN_LGB)
    params["n_estimators"] = args.max_rounds
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y, eval_set=[(X_h, y_h)],
          categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(150, verbose=False),
                     lgb.log_evaluation(200)])
    pred_h = m.predict_proba(X_h)[:, 1]
    holdout_auc = float(roc_auc_score(y_h, pred_h))

    # Also compute train-set self-AUC (for sanity)
    pred_tr = m.predict_proba(X)[:, 1]
    train_auc = float(roc_auc_score(y, pred_tr))

    print(f"\n  Train self-AUC: {train_auc:.5f}")
    print(f"  HOLDOUT AUC:    {holdout_auc:.5f}")
    print(f"  vs published OOF (v2): 0.95128")
    print(f"  vs PRIMARY OOF: 0.95090, LB 0.95059")
    print(f"  best iteration: {m.best_iteration_}")
    print(f"  total wall: {time.time()-t0:.1f}s")

    suffix = "_noTE" if args.no_te else ""
    out = ART / f"p1_holdout_results{suffix}.json"
    out.write_text(json.dumps(dict(
        no_te=args.no_te, n_train=len(train_idx), n_holdout=len(holdout_idx),
        n_feats=len(feats), train_auc=train_auc, holdout_auc=holdout_auc,
        best_iter=int(m.best_iteration_ or params["n_estimators"]),
        wall=time.time()-t0,
    ), indent=2))
    print(f"  → {out}")


if __name__ == "__main__":
    main()
