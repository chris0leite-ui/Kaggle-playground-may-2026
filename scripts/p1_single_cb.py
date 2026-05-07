"""P1 single-CatBoost probe — Rozen single-CB OOF 0.95127 reference.

Same FE + CV TE as p1_single_lgbm.py but with CatBoost native cat handling.
Uses our 5-fold StratifiedKFold(seed=42).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from p1_features import TE_CONFIGS, cv_target_encode, feature_columns_for_lgbm, make_features_A

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# Rozen CB params (CPU; GPU disabled for our sandbox)
ROZEN_CB = dict(
    iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
    bootstrap_type="Bernoulli", subsample=0.8, eval_metric="AUC",
    task_type="CPU",
    random_seed=SEED, verbose=200, early_stopping_rounds=100,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="feA_te",
                    choices=["feA", "feA_te"])
    ap.add_argument("--max_rounds", type=int, default=5000)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"p1_single_cb_{args.variant}"
    print(f"=== P1 single-CB | variant={args.variant} | name={name} ===")
    t0_total = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")
    print(f"  train {train.shape}  test {test.shape}")

    train_A, state = make_features_A(train, fit=True)
    test_A, _ = make_features_A(test, fit=False, state=state)
    y = train_A[TARGET].astype(int).reset_index(drop=True)
    feats, cat_cols = feature_columns_for_lgbm(train_A)
    print(f"  feats {len(feats)}  cat {len(cat_cols)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    if args.variant == "feA_te":
        for cols, smooth, te_name in TE_CONFIGS:
            if all(c in train_A.columns for c in cols):
                oof_enc, te_enc = cv_target_encode(
                    train_A, test_A, cols, y, fold_list, smoothing=smooth)
                train_A[te_name] = oof_enc
                test_A[te_name] = te_enc
                feats.append(te_name)
        print(f"  + 6 TE feats; total feats {len(feats)}")

    X = train_A[feats].copy()
    X_test = test_A[feats].copy()
    for c in cat_cols:
        X[c] = X[c].astype("int32")
        X_test[c] = X_test[c].astype("int32")
    num_cols = [c for c in feats if c not in cat_cols]
    X[num_cols] = X[num_cols].fillna(0).astype(np.float32)
    X_test[num_cols] = X_test[num_cols].fillna(0).astype(np.float32)

    cat_idx = [feats.index(c) for c in cat_cols]
    params = dict(ROZEN_CB)
    params["iterations"] = args.max_rounds
    print(f"  hparams: depth={params['depth']} iters={params['iterations']} cat_idx={cat_idx}")

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs, walls = [], []
    for fold, (ti, vi) in enumerate(fold_list, 1):
        t0 = time.time()
        Xtr, ytr = X.iloc[ti], y.iloc[ti]
        Xva, yva = X.iloc[vi], y.iloc[vi]
        m = cb.CatBoostClassifier(**params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=cat_idx)
        oof[vi] = m.predict_proba(Xva)[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        fold_aucs.append(float(roc_auc_score(yva, oof[vi])))
        walls.append(time.time() - t0)
        print(f"  Fold {fold}: AUC={fold_aucs[-1]:.5f}  iters={int(m.tree_count_)}  wall={walls[-1]:.1f}s")

    auc_full = float(roc_auc_score(y, oof))
    print(f"\n  OOF AUC: {auc_full:.5f}  total wall {time.time()-t0_total:.1f}s")

    # Map back to original train order
    order = train_A["id"].values
    sort_back = np.argsort(order)
    oof_aligned = oof[sort_back]
    order_te = test_A["id"].values
    id_to_pos = {tid: i for i, tid in enumerate(order_te)}
    orig_te = pd.read_csv("data/test.csv", usecols=[ID_COL])[ID_COL].values
    test_aligned = np.array([test_pred[id_to_pos[t]] for t in orig_te])

    np.save(ART / f"oof_{name}_strat.npy",
            np.column_stack([1 - oof_aligned, oof_aligned]).astype(np.float64))
    np.save(ART / f"test_{name}_strat.npy",
            np.column_stack([1 - test_aligned, test_aligned]).astype(np.float64))
    sub_out = sub[[ID_COL]].copy()
    sub_out[TARGET] = np.clip(test_aligned, 0.001, 0.999)
    Path("submissions").mkdir(exist_ok=True)
    sub_out.to_csv(f"submissions/submission_{name}.csv", index=False)
    (ART / f"{name}_results.json").write_text(json.dumps(dict(
        name=name, variant=args.variant,
        oof_auc_full=auc_full, fold_aucs=fold_aucs, fold_walls=walls,
        n_feats=len(feats), n_cat=len(cat_cols), params=params,
    ), indent=2, default=str))
    print(f"  → oof_{name}_strat.npy, test_..., submission_...")


if __name__ == "__main__":
    main()
