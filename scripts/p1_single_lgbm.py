"""P1 single-LGBM probe — replicate Rozen 0.95354 single-LGBM (~0.952 OOF).

Uses the project StratifiedKFold(seed=42) convention for OOF artifact
compatibility. Saves:
  scripts/artifacts/oof_p1_single_lgbm_strat.npy   (n_train, 2)
  scripts/artifacts/test_p1_single_lgbm_strat.npy  (n_test, 2)
  scripts/artifacts/p1_single_lgbm_results.json    metadata + AUC

Variants (CLI):
  --variant raw_only      : 14 raw features, Rozen LGBM hparams
  --variant feA           : ~118 engineered features (no TE)
  --variant feA_te        : ~118 engineered features + 6 CV TE features (default)
  --variant feA_te_orig   : feA_te + concat aadigupta_orig training rows

  --baseline-hparams      : use our project LGBM hparams (lr=0.05, leaves=63)
                            instead of Rozen (lr=0.025, leaves=255). Lets us
                            isolate FE-vs-hparams contribution.

By default runs `feA_te` with Rozen hparams.
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
    COMBO_COLS, TE_CONFIGS, cv_target_encode, feature_columns_for_lgbm,
    make_features_A,
)

ART = Path("scripts/artifacts")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

ROZEN_LGB = dict(
    objective="binary", metric="auc",
    n_estimators=6000, learning_rate=0.025,
    num_leaves=255, min_child_samples=25, min_data_in_bin=10,
    feature_fraction=0.65, bagging_fraction=0.85, bagging_freq=1,
    lambda_l1=1.2, lambda_l2=2.5, max_depth=10, path_smooth=0.1,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
PROJECT_LGB = dict(
    objective="binary", metric="auc",
    n_estimators=2000, learning_rate=0.05,
    num_leaves=63, min_child_samples=200,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    random_state=SEED, n_jobs=-1, verbose=-1,
)


def load_orig_aligned():
    """Load aadigupta original; drop Normalized_TyreLife (forbidden)."""
    p = Path("external/aadigupta_orig/f1_strategy_dataset_v4.csv")
    if not p.exists():
        return None
    o = pd.read_csv(p)
    if "Normalized_TyreLife" in o.columns:
        o = o.drop(columns=["Normalized_TyreLife"])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["raw_only", "feA", "feA_te", "feA_te_orig"],
                    default="feA_te")
    ap.add_argument("--baseline-hparams", action="store_true",
                    help="Use project hparams instead of Rozen's.")
    ap.add_argument("--name", default=None,
                    help="Override OOF artifact name (default: p1_single_lgbm_<variant>).")
    ap.add_argument("--max_rounds", type=int, default=6000)
    args = ap.parse_args()

    name = args.name or f"p1_single_lgbm_{args.variant}"
    print(f"=== P1 single-LGBM | variant={args.variant} | name={name} ===")
    t0_total = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")
    y = train[TARGET].astype(int).reset_index(drop=True)
    print(f"  train {train.shape}  test {test.shape}  pos_rate={y.mean():.4f}")

    # Build features per variant
    if args.variant == "raw_only":
        # Rozen-like, but minimal: just the 14 raw cols + cat codes
        train_A = train.copy()
        test_A = test.copy()
        for c in ["Driver", "Race", "Compound"]:
            uniques = pd.concat([train[c], test[c]]).astype(str).unique()
            mp = {v: i for i, v in enumerate(sorted(uniques))}
            train_A[f"{c}_cat"] = train[c].astype(str).map(mp).astype("int32")
            test_A[f"{c}_cat"] = test[c].astype(str).map(mp).astype("int32")
        feats = [c for c in train_A.columns
                 if c not in {ID_COL, TARGET, "Driver", "Race", "Compound"}]
        cat_cols = [c for c in feats if c.endswith("_cat")]
    else:
        print("  building engineered features A...")
        train_A, state = make_features_A(train, fit=True)
        test_A, _ = make_features_A(test, fit=False, state=state)
        # CRITICAL: the make_features_A sorts by (Driver, Race, Year, LapNumber)
        # to produce the lag/rolling. We must rebuild y aligned to that order.
        y = train_A[TARGET].astype(int).reset_index(drop=True)
        feats, cat_cols = feature_columns_for_lgbm(train_A)

    # Compute fold split AFTER reordering
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))

    # Add CV TE features (or skip)
    if args.variant in ("feA_te", "feA_te_orig"):
        print("  adding CV target encodings...")
        for cols, smooth, te_name in TE_CONFIGS:
            if all(c in train_A.columns for c in cols):
                oof_enc, te_enc = cv_target_encode(
                    train_A, test_A, cols, y, fold_list, smoothing=smooth)
                train_A[te_name] = oof_enc
                test_A[te_name] = te_enc
                feats.append(te_name)
                print(f"    {te_name}: train mean={oof_enc.mean():.4f} std={oof_enc.std():.4f}")

    # Optionally load + concat orig
    X_orig = y_orig = None
    if args.variant == "feA_te_orig":
        orig = load_orig_aligned()
        if orig is not None:
            print(f"  loaded orig {orig.shape}; building features...")
            orig_A, _ = make_features_A(orig, fit=False, state=state)
            # TE features on orig: use full-train stats_full (proxied by global mean for missing keys).
            # Vectorised string concat (avoids agg axis=1 NaN crash).
            def _key(df, cs):
                s = df[cs[0]].fillna("MISSING").astype(str)
                for c in cs[1:]:
                    s = s + "__" + df[c].fillna("MISSING").astype(str)
                return s
            for cols, smooth, te_name in TE_CONFIGS:
                if all(c in orig_A.columns for c in cols):
                    key_full = _key(train_A, cols)
                    key_orig = _key(orig_A, cols)
                    target_arr = y.reset_index(drop=True)
                    stats_full = (pd.DataFrame({"key": key_full.values,
                                                "target": target_arr.values})
                                  .groupby("key")["target"].agg(["sum", "count"]))
                    gm = float(target_arr.mean())
                    stats_full["enc"] = ((stats_full["sum"] + smooth * gm)
                                         / (stats_full["count"] + smooth))
                    orig_A[te_name] = key_orig.map(stats_full["enc"].to_dict()).fillna(gm).values
            X_orig = orig_A[feats].astype(np.float32)
            y_orig = orig_A[TARGET].astype(int).reset_index(drop=True)
            print(f"  orig X shape {X_orig.shape}")

    print(f"  feats: {len(feats)}  cat_cols: {len(cat_cols)}")
    X = train_A[feats]
    X_test = test_A[feats]

    # Cast cat_cols to int (LGBM expects integer for `categorical_feature`)
    for c in cat_cols:
        X = X.copy()
        X_test = X_test.copy()
        X[c] = X[c].astype("int32")
        X_test[c] = X_test[c].astype("int32")
        if X_orig is not None:
            X_orig = X_orig.copy()
            X_orig[c] = X_orig[c].astype("int32")
    # Numeric features fillna with 0
    num_cols = [c for c in feats if c not in cat_cols]
    X[num_cols] = X[num_cols].fillna(0).astype(np.float32)
    X_test[num_cols] = X_test[num_cols].fillna(0).astype(np.float32)
    if X_orig is not None:
        X_orig[num_cols] = X_orig[num_cols].fillna(0).astype(np.float32)

    params = dict(PROJECT_LGB if args.baseline_hparams else ROZEN_LGB)
    if args.variant == "raw_only":
        params["n_estimators"] = min(2000, args.max_rounds)
    else:
        params["n_estimators"] = args.max_rounds

    print(f"  hparams: {'PROJECT' if args.baseline_hparams else 'ROZEN'}  "
          f"lr={params['learning_rate']} leaves={params['num_leaves']}  "
          f"n_estimators={params['n_estimators']}")

    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs, walls, iters = [], [], []

    for fold, (ti, vi) in enumerate(fold_list, 1):
        t0 = time.time()
        Xtr, ytr = X.iloc[ti], y.iloc[ti]
        Xva, yva = X.iloc[vi], y.iloc[vi]
        if X_orig is not None:
            Xtr = pd.concat([Xtr, X_orig], ignore_index=True)
            ytr = pd.concat([ytr, y_orig], ignore_index=True)
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr,
              eval_set=[(Xva, yva)],
              categorical_feature=cat_cols,
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
        oof[vi] = m.predict_proba(Xva)[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        fold_aucs.append(float(roc_auc_score(yva, oof[vi])))
        walls.append(time.time() - t0)
        iters.append(int(m.best_iteration_ or params["n_estimators"]))
        print(f"  Fold {fold}: AUC={fold_aucs[-1]:.5f}  "
              f"iters={iters[-1]}  wall={walls[-1]:.1f}s")

    auc_full = float(roc_auc_score(y, oof))
    auc_std = float(np.std(fold_aucs))
    print(f"\n  OOF AUC (full): {auc_full:.5f}   fold-std={auc_std:.5f}   "
          f"total wall={time.time()-t0_total:.1f}s")

    # Save in our convention: 2-col [P0, P1], aligned to ORIGINAL train order.
    # train_A was sorted; we need to map back.
    if args.variant == "raw_only":
        oof_aligned = oof
        test_aligned = test_pred
    else:
        # train_A retains "id" — use it to map back
        order = train_A["id"].values
        # train.csv id is 0..n_train-1
        sort_back = np.argsort(order)
        oof_aligned = oof[sort_back]
        # for test: same idea
        order_te = test_A["id"].values
        # test.csv id starts at n_train (e.g., 439140)
        # map: test_pred[i] for test_A.iloc[i] -> position in original test
        id_to_pos = {tid: i for i, tid in enumerate(order_te)}
        # original test order
        orig_te = pd.read_csv("data/test.csv", usecols=[ID_COL])[ID_COL].values
        test_aligned = np.array([test_pred[id_to_pos[t]] for t in orig_te])

    oof2 = np.column_stack([1 - oof_aligned, oof_aligned]).astype(np.float64)
    test2 = np.column_stack([1 - test_aligned, test_aligned]).astype(np.float64)
    np.save(ART / f"oof_{name}_strat.npy", oof2)
    np.save(ART / f"test_{name}_strat.npy", test2)

    sub_out = sub[[ID_COL]].copy()
    sub_out[TARGET] = np.clip(test_aligned, 0.001, 0.999)
    sub_path = Path(f"submissions/submission_{name}.csv")
    sub_path.parent.mkdir(exist_ok=True)
    sub_out.to_csv(sub_path, index=False)

    results = dict(
        name=name, variant=args.variant,
        baseline_hparams=args.baseline_hparams,
        feats=feats, n_feats=len(feats), n_cat=len(cat_cols),
        oof_auc_full=auc_full,
        fold_aucs=fold_aucs, fold_std=auc_std,
        fold_iters=iters, fold_walls=walls,
        params=params,
    )
    (ART / f"{name}_results.json").write_text(json.dumps(results, indent=2,
                                                         default=str))
    print(f"\n  → oof_{name}_strat.npy   test_{name}_strat.npy   submission_{name}.csv")
    print(f"  → {ART / (name + '_results.json')}")


if __name__ == "__main__":
    main()
