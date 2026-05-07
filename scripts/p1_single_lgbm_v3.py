"""P1 v3 single-LGBM — fold-safe FS_A.

Strict rule (PI 2026-05-07): we only use data we have available at
training time. No out-of-fold labels in feature construction.

What's fold-safe:
- make_features_static: label-independent (tyre algebra, lag/rolling,
  hist priors, factorize maps, combo cats). Computed once on train and
  test. ✓
- CV TE (cv_target_encode): per-fold ti-only stats. Already fold-safe. ✓
- FS_A (label-conditional aggregates): NOW computed inside each CV
  fold using ti rows only; for the final test prediction, each fold's
  LGBM uses its own ti-fitted FS_A applied to test. Test predictions
  are 5-fold averaged. ✓ (FIXED in v3)

Saves:
  scripts/artifacts/oof_<name>_strat.npy   (n_train, 2)
  scripts/artifacts/test_<name>_strat.npy  (n_test, 2)
  scripts/artifacts/<name>_results.json
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
    TE_CONFIGS, apply_fs_a, cv_target_encode, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="p1_single_lgbm_v3_feA_te")
    ap.add_argument("--max-rounds", type=int, default=6000)
    ap.add_argument("--no-te", action="store_true",
                    help="Drop CV TE features for ablation")
    args = ap.parse_args()

    print(f"=== P1 v3 fold-safe FS_A | name={args.name} | no_te={args.no_te} ===")
    t0_total = time.time()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")
    print(f"  train {train.shape}  test {test.shape}")

    # --- Static features (label-independent), computed once each ---
    print("  building static features (train + test)...")
    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    # Determine the canonical feature list using a sample fold-1 application
    # so feats / cat_cols are consistent across the 5 folds.
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_ti = fold_list[0][0]
    sample_fs_a = fit_fs_a(train_S.iloc[sample_ti])
    train_sample = apply_fs_a(train_S, sample_fs_a)
    feats, cat_cols = feature_columns_for_lgbm(train_sample)
    if not args.no_te:
        feats = feats + [n for _, _, n in TE_CONFIGS]
    print(f"  feats: {len(feats)}  cat: {len(cat_cols)}")

    # --- 5-fold loop with per-fold FS_A and per-fold CV TE ---
    n_train, n_test = len(y), len(test_S)
    oof = np.zeros(n_train, dtype=np.float64)
    test_pred = np.zeros(n_test, dtype=np.float64)
    fold_aucs, walls, iters = [], [], []

    for fold, (ti, vi) in enumerate(fold_list, 1):
        t0 = time.time()
        print(f"\n  --- Fold {fold} | ti={len(ti)} va={len(vi)} ---")

        # 1. fit FS_A on ti rows ONLY
        fs_a = fit_fs_a(train_S.iloc[ti])

        # 2. apply FS_A to ti, va, and test
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)

        # 3. add CV TE features (ti-only stats, applied to ti/va/test)
        if not args.no_te:
            for cols, smooth, te_name in TE_CONFIGS:
                if all(c in train_ti.columns for c in cols):
                    # inner CV TE on ti rows only — fits stats per inner fold
                    inner_skf = StratifiedKFold(N_FOLDS, shuffle=True,
                                                random_state=SEED + fold)
                    y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
                    inner_folds = list(inner_skf.split(np.zeros(len(y_ti)), y_ti))
                    ti_enc, _ = cv_target_encode(
                        train_ti, train_va, cols, y_ti, inner_folds,
                        smoothing=smooth)
                    train_ti[te_name] = ti_enc
                    # for va and test: use full ti stats (no leakage; va labels never seen)
                    global_mean = float(y_ti.mean())
                    key_ti = train_ti[cols].astype(str).agg("__".join, axis=1) \
                        if False else None
                    # vectorised
                    def _key(df):
                        s = df[cols[0]].fillna("MISSING").astype(str)
                        for c in cols[1:]:
                            s = s + "__" + df[c].fillna("MISSING").astype(str)
                        return s.reset_index(drop=True)
                    k_ti = _key(train_ti)
                    stats_full = (pd.DataFrame({"key": k_ti.values,
                                                "target": y_ti.values})
                                  .groupby("key")["target"].agg(["sum", "count"]))
                    stats_full["enc"] = ((stats_full["sum"] + smooth * global_mean)
                                         / (stats_full["count"] + smooth))
                    enc_map = stats_full["enc"].to_dict()
                    train_va[te_name] = _key(train_va).map(enc_map).fillna(global_mean).values
                    test_fold[te_name] = _key(test_fold).map(enc_map).fillna(global_mean).values

        # 4. assemble feature matrices
        X_tr = train_ti.reindex(columns=feats, fill_value=0).copy()
        X_va = train_va.reindex(columns=feats, fill_value=0).copy()
        X_te = test_fold.reindex(columns=feats, fill_value=0).copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("int32")
            X_va[c] = X_va[c].astype("int32")
            X_te[c] = X_te[c].astype("int32")
        num_cols = [c for c in feats if c not in cat_cols]
        for X in [X_tr, X_va, X_te]:
            X[num_cols] = X[num_cols].fillna(0).astype(np.float32)

        # 5. train LGBM
        params = dict(ROZEN_LGB)
        params["n_estimators"] = args.max_rounds
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, train_ti[TARGET].astype(int),
              eval_set=[(X_va, train_va[TARGET].astype(int))],
              categorical_feature=cat_cols,
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

        # 6. record OOF + test prediction (sorted-train-A index aligned)
        sorted_ti = train_S.iloc[ti].index.values  # original positions in sorted train_S
        sorted_vi = train_S.iloc[vi].index.values
        oof_va = m.predict_proba(X_va)[:, 1]
        oof[sorted_vi] = oof_va
        test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS

        fold_aucs.append(float(roc_auc_score(
            train_va[TARGET].astype(int).values, oof_va)))
        walls.append(time.time() - t0)
        iters.append(int(m.best_iteration_ or params["n_estimators"]))
        print(f"    Fold {fold}: AUC={fold_aucs[-1]:.5f}  iters={iters[-1]}  wall={walls[-1]:.1f}s")

    auc_full = float(roc_auc_score(y, oof))
    print(f"\n  OOF AUC (full): {auc_full:.5f}  fold-std={np.std(fold_aucs):.5f}  "
          f"total wall={time.time()-t0_total:.1f}s")

    # Map back to original train.csv id order (train_S is sorted)
    order = train_S["id"].values
    sort_back = np.argsort(order)
    oof_aligned = oof[sort_back]
    order_te = test_S["id"].values
    id_to_pos = {tid: i for i, tid in enumerate(order_te)}
    orig_te = pd.read_csv("data/test.csv", usecols=[ID_COL])[ID_COL].values
    test_aligned = np.array([test_pred[id_to_pos[t]] for t in orig_te])

    np.save(ART / f"oof_{args.name}_strat.npy",
            np.column_stack([1 - oof_aligned, oof_aligned]).astype(np.float64))
    np.save(ART / f"test_{args.name}_strat.npy",
            np.column_stack([1 - test_aligned, test_aligned]).astype(np.float64))
    sub_out = sub[[ID_COL]].copy()
    sub_out[TARGET] = np.clip(test_aligned, 0.001, 0.999)
    Path("submissions").mkdir(exist_ok=True)
    sub_out.to_csv(f"submissions/submission_{args.name}.csv", index=False)
    (ART / f"{args.name}_results.json").write_text(json.dumps(dict(
        name=args.name, no_te=args.no_te,
        oof_auc_full=auc_full, fold_aucs=fold_aucs,
        fold_iters=iters, fold_walls=walls,
        n_feats=len(feats), n_cat=len(cat_cols),
    ), indent=2, default=str))
    print(f"  → oof_{args.name}_strat.npy   test_..._strat.npy   submission_..._strat.csv")


if __name__ == "__main__":
    main()
