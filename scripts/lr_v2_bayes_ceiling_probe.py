"""scripts/lr_v2_bayes_ceiling_probe.py — Redo R12-1 with proper loss.

R12-1 (2026-05-19) probed Bayes ceiling by training CatBoost REGRESSOR
with RMSE loss on residual y - OOF_R7.1, got standalone AUC 0.478, and
concluded "R7.1 is at Bayes-optimal calibration ceiling". Per the
professor critique (knowledge-base): this is methodologically flawed.

The right probe: train CatBoost CLASSIFIER on y (binary) with
baseline = logit(p̂) as a FROZEN offset. The model's job is to learn
ANY deviation from p̂'s logit. If it can't, p̂ is truly at the Bayes
ceiling. If it can, more signal exists.

CatBoost supports this natively via the `baseline` parameter on Pool.
The model fits f(x) such that final_logit = baseline + f(x), and the
loss is binary cross-entropy on y vs sigmoid(final_logit). Standard
GBDT-with-init-score.

DECISION RULE:
- New OOF AUC > R25 OOF (0.95450) by ≥ +0.01 bp → Bayes ceiling
  BROKEN, more axes to explore
- New OOF AUC ≈ R25 OOF (within ±0.005 bp) → ceiling confirmed at
  R25 level
- New OOF AUC < R25 OOF → CB overfit beyond the baseline; ceiling
  confirmed (R25 already optimal for this feature class)
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

import sys
sys.path.insert(0, "scripts")
from p1_features import (
    TE_CONFIGS, apply_fs_a, feature_columns_for_lgbm,
    fit_fs_a, make_features_static,
)
from p1_single_cb import cb_params, fold_safe_te_for_fold

ART = Path("scripts/artifacts")
DATA = Path("data")
AUDIT = Path("audit")
TARGET, ID_COL = "PitNextLap", "id"
SEED, N_FOLDS = 42, 5

# R25 PRIMARY baseline — current PRIMARY at LB 0.95402.
# Using the K=18 PRIMARY OOF instead of R25's blended OOF because the
# blend is a rank-mean (uniform [0,1] values, not probabilities) which
# don't have a clean logit. K=18 IS R25's heavy component (×0.89).
R25_BASELINE_KEY = "K18"
R25_BASELINE_OOF = "oof_K18_pathb_driverclass_stint_tau100000.npy"
R25_BASELINE_TEST = "test_K18_pathb_driverclass_stint_tau100000.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Fold 0 only (~10 min wall).")
    ap.add_argument("--name", default="bayes_ceiling_probe")
    ap.add_argument("--max-iters", type=int, default=4000,
                    help="CB iterations (production cb_v4 uses 8000).")
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()
    t0_total = time.time()

    print(f"=== Bayes ceiling re-probe | baseline={R25_BASELINE_KEY} | "
          f"iters={args.max_iters} depth={args.depth} | "
          f"smoke={args.smoke} ===", flush=True)

    # 1) Load R25-equivalent baseline (K=18 PRIMARY OOF)
    base_oof = np.load(ART / R25_BASELINE_OOF).ravel()
    base_test = np.load(ART / R25_BASELINE_TEST).ravel()
    print(f"  baseline {R25_BASELINE_KEY}: OOF len={len(base_oof)}, "
          f"test len={len(base_test)}", flush=True)

    # 2) Load + featurize
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    # Map baseline arrays from original train.csv id-order to train_S
    # (sorted) row order. p1_single_cb.py:466 sort_back inverts this:
    # oof_aligned = oof[sort_back] where sort_back = argsort(train_S.id).
    # So train_S row i corresponds to original-id rank at order[i];
    # base_oof is in original-id order so base_oof_sorted = base_oof[order].
    order = train_S["id"].values
    base_oof_sorted = base_oof[order]
    # test: map original test.csv id-order to test_S sorted order
    order_te = test_S["id"].values
    orig_te = pd.read_csv(DATA / "test.csv", usecols=[ID_COL])[ID_COL].values
    id_to_orig_pos = {tid: i for i, tid in enumerate(orig_te)}
    base_test_sorted = np.array([base_test[id_to_orig_pos[t]] for t in order_te])

    # 3) Compute baseline logits (clipped to avoid inf)
    eps = 1e-7
    base_oof_clip = np.clip(base_oof_sorted, eps, 1 - eps)
    base_test_clip = np.clip(base_test_sorted, eps, 1 - eps)
    baseline_logit_oof = np.log(base_oof_clip / (1 - base_oof_clip))
    baseline_logit_test = np.log(base_test_clip / (1 - base_test_clip))
    print(f"  baseline logit_oof: mean={baseline_logit_oof.mean():.3f}  "
          f"std={baseline_logit_oof.std():.3f}", flush=True)

    # Sanity: baseline OOF AUC should match K=18 PRIMARY AUC 0.95450
    base_auc = float(roc_auc_score(y, base_oof_sorted))
    print(f"  baseline OOF AUC: {base_auc:.6f}", flush=True)

    # 4) Build CB features (same as p1_single_cb.py)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_fs_a = fit_fs_a(train_S.iloc[fold_list[0][0]])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    cb_feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in cb_feats and c not in cat_cols:
            cat_cols.append(c)
    cb_feats = cb_feats + [n for _, _, n in TE_CONFIGS]
    print(f"  CB feats={len(cb_feats)} cat={len(cat_cols)}", flush=True)

    n_train = len(y)
    n_test = len(test_S)
    oof = np.zeros(n_train, dtype=np.float64)
    test_pred = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []
    fold_walls = []
    n_eff_folds = 1 if args.smoke else N_FOLDS

    for fold, (ti, vi) in enumerate(fold_list[:n_eff_folds]):
        t0 = time.time()
        print(f"\n  --- Fold {fold+1}/{n_eff_folds} | "
              f"ti={len(ti)} va={len(vi)} ---", flush=True)

        # Per-fold FS_A + TE (same as cb_v4)
        fs_a = fit_fs_a(train_S.iloc[ti])
        train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
        train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
        test_fold = apply_fs_a(test_S, fs_a)
        y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
        fold_safe_te_for_fold(train_ti, train_va, test_fold, y_ti, fold, N_FOLDS)

        X_tr = train_ti.reindex(columns=cb_feats, fill_value=0).copy()
        X_va = train_va.reindex(columns=cb_feats, fill_value=0).copy()
        X_te = test_fold.reindex(columns=cb_feats, fill_value=0).copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("int32")
            X_va[c] = X_va[c].astype("int32")
            X_te[c] = X_te[c].astype("int32")
        num_cb = [c for c in cb_feats if c not in cat_cols]
        for X in [X_tr, X_va, X_te]:
            X[num_cb] = X[num_cb].fillna(0).astype(np.float32)
        cat_idx = [cb_feats.index(c) for c in cat_cols]

        # Baseline slices (in train_S row order)
        baseline_ti = baseline_logit_oof[ti]
        baseline_va = baseline_logit_oof[vi]
        baseline_te = baseline_logit_test

        # Construct Pools WITH baseline
        train_pool = cb.Pool(X_tr, train_ti[TARGET].astype(int).values,
                              cat_features=cat_idx, baseline=baseline_ti)
        val_pool = cb.Pool(X_va, train_va[TARGET].astype(int).values,
                            cat_features=cat_idx, baseline=baseline_va)
        test_pool = cb.Pool(X_te, cat_features=cat_idx, baseline=baseline_te)

        # Train with baseline (model learns f(x); final logit = baseline + f(x))
        t_cb = time.time()
        params = cb_params(use_gpu=False, max_iters=args.max_iters,
                           seed=SEED, depth=args.depth)
        m = cb.CatBoostClassifier(**params)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        n_trees = int(m.tree_count_)

        # predict_proba with baseline-aware Pool → final probabilities
        oof_va = m.predict_proba(val_pool)[:, 1]
        test_va = m.predict_proba(test_pool)[:, 1]

        # Standalone (FULL = baseline + model) AUC
        full_va_auc = float(roc_auc_score(
            train_va[TARGET].astype(int).values, oof_va))

        # Also compute baseline-only AUC on this fold for direct comparison
        base_only_va_auc = float(roc_auc_score(
            train_va[TARGET].astype(int).values,
            base_oof_sorted[vi]))

        print(f"    CB+baseline: {n_trees} trees | wall={time.time()-t_cb:.1f}s",
              flush=True)
        print(f"    FULL AUC_va         = {full_va_auc:.5f}", flush=True)
        print(f"    baseline-only AUC_va= {base_only_va_auc:.5f}", flush=True)
        print(f"    fold Δ              = {(full_va_auc-base_only_va_auc)*1e4:+.3f} bp",
              flush=True)

        oof[vi] = oof_va
        test_pred += test_va / n_eff_folds
        fold_aucs.append(full_va_auc)
        fold_walls.append(time.time() - t0)

        del m, X_tr, X_va, X_te, train_pool, val_pool, test_pool
        import gc; gc.collect()

    if args.smoke:
        print(f"\n  SMOKE fold-0 FULL AUC: {fold_aucs[0]:.5f}  "
              f"total wall={time.time()-t0_total:.1f}s", flush=True)
        return

    full_auc = float(roc_auc_score(y, oof))

    # Save aligned to original train.csv id-order
    sort_back = np.argsort(order)
    oof_aligned = oof[sort_back]
    test_aligned = np.zeros(n_test)
    for i, tid in enumerate(orig_te):
        # Find position in test_S sorted order
        # We saved test_pred indexed by test_S row order; need to map back.
        pass
    # Simpler: map test_pred from test_S sorted order back to orig
    sort_back_te = np.argsort(order_te)
    test_aligned = test_pred[sort_back_te]

    np.save(ART / f"oof_{args.name}_strat.npy",
            np.column_stack([1 - oof_aligned, oof_aligned]).astype(np.float64))
    np.save(ART / f"test_{args.name}_strat.npy",
            np.column_stack([1 - test_aligned, test_aligned]).astype(np.float64))

    print(f"\n  === Bayes ceiling probe results ===", flush=True)
    print(f"  K=18 PRIMARY baseline OOF AUC : {base_auc:.6f}", flush=True)
    print(f"  CB-with-baseline OOF AUC      : {full_auc:.6f}", flush=True)
    print(f"  Δ                              : {(full_auc - base_auc)*1e4:+.3f} bp",
          flush=True)
    print(f"  per-fold AUCs                  : {[f'{a:.5f}' for a in fold_aucs]}",
          flush=True)
    print(f"  fold-std                       : {np.std(fold_aucs):.5f}",
          flush=True)
    print(f"  total wall                     : {(time.time()-t0_total)/60:.1f} min",
          flush=True)

    # Verdict
    delta_bp = (full_auc - base_auc) * 1e4
    if delta_bp >= 0.01:
        verdict = "BAYES_CEILING_BROKEN"
        why = (f"CB+baseline OOF +{delta_bp:.3f} bp over K=18 → extractable "
               "signal beyond R25/K=18 exists; reopens mechanism axes")
    elif abs(delta_bp) < 0.005:
        verdict = "BAYES_CEILING_CONFIRMED_AT_R25"
        why = (f"Δ {delta_bp:+.3f} bp within noise — confirms R25 PRIMARY "
               "is at Bayes ceiling for the cb_v4 feature class")
    else:
        verdict = "CB_OVERFITS_BEYOND_BASELINE"
        why = (f"Δ {delta_bp:+.3f} bp negative — CB+baseline overfits; ceiling "
               "actually below R25's OOF (R25's stacking captures more)")
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  WHY    : {why}", flush=True)

    AUDIT.mkdir(exist_ok=True)
    Path(ART / f"{args.name}_results.json").write_text(json.dumps(dict(
        baseline_key=R25_BASELINE_KEY,
        baseline_oof_auc=base_auc,
        full_oof_auc=full_auc,
        delta_bp=delta_bp,
        verdict=verdict, why=why,
        fold_aucs=fold_aucs, fold_walls=fold_walls,
        fold_std=float(np.std(fold_aucs)),
        max_iters=args.max_iters, depth=args.depth,
    ), indent=2))


if __name__ == "__main__":
    main()
