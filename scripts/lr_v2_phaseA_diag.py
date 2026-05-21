"""scripts/lr_v2_phaseA_diag.py — diagnose LR convergence on fold 0.

Phase A full run got K=19 Δ = -0.049 bp (kill verdict). The LR hit
max_iter=500 every fold (lbfgs convergence warning). This script
re-fits CB on fold 0, re-extracts leaves, then tries 4 LR variants:

  V1: lbfgs C=10 max_iter=2000      (baseline + more iters)
  V2: saga  C=10 max_iter=500       (purpose-built for sparse)
  V3: saga  C=10 max_iter=500 + leaves-only  (drops scale-mismatched dense block)
  V4: liblinear C=10 dual=False     (coord-descent reference)

If any V gets AUC > 0.93, the kill verdict was a false-negative from
solver issues. If all stay ≤ 0.90, the kill is real.

Wall: CB fit (~3 min) + 4 LR fits (~5-10 min each) = ~25-45 min.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "scripts")
from p1_features import (
    TE_CONFIGS, apply_fs_a, feature_columns_for_lgbm, fit_fs_a,
    make_features_static,
)
from p1_single_cb import cb_params, fold_safe_te_for_fold
from lr_v2_phaseA_leaf_probe import (
    LR_NUM_COLS, LEAF_MIN_FREQ, PROBE_ITERS, PROBE_DEPTH,
    build_leaf_lookup, leaves_to_csr, assemble_X,
)

ART = Path("scripts/artifacts")
DATA = Path("data")
AUDIT = Path("audit")
TARGET = "PitNextLap"
SEED, N_FOLDS = 42, 5


def main():
    t0_total = time.time()
    print("=== Phase A LR diag (fold 0 only) ===", flush=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train_S, state = make_features_static(train, fit=True)
    test_S, _ = make_features_static(test, fit=False, state=state)
    y = train_S[TARGET].astype(int).reset_index(drop=True)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_list = list(skf.split(np.zeros(len(y)), y))
    sample_fs_a = fit_fs_a(train_S.iloc[fold_list[0][0]])
    sample_train = apply_fs_a(train_S, sample_fs_a)
    cb_feats, cat_cols = feature_columns_for_lgbm(sample_train)
    for c in ("Year", "Stint"):
        if c in cb_feats and c not in cat_cols:
            cat_cols.append(c)
    cb_feats = cb_feats + [n for _, _, n in TE_CONFIGS]

    ti, vi = fold_list[0]
    print(f"  Fold 0 ti={len(ti)} va={len(vi)}", flush=True)

    # Re-fit CB on fold 0 ti
    fs_a = fit_fs_a(train_S.iloc[ti])
    train_ti = apply_fs_a(train_S.iloc[ti].reset_index(drop=True), fs_a)
    train_va = apply_fs_a(train_S.iloc[vi].reset_index(drop=True), fs_a)
    test_fold = apply_fs_a(test_S, fs_a)
    y_ti = train_ti[TARGET].astype(int).reset_index(drop=True)
    fold_safe_te_for_fold(train_ti, train_va, test_fold, y_ti, 0, N_FOLDS)
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

    t_cb = time.time()
    params = cb_params(use_gpu=False, max_iters=PROBE_ITERS,
                       seed=SEED, depth=PROBE_DEPTH)
    m = cb.CatBoostClassifier(**params)
    m.fit(X_tr, train_ti[TARGET].astype(int).values,
          eval_set=(X_va, train_va[TARGET].astype(int).values),
          cat_features=cat_idx, use_best_model=True)
    cb_auc = float(roc_auc_score(train_va[TARGET].astype(int).values,
                                  m.predict_proba(X_va)[:, 1]))
    print(f"  CB fit: AUC_va={cb_auc:.5f} wall={time.time()-t_cb:.1f}s",
          flush=True)

    # Extract leaves
    leaves_ti = m.calc_leaf_indexes(X_tr)
    leaves_va = m.calc_leaf_indexes(X_va)
    leaves_te = m.calc_leaf_indexes(X_te)
    max_leaf_id = int(max(leaves_ti.max(), leaves_va.max(), leaves_te.max()))
    del m, X_tr, X_va, X_te
    import gc; gc.collect()

    tree_lookup, offsets, _ = build_leaf_lookup(
        leaves_ti, max_leaf_id=max_leaf_id, min_freq=LEAF_MIN_FREQ)
    print(f"  leaf cols: {int(offsets[-1])}", flush=True)
    Xtr_leaf = leaves_to_csr(leaves_ti, tree_lookup, offsets)
    Xva_leaf = leaves_to_csr(leaves_va, tree_lookup, offsets)
    Xte_leaf = leaves_to_csr(leaves_te, tree_lookup, offsets)
    del leaves_ti, leaves_va, leaves_te; gc.collect()

    # Dense block: numerics + TEs (same as main script)
    num_ti = train_ti[LR_NUM_COLS].fillna(0).astype(np.float32).values
    num_va = train_va[LR_NUM_COLS].fillna(0).astype(np.float32).values
    num_te = test_fold[LR_NUM_COLS].fillna(0).astype(np.float32).values
    scl = StandardScaler().fit(num_ti)
    num_ti = scl.transform(num_ti).astype(np.float32, copy=False)
    num_va = scl.transform(num_va).astype(np.float32, copy=False)
    num_te = scl.transform(num_te).astype(np.float32, copy=False)
    te_cols = [n for _, _, n in TE_CONFIGS]
    te_ti = train_ti[te_cols].fillna(0).astype(np.float32).values
    te_va = train_va[te_cols].fillna(0).astype(np.float32).values
    te_te = test_fold[te_cols].fillna(0).astype(np.float32).values

    y_tr_arr = train_ti[TARGET].astype(int).values
    y_va_arr = train_va[TARGET].astype(int).values
    del train_ti, train_va, test_fold; gc.collect()

    # Build matrices for V1/V2/V4 (with dense) and V3 (leaves only)
    Xtr_full = assemble_X(Xtr_leaf, num_ti, te_ti)
    Xva_full = assemble_X(Xva_leaf, num_va, te_va)
    Xte_full = assemble_X(Xte_leaf, num_te, te_te)
    Xtr_lo = Xtr_leaf.copy()
    Xva_lo = Xva_leaf.copy()
    del Xtr_leaf, Xva_leaf, Xte_leaf, num_ti, num_va, num_te, te_ti, te_va, te_te
    gc.collect()
    print(f"  X_full shape: {Xtr_full.shape}  nnz={Xtr_full.nnz}",
          flush=True)
    print(f"  X_leaves_only shape: {Xtr_lo.shape}  nnz={Xtr_lo.nnz}",
          flush=True)

    results = []

    def run_variant(name, lr, Xtr, Xva, y_tr, y_va):
        t = time.time()
        lr.fit(Xtr, y_tr)
        proba = lr.predict_proba(Xva)[:, 1]
        auc = float(roc_auc_score(y_va, proba))
        wall = time.time() - t
        niter = getattr(lr, "n_iter_", None)
        if niter is not None and len(niter) > 0:
            niter = int(niter[0]) if hasattr(niter[0], "__int__") else niter[0]
        gap_bp = (cb_auc - auc) * 1e4
        print(f"  {name:35s}: AUC={auc:.5f}  "
              f"gap_vs_CB={gap_bp:+.2f}bp  iter={niter}  "
              f"wall={wall:.1f}s", flush=True)
        results.append(dict(name=name, auc=auc, gap_vs_cb_bp=gap_bp,
                            iter=niter, wall_s=wall))

    # V1: lbfgs C=10 max_iter=2000 (more iters than main)
    run_variant("V1 lbfgs C=10 iter=2000 full",
                LogisticRegression(C=10, max_iter=2000, solver="lbfgs",
                                   penalty="l2", random_state=SEED),
                Xtr_full, Xva_full, y_tr_arr, y_va_arr)

    # V2: saga C=10 max_iter=500 — purpose-built for sparse
    run_variant("V2 saga  C=10 iter=500  full",
                LogisticRegression(C=10, max_iter=500, solver="saga",
                                   penalty="l2", random_state=SEED,
                                   n_jobs=-1),
                Xtr_full, Xva_full, y_tr_arr, y_va_arr)

    # V3: saga C=10 max_iter=500 — leaves only (no scale-mismatched dense)
    run_variant("V3 saga  C=10 iter=500  leaves-only",
                LogisticRegression(C=10, max_iter=500, solver="saga",
                                   penalty="l2", random_state=SEED,
                                   n_jobs=-1),
                Xtr_lo, Xva_lo, y_tr_arr, y_va_arr)

    # V4: liblinear C=10 — coord descent (slow but gold standard convergence)
    run_variant("V4 liblinear C=10 iter=200 full",
                LogisticRegression(C=10, max_iter=200, solver="liblinear",
                                   penalty="l2", random_state=SEED),
                Xtr_full, Xva_full, y_tr_arr, y_va_arr)

    print(f"\n=== Diag summary ===", flush=True)
    print(f"  CB ref AUC: {cb_auc:.5f}", flush=True)
    print(f"  best variant: {max(results, key=lambda r: r['auc'])['name']}",
          flush=True)
    print(f"  best AUC:     {max(r['auc'] for r in results):.5f}",
          flush=True)
    print(f"  total wall:   {(time.time()-t0_total)/60:.1f} min",
          flush=True)

    AUDIT.mkdir(exist_ok=True)
    Path(AUDIT / "2026-05-22-phaseA-diag.json").write_text(
        json.dumps(dict(cb_auc=cb_auc, variants=results), indent=2))


if __name__ == "__main__":
    main()
